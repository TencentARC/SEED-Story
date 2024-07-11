import torch
import torch.nn as nn
import pyrootutils
import torch.distributed as dist
import torch.nn.functional as F

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from src.train.dist_utils import concat_all_gather


def cosine_loss(rec, target):
    target = target / target.norm(dim=-1, keepdim=True)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    return rec_loss


def contrastive_loss(image_feats, text_feats, logit_scale):
    image_feats = image_feats.unsqueeze(1).contiguous()
    image_feats_all = concat_all_gather(image_feats)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
    text_feats_all = concat_all_gather(text_feats)  # [batch_size*num_gpu, embed_dim]

    sim_q2t = torch.matmul(image_feats.unsqueeze(1), text_feats_all.unsqueeze(-1)).squeeze()
    # [batch_size, batch_size*num_gpu, num_query_tokens]

    # image-text similarity: aggregate across all query tokens
    # sim_i2t, _ = sim_q2t.max(-1)
    # sim_i2t = sim_q2t.mean(-1)
    sim_i2t = sim_q2t
    sim_i2t = sim_i2t / logit_scale

    # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
    sim_t2q = torch.matmul(text_feats.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)).squeeze()

    # print(image_feats_all.shape, text_feat_all.shape, sim_q2t.shape, sim_t2q.shape)
    # text-image similarity: aggregate across all query tokens
    # sim_t2i, _ = sim_t2q.max(-1)
    # sim_t2i = sim_t2q.mean(-1)
    sim_t2i = sim_t2q
    sim_t2i = sim_t2i / logit_scale  # [batch_size, batch_size*num_gpu]

    rank = dist.get_rank()
    bs = image_feats.size(0)
    targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(image_feats.device)

    loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) +
                F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)) / 2

    i2t_acc = (sim_i2t.argmax(-1) == targets).sum() / len(sim_i2t)
    t2i_acc = (sim_t2i.argmax(-1) == targets).sum() / len(sim_t2i)

    return loss_itc, i2t_acc, t2i_acc


class DiscreteModleOnlyDistill(nn.Module):

    def __init__(self,
                 qformer,
                 quantizer,
                 distiller=None,
                 loss_type='cosine',
                 scale_commit_loss=1.0,
                 freeze_qformer=False) -> None:
        super().__init__()
        self.qformer = qformer
        self.quantizer = quantizer
        self.distiller = distiller
        self.loss_type = loss_type
        self.scale_commit_loss = scale_commit_loss

        self.freeze_qformer = freeze_qformer

        if freeze_qformer:
            self.qformer.requires_grad_(False)

    def forward(self, image_embeds, input_ids=None, text_attention_mask=None, text_embeds=None):
        if self.freeze_qformer:
            with torch.no_grad():
                qforemr_embeds = self.qformer(image_embeds=image_embeds)
        else:
            qforemr_embeds = self.qformer(image_embeds=image_embeds)

        quantizer_output = self.quantizer(qforemr_embeds)
        recon_embeds = self.distiller(quantizer_output['quant_embeds'])

        if self.loss_type == 'cosine':
            distill_loss = cosine_loss(recon_embeds, image_embeds)
        else:
            raise NotImplementedError

        total_loss = distill_loss + self.scale_commit_loss * \
                     quantizer_output['commit_loss']

        return {
            'total_loss': total_loss,
            'distill_loss': distill_loss,
            'commit_loss': quantizer_output['commit_loss'],
            'indices': quantizer_output['indices']
        }

    def encode_image_embeds(self, image_embeds):
        qforemr_embeds = self.qformer(image_embeds=image_embeds)
        quantizer_output = self.quantizer(qforemr_embeds)

        output_embeds = quantizer_output['quant_embeds']
        if self.distiller is not None:
            output_embeds = self.distiller(output_embeds)
        return output_embeds

    @classmethod
    def from_pretrained(cls, qformer, quantizer, distiller=None, pretrained_model_path=None, **kwargs):
        model = cls(qformer=qformer, quantizer=quantizer, distiller=distiller, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        return model


class DiscreteModleIdentity(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Identity()

    def forward(self, image_embeds, input_ids=None, text_attention_mask=None, text_embeds=None):
        return

    def encode_image_embeds(self, image_embeds):
        return self.model(image_embeds)


class DiscreteModleStageOneContrastive(nn.Module):

    def __init__(self, qformer, quantizer=None, distiller=None, projection_dim=1024,
                 image_cls_token_type='last') -> None:
        super().__init__()
        self.qformer = qformer
        self.quantizer = quantizer
        self.distiller = distiller
        self.image_cls_token_type = image_cls_token_type
        self.logit_scale = nn.Parameter(0.07 * torch.ones([]))
        self.image_proj = nn.Linear(qformer.perceiver.config.projection_dim, projection_dim, bias=False)
        self.text_proj = nn.Linear(qformer.perceiver.config.projection_dim, projection_dim, bias=False)

    def forward(self, image_embeds, input_ids=None, text_attention_mask=None, text_embeds=None):
        image_embeds = self.qformer(image_embeds=image_embeds)
        if self.image_cls_token_type == 'last':
            image_embeds = image_embeds[:, -1, :]
        else:
            raise NotImplementedError

        text_embeds = self.qformer(input_ids=input_ids, text_attention_mask=text_attention_mask)
        text_embeds = text_embeds[:, 0, :]

        image_embeds = F.normalize(self.image_proj(image_embeds), dim=-1)
        text_embeds = F.normalize(self.text_proj(text_embeds), dim=-1)

        contrast_loss, i2t_acc, t2i_acc = contrastive_loss(image_feats=image_embeds,
                                                           text_feats=text_embeds,
                                                           logit_scale=self.logit_scale)

        return {
            'total_loss': contrast_loss,
            'i2t_acc': i2t_acc,
            't2i_acc': t2i_acc,
        }

    def encode_image_embeds(self, image_embeds):
        image_embeds = self.qformer(image_embeds=image_embeds)

        return image_embeds

    @classmethod
    def from_pretrained(cls, qformer, quantizer, distiller=None, pretrained_model_path=None, **kwargs):
        model = cls(qformer=qformer, quantizer=quantizer, distiller=distiller, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        return model


class DiscreteModleStageTwoContrastiveDistill(nn.Module):

    def __init__(self,
                 qformer,
                 quantizer=None,
                 distiller=None,
                 contrast_head=None,
                 projection_dim=1024,
                 distill_loss_type='cosine',
                 freeze_qformer=True,
                 image_cls_token_type='last',
                 scale_commit_loss=1.0,
                 scale_contrast_loss=1.0,
                 scale_distill_loss=1.0) -> None:
        super().__init__()
        self.qformer = qformer
        self.quantizer = quantizer
        self.distiller = distiller
        self.contrast_head = contrast_head
        self.distill_loss_type = distill_loss_type
        self.image_cls_token_type = image_cls_token_type
        if self.contrast_head is not None:
            self.logit_scale = nn.Parameter(0.07 * torch.ones([]))
            self.image_proj = nn.Linear(contrast_head.perceiver.config.projection_dim, projection_dim, bias=False)
            self.text_proj = nn.Linear(contrast_head.perceiver.config.projection_dim, projection_dim, bias=False)

        self.freeze_qformer = freeze_qformer
        if freeze_qformer:
            self.qformer.requires_grad_(False)

        self.scale_commit_loss = scale_commit_loss
        self.scale_contrast_loss = scale_contrast_loss
        self.scale_distill_loss = scale_distill_loss

    def forward(self, image_embeds, input_ids=None, text_attention_mask=None, text_embeds=None):
        if self.freeze_qformer:
            with torch.no_grad():
                qforemr_embeds = self.qformer(image_embeds=image_embeds)
        else:
            qforemr_embeds = self.qformer(image_embeds=image_embeds)

        quantizer_output = self.quantizer(qforemr_embeds)

        output_state = {}
        output_state['indices'] = quantizer_output['indices']
        output_state['commit_loss'] = quantizer_output['commit_loss']
        output_state['total_loss'] = self.scale_commit_loss * quantizer_output['commit_loss']
        if self.distiller is not None:
            recon_embeds = self.distiller(quantizer_output['quant_embeds'])

            if self.distill_loss_type == 'cosine':
                distill_loss = cosine_loss(recon_embeds, image_embeds)
            else:
                raise NotImplementedError

            output_state['distill_loss'] = distill_loss
            output_state['total_loss'] += self.scale_distill_loss * distill_loss

        if self.contrast_head is not None:
            text_embeds = self.qformer(input_ids=input_ids, text_attention_mask=text_attention_mask)
            text_embeds = text_embeds[:, 0, :]

            image_embeds = self.contrast_head(quantizer_output['quant_embeds'])
            if self.image_cls_token_type == 'last':
                image_embeds = image_embeds[:, -1, :]
            else:
                raise NotImplementedError

            image_embeds = F.normalize(self.image_proj(image_embeds), dim=-1)
            text_embeds = F.normalize(self.text_proj(text_embeds), dim=-1)

            contrast_loss, i2t_acc, t2i_acc = contrastive_loss(image_feats=image_embeds,
                                                               text_feats=text_embeds,
                                                               logit_scale=self.logit_scale)
            output_state['contrast_loss'] = contrast_loss
            output_state['total_loss'] += self.scale_contrast_loss * contrast_loss
            output_state['i2t_acc'] = i2t_acc
            output_state['t2i_acc'] = t2i_acc

        return output_state

    def encode_image_embeds(self, image_embeds):
        pass

    @classmethod
    def from_pretrained(cls, qformer, quantizer, distiller=None, contrast_head=None, pretrained_model_path=None,
                        **kwargs):
        model = cls(qformer=qformer, quantizer=quantizer, distiller=distiller, contrast_head=contrast_head, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        return model


class DiscreteModleDistillWithDoubleContrastive(nn.Module):

    def __init__(
            self,
            qformer,
            quantizer=None,
            distiller=None,
            contrast_head=None,
            projection_dim=1024,
            distill_loss_type='cosine',
            share_contrast_head=True,  # share contrastive head with distiller
            quantize_cls_token=False,
            rec_qformer=False,
            has_contrast=False,
            freeze_qformer=False,
            scale_commit_loss=1.0,
            scale_contrast_loss=1.0,
            scale_distill_loss=1.0) -> None:
        super().__init__()
        self.qformer = qformer
        self.quantizer = quantizer
        self.distiller = distiller
        self.contrast_head = contrast_head
        self.distill_loss_type = distill_loss_type
        self.quantize_cls_token = quantize_cls_token

        self.rec_qformer = rec_qformer
        self.has_contrast = has_contrast

        if freeze_qformer:
            self.qformer.requires_grad_(False)
        else:
            self.logit_scale_qformer = nn.Parameter(0.07 * torch.ones([]))
            self.image_proj_qformer = nn.Linear(qformer.perceiver.config.projection_dim, projection_dim, bias=False)
            self.text_proj_qformer = nn.Linear(qformer.perceiver.config.projection_dim, projection_dim, bias=False)
            self.cls_norm_qformer = nn.LayerNorm(qformer.perceiver.config.projection_dim)

        if self.contrast_head is not None:
            self.logit_scale_head = nn.Parameter(0.07 * torch.ones([]))
            self.image_proj_head = nn.Linear(contrast_head.perceiver.config.projection_dim, projection_dim, bias=False)
            self.text_proj_head = nn.Linear(qformer.perceiver.config.projection_dim, projection_dim, bias=False)
            self.cls_norm_head = nn.LayerNorm(contrast_head.perceiver.config.projection_dim)

        if share_contrast_head and distiller is not None:
            self.logit_scale_head = nn.Parameter(0.07 * torch.ones([]))
            self.image_proj_head = nn.Linear(distiller.perceiver.config.projection_dim, projection_dim, bias=False)
            self.text_proj_head = nn.Linear(qformer.perceiver.config.projection_dim, projection_dim, bias=False)
            self.cls_norm_head = nn.LayerNorm(distiller.perceiver.config.projection_dim)

        self.scale_commit_loss = scale_commit_loss
        self.scale_contrast_loss = scale_contrast_loss
        self.scale_distill_loss = scale_distill_loss
        self.share_contrast_head = share_contrast_head
        self.freeze_qformer = freeze_qformer
        assert int(self.share_contrast_head) + int(contrast_head is not None) <= 1

    def forward(self, image_embeds, input_ids=None, text_attention_mask=None, text_embeds=None):

        if self.freeze_qformer:
            with torch.no_grad():
                qforemr_embeds = self.qformer(image_embeds=image_embeds)
        else:
            qforemr_embeds = self.qformer(image_embeds=image_embeds)
        qforemr_cls_embeds = qforemr_embeds[:, -1, :]

        if not self.quantize_cls_token:
            qforemr_embeds = qforemr_embeds[:, :-1, :]

        if self.has_contrast:
            text_embeds = self.qformer(input_ids=input_ids, text_attention_mask=text_attention_mask)
            text_cls_embeds = text_embeds[:, 0, :]

        output_state = {}
        output_state['total_loss'] = 0.0

        if not self.freeze_qformer and self.has_contrast:
            qforemr_cls_embeds = self.cls_norm_qformer(qforemr_cls_embeds)
            qformer_image_embeds = F.normalize(self.image_proj_qformer(qforemr_cls_embeds), dim=-1)
            qformer_text_embeds = F.normalize(self.text_proj_qformer(text_cls_embeds), dim=-1)

            qformer_contrast_loss, \
                qformer_i2t_acc, \
                qformer_t2i_acc = contrastive_loss(image_feats=qformer_image_embeds,
                                                    text_feats=qformer_text_embeds,
                                                    logit_scale=self.logit_scale_qformer)
            output_state['qformer_contrast_loss'] = qformer_contrast_loss
            output_state['total_loss'] += self.scale_contrast_loss * qformer_contrast_loss
            output_state['qformer_i2t_acc'] = qformer_i2t_acc
            output_state['qformer_t2i_acc'] = qformer_t2i_acc

        if self.quantizer is not None and self.distiller is not None:
            quantizer_output = self.quantizer(qforemr_embeds)

            recon_embeds = self.distiller(quantizer_output['quant_embeds'])
            if self.share_contrast_head:
                contrast_head_cls_embeds = recon_embeds[:, -1, :]
                contrast_head_cls_embeds = self.cls_norm_head(contrast_head_cls_embeds)
                recon_embeds = recon_embeds[:, :-1, :]
            if self.contrast_head is not None:
                contrast_head_embeds = self.contrast_head(quantizer_output['quant_embeds'])
                contrast_head_cls_embeds = contrast_head_embeds[:, -1, :]
                contrast_head_cls_embeds = self.cls_norm_head(contrast_head_cls_embeds)

            output_state['indices'] = quantizer_output['indices']
            output_state['commit_loss'] = quantizer_output['commit_loss']
            output_state['total_loss'] += self.scale_commit_loss * quantizer_output['commit_loss']

            if self.rec_qformer:
                target_embeds = qforemr_embeds
            else:
                target_embeds = image_embeds

            if self.distill_loss_type == 'cosine':
                distill_loss = cosine_loss(recon_embeds, target_embeds)
            else:
                raise NotImplementedError

            output_state['distill_loss'] = distill_loss
            output_state['total_loss'] += self.scale_distill_loss * distill_loss

            if self.contrast_head is not None or self.share_contrast_head:
                head_image_embeds = F.normalize(self.image_proj_head(contrast_head_cls_embeds), dim=-1)
                head_text_embeds = F.normalize(self.text_proj_head(text_cls_embeds), dim=-1)

                head_contrast_loss, head_i2t_acc, head_t2i_acc = contrastive_loss(image_feats=head_image_embeds,
                                                                                  text_feats=head_text_embeds,
                                                                                  logit_scale=self.logit_scale_head)
                output_state['head_contrast_loss'] = head_contrast_loss
                output_state['total_loss'] += self.scale_contrast_loss * head_contrast_loss
                output_state['head_i2t_acc'] = head_i2t_acc
                output_state['head_t2i_acc'] = head_t2i_acc

        return output_state

    def encode_image_embeds(self, image_embeds):
        qforemr_embeds = self.qformer(image_embeds=image_embeds)
        return qforemr_embeds

    @classmethod
    def from_pretrained(cls, qformer, quantizer=None, distiller=None, contrast_head=None, pretrained_model_path=None,
                        **kwargs):
        model = cls(qformer=qformer, quantizer=quantizer, distiller=distiller, contrast_head=contrast_head, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        return model

    @classmethod
    def from_pretrained_stage1_yuying(cls,
                                      qformer,
                                      quantizer=None,
                                      distiller=None,
                                      contrast_head=None,
                                      pretrained_model_path=None,
                                      **kwargs):
        model = cls(qformer=qformer, quantizer=quantizer, distiller=distiller, contrast_head=contrast_head, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            ckpt = ckpt['model']

            new_ckpt = {}
            new_ckpt['qformer.embed_module.query'] = ckpt['query_tokens'].squeeze(0)
            new_ckpt['qformer.norm.weight'] = ckpt['ln_vision.weight']
            new_ckpt['qformer.norm.bias'] = ckpt['ln_vision.bias']

            for key in ckpt.keys():
                if key.startswith('Qformer'):
                    new_key = key.replace('Qformer', 'qformer.perceiver')
                    new_ckpt[new_key] = ckpt[key]
            del ckpt
            missing, unexpected = model.load_state_dict(new_ckpt, strict=False)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
            print(missing)
            print(unexpected)
        return model
