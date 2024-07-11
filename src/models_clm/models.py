import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import LogitsProcessor, LogitsProcessorList
from .generation import AutoImageTokenGenerationProcessor
import torch.nn.functional as F

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'


def cosine_loss(rec, target):
    target = target / target.norm(dim=-1, keepdim=True)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    return rec_loss


class ContinuousLVLM(nn.Module):

    def __init__(self, llm, input_resampler, output_resampler, lm_loss_scale=1.0, rec_loss_scale=1.0) -> None:
        super().__init__()
        self.llm = llm
        self.input_resampler = input_resampler
        self.output_resampler = output_resampler
        self.lm_loss_scale = lm_loss_scale
        self.rec_loss_scale = rec_loss_scale

        # input_resampler.requires_grad_(False)
        # output_resampler.requires_grad_(False)

    def forward(self, input_ids, attention_mask, labels, image_embeds, embeds_gen_mask, embeds_cmp_mask, ids_gen_mask,
                ids_cmp_mask, return_recon_image_embeds=False):

        input_embeds = self.llm.get_input_embeddings()(input_ids)  # bz x seq_len x dim, 4 x 160 x 4096

        bz, sq, dim = input_embeds.shape

        if image_embeds is not None:
            image_embeds_lm = self.input_resampler(image_embeds)  # num_imgs_in_batch x nq x dim, 4 x 64 x 4096
            has_image = True
        else:
            image_embeds = torch.randn(bz, self.output_resampler.num_queries,
                                       self.output_resampler.embed_dim).to(input_embeds.device,
                                                                           dtype=input_embeds.dtype)
            image_embeds_lm = self.input_resampler(image_embeds)
            has_image = False

        has_image_input = has_image and embeds_cmp_mask.sum().item() > 0
        has_image_output = has_image and embeds_gen_mask.sum().item() > 0

        if has_image_input:
            input_embeds[ids_cmp_mask] = image_embeds_lm[embeds_cmp_mask].view(-1, dim)  # eg, 128 x 4096
            # zero_loss = 0.0
        else:
            min_bz = min(input_embeds.shape[0], image_embeds_lm.shape[0])
            input_embeds[:min_bz, :self.input_resampler.
            num_queries, :] = input_embeds[:min_bz, :self.input_resampler.
            num_queries, :] + 0.0 * image_embeds_lm[:min_bz, :, :]

        output_lm = self.llm(attention_mask=attention_mask,
                             inputs_embeds=input_embeds,
                             labels=labels,
                             output_hidden_states=True,
                             return_dict=True)
        lm_loss = output_lm['loss']

        last_hidden_state = output_lm.hidden_states[-1]  # 4 x 160 x 4096

        if has_image_output:
            target_embeds = image_embeds[embeds_gen_mask]  # num_imgs_gen_target x nq_in x dim_in, 2 x 256 x 4096
            num_imgs_for_rec = target_embeds.shape[0]
            output_image_embeds = last_hidden_state[ids_gen_mask].view(num_imgs_for_rec, -1,
                                                                       dim)  # 128 x 4096 -> 2 x 64 x 4096

            recon_image_embeds = self.output_resampler(output_image_embeds)  # 2 x 256 x 4096

            rec_loss = cosine_loss(recon_image_embeds, target_embeds)
        else:
            output_image_embeds = torch.randn(bz, self.input_resampler.num_queries,
                                              self.input_resampler.embed_dim).to(input_embeds.device,
                                                                                 dtype=input_embeds.dtype)
            recon_image_embeds = self.output_resampler(output_image_embeds)
            target_embeds = torch.randn(bz, self.output_resampler.num_queries,
                                        self.output_resampler.embed_dim).to(input_embeds.device,
                                                                            dtype=input_embeds.dtype)
            rec_loss = cosine_loss(recon_image_embeds, target_embeds) * 0.0

        total_loss = self.lm_loss_scale * lm_loss + self.rec_loss_scale * rec_loss

        if return_recon_image_embeds and has_image_output:
            return {'total_loss': total_loss, 'lm_loss': lm_loss, 'rec_loss': rec_loss,
                    'recon_image_embeds': recon_image_embeds}
        else:
            return {'total_loss': total_loss, 'lm_loss': lm_loss, 'rec_loss': rec_loss}

    def generate(self,
                 tokenizer,
                 prompt=None,
                 input_ids=None,
                 image_embeds=None,
                 embeds_cmp_mask=None,
                 ids_cmp_mask=None,
                 logits_processor=None,
                 num_img_gen_tokens=64,
                 temperature=0.7,
                 num_beams=1,
                 max_new_tokens=120,
                 top_p=0.5,
                 past_key_values=None,
                 # position_ids=None,
                 dtype=torch.float16,
                 device='cuda'):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
            logits_processor.append(
                AutoImageTokenGenerationProcessor(tokenizer=tokenizer, num_img_gen_tokens=num_img_gen_tokens))

        if prompt is not None:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)

        input_ids = input_ids.to(device=device)
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        bz, sq, dim = input_embeds.shape

        if image_embeds is not None:
            assert embeds_cmp_mask is not None and ids_cmp_mask is not None
            with torch.no_grad():
                image_embeds_lm = self.input_resampler(image_embeds)

            input_embeds[ids_cmp_mask] = image_embeds_lm[embeds_cmp_mask].view(-1, dim)

        generation_config = {
            'temperature': temperature,
            'num_beams': num_beams,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'do_sample': False
        }

        # generate_ids = self.llm.generate(input_ids=input_ids, **generation_config)
        output = self.llm.generate(input_ids=input_ids,
                                   inputs_embeds=input_embeds,
                                   output_hidden_states=True,
                                   return_dict_in_generate=True,
                                   logits_processor=logits_processor,
                                   past_key_values=past_key_values,
                                   # position_ids=position_ids,
                                   **generation_config)
        # self.llm.base_model.model.position_ids = self.llm.base_model.model.position_ids[:, :-2]

        output_past_key_values = self.llm.past_key_values
        generate_ids = output.sequences[0][input_ids.shape[1]:]
        generate_id_list = generate_ids.tolist()
        boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
        eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

        attn_weights = ()

        def merge_attn_weights(attn_weights):
            merged_attn_weights = attn_weights[0]

            # Iterate through the remaining attention weight tensors
            for i, attn_weight in enumerate(attn_weights[1:]):
                merged_attn_weights = F.pad(merged_attn_weights, (0, 1), "constant", float('nan'))
                # Concatenate the expanded tensor to the merged tensor along the kv_len dimension
                merged_attn_weights = torch.cat([merged_attn_weights, attn_weight], dim=1)

            return merged_attn_weights

        if output.attentions is not None:
            # for idx in [0, 1, 2, 9, 16, 23, 31]:
            for idx in range(32):
                attn_weights += (
                merge_attn_weights([output.attentions[j][idx] for j in range(len(output.attentions))]),)

        # for skip image multi turn kvcache
        last_hidden_states = torch.cat([hidden_state[-1] for hidden_state in output.hidden_states], dim=1)
        if past_key_values is None:
            last_hidden_states = last_hidden_states[0, input_ids.shape[1]:, :]
            eoi_indices = torch.where(generate_ids == eoi_token_id)[0].tolist()
        else:
            last_hidden_states = last_hidden_states[0, :, :]
            hidden_len = last_hidden_states.shape[0]
            eoi_indices = torch.where(output.sequences[0][-hidden_len:] == eoi_token_id)[0].tolist()

        num_gen_imgs = 1 if len(eoi_indices) > 0 else 0

        text_mask = torch.ones_like(generate_ids, dtype=torch.bool)
        has_img_output = num_gen_imgs > 0
        if has_img_output:
            img_gen_feats = []
            img_gen_feats.append(last_hidden_states[eoi_indices[-1] - num_img_gen_tokens:eoi_indices[-1]])
            text_mask[eoi_indices[-1] - num_img_gen_tokens:eoi_indices[-1]] = False

            # for eoi_idx in eoi_indices:
            #     img_gen_feats.append(last_hidden_states[eoi_idx - num_img_gen_tokens:eoi_idx])
            #     text_mask[eoi_idx - num_img_gen_tokens:eoi_idx] = False

            img_gen_feats = torch.stack(img_gen_feats)
            img_gen_feat = self.output_resampler(img_gen_feats)
        else:
            img_gen_feat = None

        text_mask[generate_ids == boi_token_id] = False
        # generate_ids = generate_ids[text_mask]
        generate_text = tokenizer.decode(generate_ids, skip_special_tokens=False)

        return {
            'text': generate_text,
            'generate_ids': generate_ids,
            'has_img_output': has_img_output,
            'img_gen_feat': img_gen_feat,
            'num_gen_imgs': num_gen_imgs,
            'attn_weights': attn_weights,
            'past_key_values': output_past_key_values
        }

    @classmethod
    def from_pretrained(cls, llm, input_resampler, output_resampler, pretrained_model_path=None, **kwargs):
        model = cls(llm=llm, input_resampler=input_resampler, output_resampler=output_resampler, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('agent model, missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        return model


class SEEDLLaMAAlignGeneration(nn.Module):

    def __init__(self, llm, output_resampler) -> None:
        super().__init__()

        self.llm = llm
        self.output_resampler = output_resampler
        # self.rec_loss_scale = rec_loss_scale

        self.llm.requires_grad_(False)

    def forward(self, input_ids, attention_mask, labels, image_embeds, embeds_gen_mask, embeds_cmp_mask, ids_gen_mask,
                ids_cmp_mask):

        input_embeds = self.llm.get_input_embeddings()(input_ids)  # bz x seq_len x dim, 4 x 160 x 4096

        bz, sq, dim = input_embeds.shape

        output_lm = self.llm(attention_mask=attention_mask,
                             inputs_embeds=input_embeds,
                             labels=labels,
                             output_hidden_states=True,
                             return_dict=True)

        last_hidden_state = output_lm.hidden_states[-1]  # 4 x 160 x 4096

        target_embeds = image_embeds[embeds_gen_mask]  # num_imgs_gen_target x nq_in x dim_in, 2 x 256 x 4096
        num_imgs_for_rec = target_embeds.shape[0]
        output_image_embeds = last_hidden_state[ids_gen_mask].view(num_imgs_for_rec, -1,
                                                                   dim)  # 128 x 4096 -> 2 x 64 x 4096

        recon_image_embeds = self.output_resampler(output_image_embeds)  # 2 x 256 x 4096

        rec_loss = cosine_loss(recon_image_embeds, target_embeds)

        return {'total_loss': rec_loss, 'rec_loss': rec_loss}

    @classmethod
    def from_pretrained(cls, llm, output_resampler, pretrained_model_path=None, **kwargs):
        model = cls(llm=llm, output_resampler=output_resampler, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('agent model, missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        return model

    def generate(self,
                 tokenizer,
                 input_ids=None,
                 temperature=0.7,
                 num_beams=1,
                 max_new_tokens=120,
                 num_img_gen_tokens=64,
                 top_p=0.5,
                 dtype=torch.float16,
                 device='cuda'):
        input_ids = input_ids.to(device=device)
        input_embeds = self.llm.get_input_embeddings()(input_ids)  # bz x seq_len x dim, 4 x 160 x 4096

        generation_config = {
            'temperature': temperature,
            'num_beams': num_beams,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'do_sample': False
        }
        output = self.llm.generate(input_ids=input_ids,
                                   inputs_embeds=input_embeds,
                                   output_hidden_states=True,
                                   return_dict_in_generate=True,
                                   **generation_config)

        generate_ids = output.sequences[0][input_ids.shape[1]:]
        generate_id_list = generate_ids.tolist()
        # boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
        eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

        # print('output ids: ', generate_ids, generate_ids.shape)
        # last_hidden_states = output.hidden_states[-1]

        last_hidden_states = torch.cat([hidden_state[-1] for hidden_state in output.hidden_states],
                                       dim=1)[:1, input_ids.shape[1]:, :]

        has_img_output = eoi_token_id in generate_id_list

        if has_img_output:
            # print(boi_token_id, generate_id_list, generate_id_list.index(boi_token_id))
            # boi_idx = generate_id_list.index(boi_token_id)
            eoi_idx = generate_id_list.index(eoi_token_id)
            print(len(generate_id_list), generate_id_list, eoi_idx)
            # print(generate_id_list[boi_idx + 1:boi_idx + 1 + num_img_gen_tokens])

            # img_gen_feat = last_hidden_states[:, eoi_idx - num_img_gen_tokens:eoi_idx]
            img_gen_feat = last_hidden_states[:, 0:eoi_idx]
            print('img_gen_feat', img_gen_feat.shape, last_hidden_states.shape, num_img_gen_tokens)
            img_gen_feat = self.output_resampler(img_gen_feat)

        else:
            img_gen_feat = None

        generate_text = tokenizer.decode(generate_ids, skip_special_tokens=False)
        # print('output keys: ', output.keys())

        return {'text': generate_text, 'has_img_output': has_img_output, 'img_gen_feat': img_gen_feat}
