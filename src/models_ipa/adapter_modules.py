import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F
from typing import List
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLInstructPix2PixPipeline,
    StableDiffusionInstructPix2PixPipeline,
)
from PIL import Image
from .ipa_utils import is_torch2_available

if is_torch2_available():
    from .attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from .attention_processor import IPAttnProcessor, AttnProcessor

from diffusers.loaders import LoraLoaderMixin
from diffusers.models.lora import LoRALinearLayer
from diffusers.models.unet_2d_blocks import DownBlock2D


# from .pipeline_stable_diffusion_xl_t2i_edit import StableDiffusionXLText2ImageAndEditPipeline
# from .pipeline_stable_diffusion_t2i_edit import StableDiffusionText2ImageAndEditPipeline


class IPAdapterSD(nn.Module):

    def __init__(self, unet, resampler) -> None:
        super().__init__()
        self.unet = unet
        self.resampler = resampler
        self.set_ip_adapter()
        self.set_trainable()

    def set_ip_adapter(self):
        attn_procs = {}
        unet_sd = self.unet.state_dict()
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights)
        self.unet.set_attn_processor(attn_procs)
        self.adapter = torch.nn.ModuleList(self.unet.attn_processors.values())

    def set_trainable(self):
        self.unet.requires_grad_(False)
        self.resampler.requires_grad_(True)
        self.adapter.requires_grad_(True)

    def params_to_opt(self):
        return itertools.chain(self.resampler.parameters(), self.adapter.parameters())

    def forward(self, noisy_latents, timesteps, image_embeds, text_embeds, noise):

        image_embeds = self.resampler(image_embeds)
        # image_embeds = image_embeds.to(dtype=text_embeds.dtype)

        text_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        # Predict the noise residual and compute loss
        noise_pred = self.unet(noisy_latents, timesteps, text_embeds).sample

        # if noise is not None:
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        # else:
        #     loss = torch.tensor(0.0, device=noisy_latents)

        return {'total_loss': loss, 'noise_pred': noise_pred}

    def encode_image_embeds(self, image_embeds):
        dtype = image_embeds.dtype
        image_embeds = self.resampler(image_embeds)
        image_embeds = image_embeds.to(dtype=dtype)
        return image_embeds

    @classmethod
    def from_pretrained(cls,
                        unet,
                        resampler,
                        pretrained_model_path=None,
                        pretrained_resampler_path=None,
                        pretrained_adapter_path=None):
        model = cls(unet=unet, resampler=resampler)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        if pretrained_resampler_path is not None:
            ckpt = torch.load(pretrained_resampler_path, map_location='cpu')
            missing, unexpected = model.resampler.load_state_dict(ckpt, strict=True)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        if pretrained_adapter_path is not None:
            ckpt = torch.load(pretrained_adapter_path, map_location='cpu')
            missing, unexpected = model.adapter.load_state_dict(ckpt, strict=True)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        return model

    @classmethod
    def from_pretrained_legacy(cls, unet, resampler, pretrained_model_path=None):
        model = cls(unet=unet, resampler=resampler)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            ckpt_image_proj = {}
            ckpt_ip_layers = {}

            for key, value in ckpt.items():
                if key.startswith('image_proj_model'):
                    new_key = key.replace('image_proj_model.', '')
                    ckpt_image_proj[new_key] = value
                elif key.startswith('adapter_modules.'):
                    new_key = key.replace('adapter_modules.', '')
                    ckpt_ip_layers[new_key] = value

            missing, unexpected = model.resampler.load_state_dict(ckpt_image_proj, strict=True)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
            missing, unexpected = model.adapter.load_state_dict(ckpt_ip_layers, strict=True)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))

        return model


class IPAdapterSDPipe(nn.Module):

    def __init__(
            self,
            ip_adapter,
            discrete_model,
            vae,
            visual_encoder,
            text_encoder,
            tokenizer,
            scheduler,
            image_transform,
            device,
            dtype,
    ) -> None:
        super().__init__()

        self.ip_adapter = ip_adapter
        self.vae = vae
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.image_transform = image_transform
        self.discrete_model = discrete_model
        self.device = device
        self.dtype = dtype

        self.sd_pipe = StableDiffusionPipeline(vae=vae,
                                               text_encoder=text_encoder,
                                               tokenizer=tokenizer,
                                               unet=ip_adapter.unet,
                                               scheduler=scheduler,
                                               safety_checker=None,
                                               feature_extractor=None,
                                               requires_safety_checker=False)

    def set_scale(self, scale):
        for attn_processor in self.sd_pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    @torch.inference_mode()
    def get_image_embeds(self, image_pil=None, image_tensor=None, return_negative=True):
        assert int(image_pil is not None) + int(image_tensor is not None) == 1
        if image_pil is not None:
            image_tensor = self.image_transform(image_pil).unsqueeze(0).to(self.device, dtype=self.dtype)
        if return_negative:
            image_tensor_neg = torch.zeros_like(image_tensor)
            image_tensor = torch.cat([image_tensor, image_tensor_neg], dim=0)
        with torch.cuda.amp.autocast(dtype=self.dtype):
            image_embeds = self.visual_encoder(image_tensor)
            image_embeds = self.discrete_model.encode_image_embeds(image_embeds)
        image_embeds = self.ip_adapter.encode_image_embeds(image_embeds)

        if return_negative:
            # bz = image_embeds.shape[0]
            # image_embeds_neg = image_embeds[bz // 2:]
            # image_embeds = image_embeds[0:bz // 2]
            image_embeds, image_embeds_neg = image_embeds.chunk(2)
        else:
            image_embeds_neg = None

        return image_embeds, image_embeds_neg

    def generate(self,
                 image_pil=None,
                 image_tensor=None,
                 prompt=None,
                 negative_prompt=None,
                 scale=1.0,
                 num_samples=1,
                 seed=42,
                 guidance_scale=7.5,
                 num_inference_steps=30,
                 **kwargs):
        self.set_scale(scale)
        assert int(image_pil is not None) + int(image_tensor is not None) == 1

        if image_pil is not None:
            assert isinstance(image_pil, Image.Image)
            num_prompts = 1
        else:
            num_prompts = image_tensor.shape[0]

        if prompt is None:
            # prompt = "best quality, high quality"
            prompt = ""
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            image_pil=image_pil,
            image_tensor=image_tensor,
            return_negative=True,
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = self.sd_pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.sd_pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


def compute_time_ids(original_size, crops_coords_top_left, target_resolution):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    target_size = (target_resolution, target_resolution)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    # add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
    return add_time_ids


class SDXLAdapter(nn.Module):

    def __init__(self, unet, resampler, full_ft=False) -> None:
        super().__init__()
        self.unet = unet
        self.resampler = resampler
        self.full_ft = full_ft
        self.set_trainable_v2()
        # self.set_adapter()

    #     self.set_trainable()

    # def set_adapter(self):

    #     adapter = []
    #     for name, module in self.unet.named_modules():
    #         if name.endswith('to_k') or name.endswith('to_v'):
    #             if module is not None:
    #                 adapter.append(module)

    #     self.adapter = torch.nn.ModuleList(adapter)
    #     print(f'adapter: {self.adapter}')

    # def set_trainable(self):
    #     self.unet.requires_grad_(False)
    #     self.resampler.requires_grad_(True)
    #     self.adapter.requires_grad_(True)

    def set_trainable_v2(self):
        self.resampler.requires_grad_(True)
        adapter_parameters = []
        if self.full_ft:
            self.unet.requires_grad_(True)
            adapter_parameters.extend(self.unet.parameters())
        else:
            self.unet.requires_grad_(False)
            for name, module in self.unet.named_modules():
                if name.endswith('to_k') or name.endswith('to_v'):
                    if module is not None:
                        adapter_parameters.extend(module.parameters())
        self.adapter_parameters = adapter_parameters
        for param in self.adapter_parameters:
            param.requires_grad_(True)

    # def params_to_opt(self):
    #     return itertools.chain(self.resampler.parameters(), self.adapter.parameters())
    def params_to_opt(self):
        return itertools.chain(self.resampler.parameters(), self.adapter_parameters)

    def forward(self, noisy_latents, timesteps, image_embeds, text_embeds, noise, time_ids):

        image_embeds, pooled_image_embeds = self.resampler(image_embeds)

        unet_added_conditions = {"time_ids": time_ids, 'text_embeds': pooled_image_embeds}

        noise_pred = self.unet(noisy_latents, timesteps, image_embeds, added_cond_kwargs=unet_added_conditions).sample

        # if noise is not None:
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        # else:
        #     loss = torch.tensor(0.0, device=noisy_latents)

        return {'total_loss': loss, 'noise_pred': noise_pred}

    def encode_image_embeds(self, image_embeds):
        image_embeds, pooled_image_embeds = self.resampler(image_embeds)

        return image_embeds, pooled_image_embeds

    @classmethod
    def from_pretrained(cls, unet, resampler, pretrained_model_path=None, **kwargs):
        model = cls(unet=unet, resampler=resampler, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        return model

    def init_pipe(self,
                  vae,
                  scheduler,
                  visual_encoder,
                  image_transform,
                  discrete_model=None,
                  dtype=torch.float16,
                  device='cuda'):
        self.device = device
        self.dtype = dtype
        sdxl_pipe = StableDiffusionXLPipeline(tokenizer=None,
                                              tokenizer_2=None,
                                              text_encoder=None,
                                              text_encoder_2=None,
                                              vae=vae,
                                              unet=self.unet,
                                              scheduler=scheduler)

        self.sdxl_pipe = sdxl_pipe  # .to(self.device, dtype=self.dtype)
        # print(sdxl_pipe.text_encoder_2, sdxl_pipe.text_encoder)

        self.visual_encoder = visual_encoder.to(self.device, dtype=self.dtype)
        if discrete_model is not None:
            self.discrete_model = discrete_model.to(self.device, dtype=self.dtype)
        else:
            self.discrete_model = None
        self.image_transform = image_transform

    @torch.inference_mode()
    def get_image_embeds(self,
                         image_pil=None,
                         image_tensor=None,
                         image_embeds=None,
                         return_negative=True,
                         image_size=448
                         ):
        assert int(image_pil is not None) + int(image_tensor is not None) + int(image_embeds is not None) == 1

        if image_pil is not None:
            image_tensor = self.image_transform(image_pil).unsqueeze(0).to(self.device, dtype=self.dtype)

        if image_tensor is not None:
            if return_negative:
                image_tensor_neg = torch.zeros_like(image_tensor)
                image_tensor = torch.cat([image_tensor, image_tensor_neg], dim=0)

            image_embeds = self.visual_encoder(image_tensor)
        elif return_negative:
            image_tensor_neg = torch.zeros(
                1, 3,
                image_size, image_size
            ).to(
                image_embeds.device, dtype=image_embeds.dtype
            )
            image_embeds_neg = self.visual_encoder(image_tensor_neg)
            image_embeds = torch.cat([image_embeds, image_embeds_neg], dim=0)

        if self.discrete_model is not None:
            image_embeds = self.discrete_model.encode_image_embeds(image_embeds)
        image_embeds, pooled_image_embeds = self.encode_image_embeds(image_embeds)

        if return_negative:
            image_embeds, image_embeds_neg = image_embeds.chunk(2)
            pooled_image_embeds, pooled_image_embeds_neg = pooled_image_embeds.chunk(2)

        else:
            image_embeds_neg = None
            pooled_image_embeds_neg = None

        return image_embeds, image_embeds_neg, pooled_image_embeds, pooled_image_embeds_neg

    def generate(self,
                 image_pil=None,
                 image_tensor=None,
                 image_embeds=None,
                 seed=42,
                 height=1024,
                 width=1024,
                 guidance_scale=7.5,
                 num_inference_steps=30,
                 input_image_size=448,
                 **kwargs):
        if image_pil is not None:
            assert isinstance(image_pil, Image.Image)

        image_prompt_embeds, uncond_image_prompt_embeds, pooled_image_prompt_embeds, \
            pooled_uncond_image_prompt_embeds = self.get_image_embeds(
            image_pil=image_pil,
            image_tensor=image_tensor,
            image_embeds=image_embeds,
            return_negative=True,
            image_size=input_image_size,
        )
        # print(image_prompt_embeds.shape, pooled_image_prompt_embeds.shape)
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        images = self.sdxl_pipe(
            prompt_embeds=image_prompt_embeds,
            negative_prompt_embeds=uncond_image_prompt_embeds,
            pooled_prompt_embeds=pooled_image_prompt_embeds,
            negative_pooled_prompt_embeds=pooled_uncond_image_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            **kwargs,
        ).images

        return images


class SDXLText2ImageAndEditAdapter(nn.Module):

    def __init__(self, unet, resampler, lora_rank=16, fully_ft=False) -> None:
        super().__init__()

        self.unet = unet
        self.resampler = resampler
        self.lora_rank = lora_rank

        if fully_ft:
            self.set_fully_trainable()
        else:
            self.set_adapter()

    def set_adapter(self):
        self.unet.requires_grad_(False)
        adapter_parameters = []

        in_channels = 8
        out_channels = self.unet.conv_in.out_channels
        self.unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = nn.Conv2d(in_channels, out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride,
                                    self.unet.conv_in.padding)

            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = new_conv_in
        self.unet.conv_in.requires_grad_(True)
        print('Make conv_in trainable.')
        adapter_parameters.extend(self.unet.conv_in.parameters())

        for name, module in self.unet.named_modules():
            if isinstance(module, DownBlock2D):
                module.requires_grad_(True)
                adapter_parameters.extend(module.parameters())
                print('Make DownBlock2D trainable.')

        for attn_processor_name, attn_processor in self.unet.attn_processors.items():
            # Parse the attention module.
            attn_module = self.unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)

            # Set the `lora_layer` attribute of the attention-related matrices.
            attn_module.to_q.set_lora_layer(
                LoRALinearLayer(in_features=attn_module.to_q.in_features,
                                out_features=attn_module.to_q.out_features,
                                rank=self.lora_rank))
            # attn_module.to_k.set_lora_layer(
            #     LoRALinearLayer(in_features=attn_module.to_k.in_features,
            #                     out_features=attn_module.to_k.out_features,
            #                     rank=self.lora_rank))
            # attn_module.to_v.set_lora_layer(
            #     LoRALinearLayer(in_features=attn_module.to_v.in_features,
            #                     out_features=attn_module.to_v.out_features,
            #                     rank=self.lora_rank))
            attn_module.to_out[0].set_lora_layer(
                LoRALinearLayer(
                    in_features=attn_module.to_out[0].in_features,
                    out_features=attn_module.to_out[0].out_features,
                    rank=self.lora_rank,
                ))

            attn_module.to_k.requires_grad_(True)
            attn_module.to_v.requires_grad_(True)

            adapter_parameters.extend(attn_module.to_q.lora_layer.parameters())
            adapter_parameters.extend(attn_module.to_k.parameters())
            adapter_parameters.extend(attn_module.to_v.parameters())
            adapter_parameters.extend(attn_module.to_out[0].lora_layer.parameters())

        self.adapter_parameters = adapter_parameters

    def set_fully_trainable(self):

        in_channels = 8
        out_channels = self.unet.conv_in.out_channels
        self.unet.register_to_config(in_channels=in_channels)
        with torch.no_grad():
            new_conv_in = nn.Conv2d(in_channels, out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride,
                                    self.unet.conv_in.padding)

            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = new_conv_in

        self.unet.requires_grad_(True)
        self.adapter_parameters = self.unet.parameters()

    def params_to_opt(self):
        return itertools.chain(self.resampler.parameters(), self.adapter_parameters)

    def forward(self, noisy_latents, timesteps, image_embeds, text_embeds, noise, time_ids, pooled_text_embeds=None):

        text_embeds, pooled_text_embeds = self.resampler(text_embeds, pooled_text_embeds=pooled_text_embeds)
        unet_added_conditions = {"time_ids": time_ids, 'text_embeds': pooled_text_embeds}

        noise_pred = self.unet(noisy_latents, timesteps, text_embeds, added_cond_kwargs=unet_added_conditions).sample

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        return {'total_loss': loss, 'noise_pred': noise_pred}

    def encode_text_embeds(self, text_embeds, pooled_text_embeds=None):
        text_embeds, pooled_text_embeds = self.resampler(text_embeds, pooled_text_embeds=pooled_text_embeds)

        return text_embeds, pooled_text_embeds

    @classmethod
    def from_pretrained(cls, unet, resampler, pretrained_model_path=None, **kwargs):
        model = cls(unet=unet, resampler=resampler, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        return model

    def init_pipe(self,
                  vae,
                  scheduler,
                  text_encoder,
                  text_encoder_2,
                  tokenizer,
                  tokenizer_2,
                  dtype=torch.float16,
                  device='cuda'):
        self.device = device
        self.dtype = dtype

        sdxl_pipe = StableDiffusionXLText2ImageAndEditPipeline(
            tokenizer=None,
            tokenizer_2=None,
            text_encoder=None,
            text_encoder_2=None,
            vae=vae,
            unet=self.unet,
            scheduler=scheduler,
        )

        self.sdxl_pipe = sdxl_pipe
        self.sdxl_pipe.to(device, dtype=dtype)

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

    @torch.inference_mode()
    def get_text_embeds(self, prompt=None, negative_prompt='', text_embeds=None):
        assert int(prompt is not None) + int(text_embeds is not None) == 1

        if prompt is not None:
            text_input_ids = self.tokenizer([prompt, negative_prompt],
                                            max_length=self.tokenizer.model_max_length,
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt").input_ids
            text_input_ids_2 = self.tokenizer_2([prompt, negative_prompt],
                                                max_length=self.tokenizer.model_max_length,
                                                padding="max_length",
                                                truncation=True,
                                                return_tensors="pt").input_ids
            encoder_output = self.text_encoder(text_input_ids.to(self.device), output_hidden_states=True)
            text_embeds = encoder_output.hidden_states[-2]

            encoder_output_2 = self.text_encoder_2(text_input_ids_2.to(self.device), output_hidden_states=True)
            pooled_text_embeds = encoder_output_2[0]
            text_embeds_2 = encoder_output_2.hidden_states[-2]

            text_embeds = torch.cat([text_embeds, text_embeds_2], dim=-1)
        else:
            text_input_ids = self.tokenizer(negative_prompt,
                                            max_length=self.tokenizer.model_max_length,
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt").input_ids
            text_input_ids_2 = self.tokenizer_2(negative_prompt,
                                                max_length=self.tokenizer.model_max_length,
                                                padding="max_length",
                                                truncation=True,
                                                return_tensors="pt").input_ids
            encoder_output = self.text_encoder(text_input_ids.to(self.device), output_hidden_states=True)
            text_embeds_neg = encoder_output.hidden_states[-2]

            encoder_output_2 = self.text_encoder_2(text_input_ids_2.to(self.device), output_hidden_states=True)
            text_embeds_neg_2 = encoder_output_2.hidden_states[-2]
            pooled_text_embeds = encoder_output_2[0]

            text_embeds_neg = torch.cat([text_embeds_neg, text_embeds_neg_2], dim=-1)

            text_embeds = torch.cat([text_embeds, text_embeds_neg], dim=0)

        text_embeds, pooled_text_embeds = self.encode_text_embeds(text_embeds, pooled_text_embeds=pooled_text_embeds)
        text_embeds, text_embeds_neg = text_embeds.chunk(2)
        pooled_text_embeds, pooled_text_embeds_neg = pooled_text_embeds.chunk(2)

        return text_embeds, text_embeds_neg, pooled_text_embeds, pooled_text_embeds_neg

    def generate(self,
                 prompt=None,
                 negative_prompt='',
                 image=None,
                 text_embeds=None,
                 seed=42,
                 height=1024,
                 width=1024,
                 guidance_scale=7.5,
                 num_inference_steps=30,
                 **kwargs):

        text_embeds, text_embeds_neg, pooled_text_embeds, pooled_text_embeds_neg = self.get_text_embeds(
            prompt=prompt, negative_prompt=negative_prompt, text_embeds=text_embeds)
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        images = self.sdxl_pipe(
            image=image,
            prompt_embeds=text_embeds,
            negative_prompt_embeds=text_embeds_neg,
            pooled_prompt_embeds=pooled_text_embeds,
            negative_pooled_prompt_embeds=pooled_text_embeds_neg,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            **kwargs,
        ).images

        return images


class SD21Text2ImageAndEditAdapter(SDXLText2ImageAndEditAdapter):

    def forward(self, noisy_latents, timesteps, image_embeds, text_embeds, noise):

        text_embeds, _ = self.resampler(text_embeds)
        # unet_added_conditions = {"time_ids": time_ids, 'text_embeds': pooled_text_embeds}

        noise_pred = self.unet(noisy_latents, timesteps, text_embeds).sample

        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        return {'total_loss': loss, 'noise_pred': noise_pred}

    def init_pipe(self,
                  vae,
                  scheduler,
                  text_encoder,
                  tokenizer,
                  feature_extractor,
                  dtype=torch.float16,
                  device='cuda'):
        self.device = device
        self.dtype = dtype

        sd_pipe = StableDiffusionText2ImageAndEditPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=self.unet,
            feature_extractor=feature_extractor,
            safety_checker=None,
            requires_safety_checker=False,
            scheduler=scheduler,
        )

        self.sd_pipe = sd_pipe
        self.sd_pipe.to(device, dtype=dtype)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    @torch.inference_mode()
    def get_text_embeds(self, prompt=None, negative_prompt='', text_embeds=None):
        assert int(prompt is not None) + int(text_embeds is not None) == 1

        if prompt is not None:
            text_input_ids = self.tokenizer([prompt, negative_prompt],
                                            max_length=self.tokenizer.model_max_length,
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt").input_ids
            encoder_output = self.text_encoder(text_input_ids.to(self.device))
            text_embeds = encoder_output[0]

        else:
            text_input_ids = self.tokenizer(negative_prompt,
                                            max_length=self.tokenizer.model_max_length,
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt").input_ids
            encoder_output = self.text_encoder(text_input_ids.to(self.device))
            text_embeds_neg = encoder_output[0]

            text_embeds = torch.cat([text_embeds, text_embeds_neg], dim=0)

        text_embeds, _ = self.encode_text_embeds(text_embeds)
        text_embeds, text_embeds_neg = text_embeds.chunk(2)

        return text_embeds, text_embeds_neg

    def generate(self,
                 prompt=None,
                 negative_prompt='',
                 image=None,
                 text_embeds=None,
                 seed=42,
                 height=1024,
                 width=1024,
                 guidance_scale=7.5,
                 num_inference_steps=30,
                 **kwargs):

        text_embeds, text_embeds_neg = self.get_text_embeds(
            prompt=prompt, negative_prompt=negative_prompt, text_embeds=text_embeds)
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        print(f'text_embeds: {text_embeds.shape}')
        print(f'text_embeds_neg: {text_embeds_neg.shape}')
        images = self.sd_pipe(
            image=image,
            prompt_embeds=text_embeds,
            negative_prompt_embeds=text_embeds_neg,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            **kwargs,
        ).images

        return images


class SDXLAdapterWithLatentImage(SDXLAdapter):
    def __init__(self, unet, resampler, full_ft=False, set_trainable_late=False) -> None:
        nn.Module.__init__(self)
        self.unet = unet
        self.resampler = resampler
        self.full_ft = full_ft
        if not set_trainable_late:
            self.set_trainable()

    def set_trainable(self):
        self.resampler.requires_grad_(True)
        adapter_parameters = []

        in_channels = 8
        out_channels = self.unet.conv_in.out_channels
        self.unet.register_to_config(in_channels=in_channels)
        self.unet.requires_grad_(False)
        with torch.no_grad():
            new_conv_in = nn.Conv2d(in_channels, out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride,
                                    self.unet.conv_in.padding)

            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = new_conv_in
        self.unet.conv_in.requires_grad_(True)

        if self.full_ft:
            self.unet.requires_grad_(True)
            adapter_parameters.extend(self.unet.parameters())
        else:
            adapter_parameters.extend(self.unet.conv_in.parameters())
            for name, module in self.unet.named_modules():
                if name.endswith('to_k') or name.endswith('to_v'):
                    if module is not None:
                        adapter_parameters.extend(module.parameters())
        self.adapter_parameters = adapter_parameters

    @classmethod
    def from_pretrained(cls, unet, resampler, pretrained_model_path=None, set_trainable_late=False, **kwargs):
        model = cls(unet=unet, resampler=resampler, set_trainable_late=set_trainable_late, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        if set_trainable_late:
            model.set_trainable()
        return model

    def init_pipe(self,
                  vae,
                  scheduler,
                  visual_encoder,
                  image_transform,
                  dtype=torch.float16,
                  device='cuda'):
        self.device = device
        self.dtype = dtype

        sdxl_pipe = StableDiffusionXLText2ImageAndEditPipeline(
            tokenizer=None,
            tokenizer_2=None,
            text_encoder=None,
            text_encoder_2=None,
            vae=vae,
            unet=self.unet,
            scheduler=scheduler,
        )

        self.sdxl_pipe = sdxl_pipe
        self.sdxl_pipe.to(device, dtype=dtype)
        self.discrete_model = None

        self.visual_encoder = visual_encoder.to(self.device, dtype=self.dtype)
        self.image_transform = image_transform

    def generate(self,
                 image_pil=None,
                 image_tensor=None,
                 image_embeds=None,
                 latent_image=None,
                 seed=42,
                 height=1024,
                 width=1024,
                 guidance_scale=7.5,
                 num_inference_steps=30,
                 input_image_size=448,
                 **kwargs):
        if image_pil is not None:
            assert isinstance(image_pil, Image.Image)

        image_prompt_embeds, uncond_image_prompt_embeds, \
            pooled_image_prompt_embeds, pooled_uncond_image_prompt_embeds = self.get_image_embeds(
            image_pil=image_pil,
            image_tensor=image_tensor,
            image_embeds=image_embeds,
            return_negative=True,
            image_size=input_image_size,
        )
        # print(image_prompt_embeds.shape, pooled_image_prompt_embeds.shape)
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        images = self.sdxl_pipe(
            image=latent_image,
            prompt_embeds=image_prompt_embeds,
            negative_prompt_embeds=uncond_image_prompt_embeds,
            pooled_prompt_embeds=pooled_image_prompt_embeds,
            negative_pooled_prompt_embeds=pooled_uncond_image_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            **kwargs,
        ).images

        return images
