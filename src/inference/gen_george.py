# flake8: noqa
import hydra
from omegaconf import OmegaConf
import torch
import os
import re
import pyrootutils
from PIL import Image, ImageDraw, ImageFont
import json
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, EulerDiscreteScheduler

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

device = 'cuda:0'
dtype = torch.float16
dtype_str = 'fp16'
num_img_in_tokens = 64
num_img_out_tokens = 64
instruction_prompt = '{instruction}'

tokenizer_cfg_path = 'configs/tokenizer/clm_llama_tokenizer.yaml'
image_transform_cfg_path = 'configs/processer/qwen_448_transform.yaml'
visual_encoder_cfg_path = 'configs/visual_tokenizer/qwen_vitg_448.yaml'

llm_cfg_path = 'configs/clm_models/llama2chat7b_lora.yaml'
agent_cfg_path = 'configs/clm_models/agent_7b_sft.yaml'

adapter_cfg_path = 'configs/detokenizer/detokenizer_sdxl_qwen_vit_adapted.yaml'
discrete_model_cfg_path = 'configs/discrete_model/discrete_identity.yaml'

diffusion_model_path = 'pretrained/stable-diffusion-xl-base-1.0'

save_dir = "output"

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg)

image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
image_transform = hydra.utils.instantiate(image_transform_cfg)

visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
visual_encoder.eval().to(device, dtype=dtype)
print('Init visual encoder done')

llm_cfg = OmegaConf.load(llm_cfg_path)
llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype_str)
print('Init llm done.')

agent_model_cfg = OmegaConf.load(agent_cfg_path)
agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)

agent_model.eval().to(device, dtype=dtype)
print('Init agent model Done')

noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
print('init vae')
vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(device, dtype=dtype)
print('init unet')
unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").to(device, dtype=dtype)

adapter_cfg = OmegaConf.load(adapter_cfg_path)
adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(device, dtype=dtype).eval()
print('Init adapter done')

discrete_model_cfg = OmegaConf.load(discrete_model_cfg_path)
discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(device).eval()
print('Init discrete model done')

adapter.init_pipe(vae=vae,
                  scheduler=noise_scheduler,
                  visual_encoder=visual_encoder,
                  image_transform=image_transform,
                  discrete_model=discrete_model,
                  dtype=dtype,
                  device=device)

print('Init adapter pipe done')
boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]


def read_jsonl_to_dict(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Each line is a valid JSON object
            json_object = json.loads(line)
            data.append(json_object)
    return data


# data
filename = 'data/json/val.jsonl'
image_root = 'data/image/george_full'
data = read_jsonl_to_dict(filename)
image_paths = [
    os.path.join(image_root, d['images'][0]) for d in data
]
questions = [
    d['captions'][0] for d in data
]


# texts = [
#     d['captions'][1:] for d in data
# ]


def add_subtitle(original_image, text):
    # Calculate the size of the new image
    text_height = 80  # Height of the black bar for the text
    new_image_size = (original_image.width, original_image.height + text_height)

    # Create a new image with a black background
    new_image = Image.new("RGB", new_image_size, "black")
    # Paste the original image onto the new image
    new_image.paste(original_image, (0, 0))

    # Prepare the new image for drawing
    draw = ImageDraw.Draw(new_image)

    # Specify the font size and font path
    font_size = 14  # Adjust font size as needed
    # font = ImageFont.truetype(font_path, font_size)

    # Manually split the text into two lines
    line1, line2 = text[:len(text) // 2], text[len(text) // 2:]

    # Update the position for the first line of text to ensure both lines are centered vertically
    text_position_line1 = (10, original_image.height + (text_height - font_size) // 2)

    # Define the text color
    text_color = "white"

    # Add the first line of text to the new image
    draw.text(text_position_line1, line1, fill=text_color)

    # Adjust the position for the second line of text, based on the height of the first line
    text_position_line2 = (10, text_position_line1[1] + font_size)

    # Add the second line of text to the new image
    draw.text(text_position_line2, line2, fill=text_color)

    return new_image


for j in range(len(image_paths)):
    image_path = image_paths[j]
    question = questions[j]
    image = Image.open(image_path).convert('RGB')

    save_folder = '{}/val_{}'.format(save_dir, j)

    os.makedirs(save_folder, exist_ok=True)

    init_image = add_subtitle(image, question)
    save_path = os.path.join(save_folder, '000start_image.jpg')
    init_image.save(save_path)

    agent_model.llm.base_model.model.use_kv_cache_head = False
    image_tensor = image_transform(image).unsqueeze(0).to(device, dtype=dtype)

    image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)]) + EOI_TOKEN

    prompt = instruction_prompt.format_map({'instruction': question + image_tokens})
    print(prompt)
    print('*' * 20)

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + input_ids

    boi_idx = input_ids.index(boi_token_id)
    eoi_idx = input_ids.index(eoi_token_id)

    input_ids = torch.tensor(input_ids).to(device, dtype=torch.long).unsqueeze(0)

    ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    ids_cmp_mask[0, boi_idx + 1:eoi_idx] = True
    embeds_cmp_mask = torch.tensor([True]).to(device, dtype=torch.bool)

    with torch.no_grad():
        image_embeds = visual_encoder(image_tensor)
    output = agent_model.generate(tokenizer=tokenizer,
                                  input_ids=input_ids,
                                  image_embeds=image_embeds,
                                  embeds_cmp_mask=embeds_cmp_mask,
                                  ids_cmp_mask=ids_cmp_mask,
                                  max_new_tokens=500,
                                  num_img_gen_tokens=num_img_out_tokens)
    text = re.sub(r'\s*<[^>]*>\s*', ' ', output['text']).strip()

    with open("{}/text.txt".format(save_folder), 'a+') as text_file:
        text_file.write(text + '\n')
    with open("{}/token.txt".format(save_folder), 'a+') as token_file:
        token_file.write("context token: {}\n".format(input_ids.shape))
    print(output['text'])
    print('*' * 20)

    story_len = 25
    window_size = 8
    text_id = 1
    while output['has_img_output'] and image_embeds.shape[0] < story_len:
        image_embeds_gen = output['img_gen_feat']
        images_gen = adapter.generate(image_embeds=output['img_gen_feat'], num_inference_steps=50)

        name = '{:02d}.jpg'.format(text_id)
        save_path = os.path.join(save_folder, name)

        # Open the generated image
        original_image = images_gen[0]
        ori_path = os.path.join(save_folder, 'ori_{:02d}.jpg'.format(text_id))
        original_image.save(ori_path)

        new_image = add_subtitle(original_image, text)
        # Save the modified image
        new_image.save(save_path)

        image_embeds = torch.cat((image_embeds, image_embeds_gen), dim=0)

        # image_embeds = torch.cat((image_embeds, image_embeds_gen), dim=0)

        if text_id >= story_len - 1:
            break

        prompt = prompt + text + image_tokens
        text_id += 1

        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        while image_embeds.shape[0] > window_size:
            eoi_prompt_idx = prompt.index(EOI_TOKEN)
            prompt = prompt[eoi_prompt_idx + len(EOI_TOKEN) + len('[INST]'):]
            image_embeds = image_embeds[1:]
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)

        print(prompt)
        print('*' * 20)

        input_ids = [tokenizer.bos_token_id] + input_ids

        boi_idx = torch.where(torch.tensor(input_ids) == boi_token_id)[0].tolist()
        eoi_idx = torch.where(torch.tensor(input_ids) == eoi_token_id)[0].tolist()

        input_ids = torch.tensor(input_ids).to(device, dtype=torch.long).unsqueeze(0)

        ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for i in range(image_embeds.shape[0]):
            ids_cmp_mask[0, boi_idx[i] + 1:eoi_idx[i]] = True
        embeds_cmp_mask = torch.tensor([True] * image_embeds.shape[0]).to(device, dtype=torch.bool)

        output = agent_model.generate(tokenizer=tokenizer,
                                      input_ids=input_ids,
                                      image_embeds=image_embeds,
                                      embeds_cmp_mask=embeds_cmp_mask,
                                      ids_cmp_mask=ids_cmp_mask,
                                      max_new_tokens=500,
                                      num_img_gen_tokens=num_img_out_tokens)
        text = re.sub(r'\s*<[^>]*>\s*', ' ', output['text']).strip()
        print(output['text'])
        print('*' * 20)
        with open("{}/text.txt".format(save_folder), 'a+') as text_file:
            text_file.write(text + '\n')
        with open("{}/token.txt".format(save_folder), 'a+') as token_file:
            token_file.write("context token: {}\n".format(input_ids.shape))
