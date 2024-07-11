import torchdata.datapipes as dp
import json
from PIL import Image
import functools
import numpy as np
import torch
import pickle
import os
import cv2
import random
from torchvision import transforms
from braceexpand import braceexpand
import hydra
from random import choice
import tarfile
from torchdata.datapipes.iter import TarArchiveLoader
from typing import cast, IO, Iterable, Iterator, Optional, Tuple, Dict
from torchdata.datapipes import functional_datapipe
from io import BufferedIOBase
from torchdata.datapipes.utils import StreamWrapper
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
import warnings
from torchdata.datapipes.iter import IterDataPipe

import pyrootutils

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

gen_prompt = [
    "Please show me a picture of ",
    "Please design an image of ",
    "Please produce a photo of ",
    "Please generate an image of ",
    "Please draw a painting of ",
    "I'd like to see a drawing of ",
    "I'd love to see an illustration of ",
    "I'd like to view an image of ",
    "I want to see a picture of ",
    "I would like to see a photo of ",
    "Show me a photo of ",
    "Generate a picture of ",
    "Show me a photograph of ",
    "Generate an image of ",
    "Generate an image: ",
    "Generate a picture: ",
    "Generate a painting: ",
    "Generate a photograph: ",
    "Show me a photograph: ",
    "Draw a picture: ",
    "Draw a painting: ",
    "Draw an image: ",
    "Can you make an image of ",
    "Can you draw a painting of ",
    "Can you produce a picture of ",
    "Can you generate a photo of ",
    "Can you depict a picture of ",
    "Can you show me an illustration of ",
]

gen_prompt_response = [
    "Here is a picture.",
    "I have designed an image.",
    "Here is a photo.",
    "I have generated an image.",
    "Here's a painting.",
    "Here's a drawing.",
    "Enjoy this illustration.",
    "Take a look at this image.",
    "Here is a picture.",
    "I have created a photo.",
    "Enjoy this photo.",
    "I have generated a picture.",
    "Here is a photograph.",
    "Here's an image.",
    "Certainly, here's an image.",
    "Absolutely, here is a painting.",
    "Sure, here is a picture.",
    "Of course, here is a photo.",
    "Certainly, please enjoy this picture.",
    "Sure, please enjoy this illustration.",
    "",
]

jdb_filter_vocab = ['watermark', 'watermark,', 'chaos 100', 'chaos 100,']


def filter_data_with_image_ids(item):
    if ('images' not in item):
        # print(item['__key__'])
        # print('filtered because no images')
        return False
    elif 'input_ids' not in item:
        return False
    else:
        return True


def calculate_new_dimensions(height, width, target_size):
    if height < width:
        new_height = target_size
        new_width = int(width * (target_size / height))
    else:
        new_width = target_size
        new_height = int(height * (target_size / width))
    return new_height, new_width


def unwarp_data(item):
    unwarpped = {}
    for key, value in item.items():
        if isinstance(value, dict):
            unwarpped.update(value)
        elif value is not None:
            unwarpped[key] = value
    if 'metadata' not in unwarpped:
        unwarpped['metadata'] = '{}'
    # if '__key__' in unwarpped:
    #     unwarpped['__key__'] = unwarpped['__key__'].split('/')[-1]
    return unwarpped


# def filter_data_with_similarity(item, similarity_thr=0.2, min_resolution=180, min_aspect_ratio=0.666):
def filter_data_with_similarity(item, similarity_thr=0.2, assure_text=True):
    if ('images' not in item):
        # print(item['__key__'])
        # print('filtered because no images')
        return False
    elif (not item.get('filter_flag', True)):
        # print(item['__key__'])
        # print('filtered because filter flag.')
        return False
    elif assure_text and ('text' not in item):
        # print(item['__key__'])
        # print('filtered because assure_text')
        return False
    else:
        metadata = json.loads(item['metadata'])

        if 'all_similarities' in metadata:
            similarity = max(metadata['all_similarities'])
        elif 'similarity' in metadata:
            similarity = metadata['similarity']
        elif 'score' in metadata:
            similarity = metadata['score']
        elif 'SCORE' in metadata:
            similarity = metadata['SCORE']
        else:
            similarity = None

        if similarity is not None:
            if similarity < similarity_thr:
                # print(item['__key__'])
                # print('filtered because similarity')
                return False

        return True


def single_turn_edit_collate(batch):
    results = {}
    keys = batch[0].keys()

    for key in keys:
        cur = [batch[i][key] for i in range(len(batch)) if batch[i][key] is not None]
        if len(cur) == 0:
            results[key] = None
        elif isinstance(cur[0], torch.Tensor):
            if key in ['embeds_gen_mask', 'embeds_cmp_mask', 'images']:
                results[key] = torch.cat(cur, dim=0)
            else:
                results[key] = torch.stack(cur, dim=0)
        else:
            results[key] = cur

    return results


def decode_t2i_data(item,
                    image_dir,
                    tokenizer,
                    image_transform=None,
                    sd_image_transform=None,
                    max_length=128,
                    min_resolution=400,
                    instruction_prompt='[INST] {instruction} [/INST]\n',
                    turn_sep='\n',
                    system_message='',
                    min_aspect_ratio=0.666,
                    num_img_in_tokens=64,
                    num_img_out_tokens=64):
    key, value = item

    if 'image' not in value or 'caption' not in value:
        return {}

    image_path = os.path.join(image_dir, value["image"])

    try:
        image = Image.open(image_path).convert('RGB')

        width, height = image.size

        aspect_ratio = height / width
        if height < min_resolution or width < min_resolution:
            print(f'filtered because resolution: ({width},{height})')
            return {}
        if aspect_ratio < min_aspect_ratio or aspect_ratio > 1 / min_aspect_ratio:
            print(f'filtered because aspect ratio: ({width},{height})')
            return {}
            ### SD related

        image_data = {}

        if sd_image_transform is not None:
            # image_data['original_sizes'] = torch.tensor([height, width])
            sd_image_tensor = sd_image_transform(image)
            target_size = sd_image_tensor.shape[-2]
            target_width, target_height = calculate_new_dimensions(height=height, width=width, target_size=target_size)
            y1 = max(0, int(round((target_height - target_size) / 2.0)))
            x1 = max(0, int(round((target_width - target_size) / 2.0)))
            # image_data['crop_top_lefts'] = torch.tensor([y1, x1])
            image_data['time_ids'] = torch.tensor([height, width, y1, x1, target_size, target_size])

            image_data['sd_images'] = sd_image_tensor

        if image_transform is not None:
            image = image_transform(image)

    except Exception as e:
        print('Error while decode image: ', e)
        return {}

    input_ids = []
    labels = []
    input_text = ''

    if system_message != '':
        if not system_message.endswith('\n'):
            system_message += '\n'
        input_text += system_message
        item_ids = tokenizer.encode(system_message, add_special_tokens=False)
        item_labels = [-100] * len(item_ids)
        input_ids.extend(item_ids)
        labels.extend(item_labels)

    caption = value["caption"]

    image_cmp_tokens = BOI_TOKEN + ''.join(
        [IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)]) + EOI_TOKEN

    image_gen_tokens = BOI_TOKEN + ''.join(
        [IMG_TOKEN.format(int(item)) for item in range(num_img_out_tokens)]) + EOI_TOKEN

    instruction = instruction_prompt.format_map({'instruction': caption})

    response = image_gen_tokens
    images = torch.stack([image], dim=0)
    # print(instruction)

    item_ids = tokenizer.encode(instruction, add_special_tokens=False)
    item_labels = [-100] * len(item_ids)
    input_text += instruction
    input_ids.extend(item_ids)
    labels.extend(item_labels)

    item_ids = tokenizer.encode(response, add_special_tokens=False)
    item_labels = item_ids
    input_text += response
    input_ids.extend(item_ids)
    labels.extend(item_labels)

    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] + labels + [tokenizer.eos_token_id]

    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]
    ids_cmp_mask = [False] * len(input_ids)
    ids_gen_mask = [False] * len(input_ids)

    embeds_cmp_mask = [False]
    embeds_gen_mask = [True]

    # print(len(input_ids))
    if len(input_ids) >= max_length:
        # input_ids = input_ids[:max_length]
        # attention_mask = attention_mask[:max_length]
        # labels = labels[:max_length]
        # ids_cmp_mask = ids_cmp_mask[:max_length]
        # ids_gen_mask = ids_gen_mask[:max_length]
        # print('An edit sample has been removed because of max length. input_text: ', input_text)
        return {}
    else:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length
        ids_cmp_mask = ids_cmp_mask + [False] * padding_length
        ids_gen_mask = ids_gen_mask + [False] * padding_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
    ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)
    embeds_cmp_mask = torch.tensor(embeds_cmp_mask) if embeds_cmp_mask is not None else None
    embeds_gen_mask = torch.tensor(embeds_gen_mask) if embeds_gen_mask is not None else None

    boi_idx = torch.where(input_ids == boi_token_id)[0].tolist()
    eoi_idx = torch.where(input_ids == eoi_token_id)[0].tolist()

    ids_gen_mask[boi_idx[0] + 1:eoi_idx[0]] = True
    labels[boi_idx[0] + 1:eoi_idx[0] + 1] = -100

    ret = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids_gen_mask': ids_gen_mask,
        'ids_cmp_mask': ids_cmp_mask,
        'embeds_gen_mask': embeds_gen_mask,
        'embeds_cmp_mask': embeds_cmp_mask,
        'images': images,
        'text': input_text,
    }

    ret.update(image_data)

    return ret


def build_t2i_datapipe(data_dir,
                       image_dir,
                       tokenizer=None,
                       max_length=77,
                       batch_size=None,
                       min_resolution=180,
                       image_transform=None,
                       sd_image_transform=None,
                       instruction_prompt='[INST] {instruction} [INST]\n',
                       turn_sep='\n',
                       system_message='',
                       min_aspect_ratio=0.666,
                       num_img_in_tokens=64,
                       num_img_out_tokens=64,
                       cycle_count=None):
    decode_partial = functools.partial(decode_t2i_data,
                                       image_dir=image_dir,
                                       tokenizer=tokenizer,
                                       image_transform=image_transform,
                                       sd_image_transform=sd_image_transform,
                                       max_length=max_length,
                                       instruction_prompt=instruction_prompt,
                                       turn_sep=turn_sep,
                                       system_message=system_message,
                                       min_resolution=min_resolution,
                                       min_aspect_ratio=min_aspect_ratio,
                                       num_img_in_tokens=num_img_in_tokens,
                                       num_img_out_tokens=num_img_out_tokens)

    filter_partial = functools.partial(filter_data_with_image_ids)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))

    datapipe = dp.iter.FileLister(root=data_dir, masks='*.jsonl', recursive=True)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    # datapipe = dp.iter.FileLister(root=data_dir, masks='0000000.tar', recursive=True)
    datapipe = datapipe.sharding_filter()
    # datapipe = datapipe.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)

    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_jsonl_files()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.filter(filter_partial)

    # datapipe = datapipe.shuffle(buffer_size=1024)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate(single_turn_edit_collate)
    return datapipe


def decode_long_story_data(item,
                           image_dir,
                           tokenizer,
                           story_len,
                           image_transform=None,
                           sd_image_transform=None,
                           max_length=128,
                           min_resolution=400,
                           instruction_prompt='{instruction}',
                           turn_sep='\n',
                           system_message='',
                           min_aspect_ratio=0.666,
                           num_img_in_tokens=64,
                           num_img_out_tokens=64, ):
    key, value = item
    if 'images' not in value or 'captions' not in value:
        return {}

    image_paths = [os.path.join(image_dir, image_path) for image_path in value["images"]]
    # assert len(image_paths) == story_len
    story_len = len(image_paths)
    num_image_given = random.randint(0, story_len - 2)

    try:
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            images.append(image)
            width, height = image.size

        aspect_ratio = height / width
        if height < min_resolution or width < min_resolution:
            print(f'filtered because resolution: ({width},{height})')
            return {}
        if aspect_ratio < min_aspect_ratio or aspect_ratio > 1 / min_aspect_ratio:
            print(f'filtered because aspect ratio: ({width},{height})')
            return {}

        image_data = {}
        sd_image = images[num_image_given + 1]
        if sd_image_transform is not None:
            # image_data['original_sizes'] = torch.tensor([height, width])
            sd_image_tensor = sd_image_transform(sd_image)
            target_size = sd_image_tensor.shape[-2]
            target_width, target_height = calculate_new_dimensions(height=height, width=width, target_size=target_size)
            y1 = max(0, int(round((target_height - target_size) / 2.0)))
            x1 = max(0, int(round((target_width - target_size) / 2.0)))
            # image_data['crop_top_lefts'] = torch.tensor([y1, x1])
            image_data['time_ids'] = torch.tensor([height, width, y1, x1, target_size, target_size])

            image_data['sd_images'] = sd_image_tensor

        if image_transform is not None:
            for i in range(len(images)):
                images[i] = image_transform(images[i])
            images = torch.stack(images, dim=0)

    except Exception as e:
        print('Error while decode image: ', e)
        return {}

    input_ids = []
    labels = []
    input_text = ''

    if system_message != '':
        if not system_message.endswith('\n'):
            system_message += '\n'
        input_text += system_message
        item_ids = tokenizer.encode(system_message, add_special_tokens=False)
        item_labels = [-100] * len(item_ids)
        input_ids.extend(item_ids)
        labels.extend(item_labels)

    captions_all = []
    for i in range(story_len):
        caption = value["captions"][i]
        captions_all.append(caption)

    image_cmp_tokens = BOI_TOKEN + ''.join(
        [IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)]) + EOI_TOKEN

    image_gen_tokens = BOI_TOKEN + ''.join(
        [IMG_TOKEN.format(int(item)) for item in range(num_img_out_tokens)]) + EOI_TOKEN

    instruction = instruction_prompt.format_map({'instruction': captions_all[0] + image_cmp_tokens})
    for i in range(num_image_given):
        instruction = instruction + "[INST]" + captions_all[i + 1] + image_cmp_tokens

    response = "[INST]" + captions_all[num_image_given + 1] + image_gen_tokens

    images = images[:num_image_given + 2]
    # print(instruction)

    item_ids = tokenizer.encode(instruction, add_special_tokens=False)
    item_labels = [-100] * len(item_ids)
    input_text += instruction
    input_ids.extend(item_ids)
    labels.extend(item_labels)

    item_ids = tokenizer.encode(response, add_special_tokens=False)
    item_labels = item_ids
    input_text += response
    input_ids.extend(item_ids)
    labels.extend(item_labels)

    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = [-100] + labels + [tokenizer.eos_token_id]

    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]
    ids_cmp_mask = [False] * len(input_ids)
    ids_gen_mask = [False] * len(input_ids)

    embeds_cmp_mask = [True] + [True] * num_image_given + [False]
    embeds_gen_mask = [False] + [False] * num_image_given + [True]

    # print(len(input_ids))
    if len(input_ids) >= max_length:
        # input_ids = input_ids[:max_length]
        # attention_mask = attention_mask[:max_length]
        # labels = labels[:max_length]
        # ids_cmp_mask = ids_cmp_mask[:max_length]
        # ids_gen_mask = ids_gen_mask[:max_length]
        # print('An edit sample has been removed because of max length. input_text: ', input_text)
        return {}
    else:
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        attention_mask = attention_mask + [0] * padding_length
        labels = labels + [-100] * padding_length
        ids_cmp_mask = ids_cmp_mask + [False] * padding_length
        ids_gen_mask = ids_gen_mask + [False] * padding_length

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    ids_cmp_mask = torch.tensor(ids_cmp_mask, dtype=torch.bool)
    ids_gen_mask = torch.tensor(ids_gen_mask, dtype=torch.bool)
    embeds_cmp_mask = torch.tensor(embeds_cmp_mask) if embeds_cmp_mask is not None else None
    embeds_gen_mask = torch.tensor(embeds_gen_mask) if embeds_gen_mask is not None else None

    boi_idx = torch.where(input_ids == boi_token_id)[0].tolist()
    eoi_idx = torch.where(input_ids == eoi_token_id)[0].tolist()

    ids_cmp_mask[boi_idx[0] + 1:eoi_idx[0]] = True
    for i in range(num_image_given):
        ids_cmp_mask[boi_idx[i + 1] + 1:eoi_idx[i + 1]] = True

    ids_gen_mask[boi_idx[-1] + 1:eoi_idx[-1]] = True
    labels[boi_idx[-1] + 1:eoi_idx[-1] + 1] = -100

    ret = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids_gen_mask': ids_gen_mask,
        'ids_cmp_mask': ids_cmp_mask,
        'embeds_gen_mask': embeds_gen_mask,
        'embeds_cmp_mask': embeds_cmp_mask,
        'images': images,
        'text': input_text,
    }

    ret.update(image_data)

    return ret


def build_long_story_datapipe(data_dir,
                              image_dir,
                              tokenizer=None,
                              story_len=30,
                              max_length=77,
                              batch_size=None,
                              min_resolution=180,
                              image_transform=None,
                              sd_image_transform=None,
                              instruction_prompt='{instruction}',
                              turn_sep='\n',
                              system_message='',
                              min_aspect_ratio=0.666,
                              num_img_in_tokens=64,
                              num_img_out_tokens=64,
                              cycle_count=None):
    decode_partial = functools.partial(decode_long_story_data,
                                       image_dir=image_dir,
                                       tokenizer=tokenizer,
                                       story_len=story_len,
                                       image_transform=image_transform,
                                       sd_image_transform=sd_image_transform,
                                       max_length=max_length,
                                       instruction_prompt=instruction_prompt,
                                       turn_sep=turn_sep,
                                       system_message=system_message,
                                       min_resolution=min_resolution,
                                       min_aspect_ratio=min_aspect_ratio,
                                       num_img_in_tokens=num_img_in_tokens,
                                       num_img_out_tokens=num_img_out_tokens)

    filter_partial = functools.partial(filter_data_with_image_ids)

    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))

    datapipe = dp.iter.FileLister(root=data_dir, masks='*.jsonl', recursive=True)
    datapipe = datapipe.shuffle()
    datapipe = datapipe.cycle(count=cycle_count)
    datapipe = datapipe.shuffle()
    # datapipe = dp.iter.FileLister(root=data_dir, masks='0000000.tar', recursive=True)
    datapipe = datapipe.sharding_filter()
    # datapipe = datapipe.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)

    datapipe = datapipe.open_files(mode='r')
    datapipe = datapipe.parse_jsonl_files()
    datapipe = datapipe.map(decode_partial)
    datapipe = datapipe.filter(filter_partial)

    # datapipe = datapipe.shuffle(buffer_size=1024)
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate(single_turn_edit_collate)
    return datapipe


def build_multi_datapipes(datapipes, tokenizer=None, image_transform=None, sd_image_transform=None,
                          sample_weights=None):
    # assert concat_type in ['concat', 'mux_longest', 'sample']
    if sample_weights is None:
        sample_weights = [1] * len(datapipes)
    else:
        assert len(sample_weights) == len(datapipes)

    datapipes = [
        hydra.utils.instantiate(datapipe, tokenizer=tokenizer, image_transform=image_transform,
                                sd_image_transform=sd_image_transform) for datapipe in datapipes
    ]

    datasets_to_weights_dict = {}
    for dataset, sample_weight in zip(datapipes, sample_weights):
        datasets_to_weights_dict[dataset] = sample_weight
    datapipe = dp.iter.SampleMultiplexer(datasets_to_weights_dict)

    return datapipe
