import json
from openai import OpenAI
import ast
import time
import os
import base64
# from PIL import Image
import io
import re

client = OpenAI(
    base_url="YOUR_URL",
    api_key="YOUR_KEY",
)

story_instruction = (
    "You are a gifted storyteller specializing in creating engaging narratives "
    "for children based on visual cues and the previous story. Your task is to craft "
    "a charming story from a series of images from the cartoon \"Rabbits Invasion.\" "
    "\nImage Use: I will provide every image to you. File names are listed below. "
    "You should fully understand the semantics and details of these images and use "
    "them for the story. "
    "\nPrevious Story Use: I will provide you the previous story. If the previous "
    "story is empty, then you can start a new story on your own. When the previous "
    "story exists, make sure the new story is continuous. "
    "\nNarrative Requirements: Ensure that the narrative is child-friendly and "
    "coherent across all images. The language should be simple and understandable "
    "for children aged 5-8 years. "
    "\nOutput Format: Deliver the story in the following format, ensuring all parts "
    "are connected: "
    "\n    * \\{\\{[keyframe_file_name_0]->[story_0]@@keyframe_file_name_1->story_1@@"
    "keyframe_file_name_2->story_2@@…\\}\\} "
    "\n    * replace the [keyframe_file_name_x] with the real keyframe name. replace "
    "the [story_x] with your generated story. "
    "\nYour goal is to weave these individual images into a seamless and "
    "entertaining story that captures the imagination of young readers."
)
link_instruction = (
    "You are a gifted storyteller specializing in creating engaging narratives for children. "
    "Your task is to link several charming stories from the cartoon \"Rabbits Invasion Into\" a long story. "
    "Story Use: I will provide several stories for you. You may modify the story text to make them more continuous. "
    "Narrative Requirements: Ensure that the narrative is child-friendly and coherent across all images. "
    "The language should be simple and understandable for children aged 5-8 years. "
    "Output Format: Deliver the story in the following format, ensuring all parts are connected: "
    "* \\{\\{[keyframe_file_name_0]->[story_0]@@keyframe_file_name_1->story_1@@keyframe_file_name_2->story_2@@…\\}\\} "
    "* replace the [keyframe_file_name_x] with the real keyframe name. replace the [story_x] with your generated story."
    "Your goal is to weave these individual stories into a seamless and "
    "entertaining long story that captures the imagination of young readers."
)


def api_call(messages):
    try_times = 0
    while try_times < 3:
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="gpt-4-turbo-2024-04-09",
                # "gpt-4-0125-preview", #"claude-3-opus-20240229", #"gpt-4-1106-preview",
                max_tokens=4096,
                temperature=0.3,
                # stop=['<wait to execute>']
            )
            success = True
            break
        except Exception as e:
            print(f"Error during API call: {e}")
            time.sleep(15)
            try_times += 1
            success = False
    if success:
        cleaned_string = chat_completion.choices[0].message.content.strip()
        return cleaned_string
    else:
        return None


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def construct_dataset(images, pool):
    images_base64 = [encode_image(img_path) for img_path in images]
    image_name = [img_path.split('/')[-1] for img_path in images]
    messages = []

    for img in images_base64:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                }
            ]
        })
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": story_instruction
                }
            ]
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "File names: {}".format(image_name)
                }
            ]
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Previous Story: {}".format(pool)
                }
            ]
        }
    )

    result = api_call(messages)
    print(result)
    return result


def link_dataset(pools):
    messages = []
    # Placeholder function to link datasets into a story
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": link_instruction
                }
            ]
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "\nstories: {}".format(pools)
                }
            ]
        }
    )
    result = api_call(messages)
    print(result)
    return result


def convert_to_jsonl(input_string):
    # Extract content between double braces
    content = re.search(r'\{\{(.*?)\}\}', input_string).group(1)

    # Split the content based on '@@' to separate different entries
    entries = content.split("@@")

    # Initialize dictionaries to hold images and captions
    images = []
    captions = []

    # Process each entry to separate image file names and captions
    for entry in entries:
        if '->' in entry:
            image, caption = entry.split('->', 1)
            images.append(image.strip())
            captions.append(caption.strip())

    # Create a dictionary for JSONL output
    jsonl_data = {
        "images": images,
        "captions": captions
    }

    # Convert dictionary to JSONL string (one-line JSON)
    return json.dumps(jsonl_data)


def find_jpg_files(directory):
    # List to store all jpg files found
    jpg_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                # Construct the full path to the file
                full_path = os.path.join(root, file)
                jpg_files.append(full_path)

    fns = lambda s: sum(((s, int(n)) for s, n in re.findall('(\D+)(\d+)', 'a%s0' % s)), ())
    # Sort the files by name
    sorted_filenames = sorted(jpg_files, key=lambda x: fns(x.split('/')[-1]))

    return sorted_filenames


def main():
    image_path = 'image_dir'
    images = find_jpg_files(image_path)
    pool = []
    output_path = 'story_30.jsonl'
    dir_path = os.path.dirname(output_path)
    os.makedirs(dir_path, exist_ok=True)

    # for i in range(0, len(images), 10):
    for i in range(0, len(images), 10):
        # Process each batch of 10 images
        batch = images[i:i + 10]
        story = construct_dataset(batch, pool)
        if story is None:
            continue
        pool.append(story)

        # If pool size is large enough, link datasets and write to file
        if len(pool) >= 3:
            story = link_dataset(pool)
            if story is not None:
                story_json = convert_to_jsonl(story)
                # Append the story JSON to the output file
                with open(output_path, 'a+') as file:
                    file.write(story_json + '\n')
            pool = []  # Reset pool after processing


main()
