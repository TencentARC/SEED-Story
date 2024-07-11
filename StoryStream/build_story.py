import os
import openai
import re
import json
import time

# Constants
OPENAI_API_KEY = ""  # Add your OpenAI API key here
MODEL = "gpt-4-1106-preview"  # Can be gpt-3.5-turbo or gpt-4

LEN_STORY = 30
IMAGE_PATH = 'image/george_full'
description_path = 'gpt4v_descriptive/george_full/captions.jsonl'
subtitle_path = 'subtitle/george'
output_path = 'gpt4_story/george_len{}'.format(LEN_STORY)
with_subtitle = False

PROMPT = """
Create a connected story from the captions of these 'Curious George' cartoon keyframes, following these guidelines:

1. Ensure each part of the story aligns with its corresponding image caption.
2. Include "George" in the narrative whenever the caption mentions a monkey.
3. The story should flow logically from one image to the next, using child-friendly language.
4. Format the output as: [filename.jpg]->[narrative], with each image and its story on a separate line.
5. Directly provide the requested output without including this instruction conversation.
6. The overall story should be cohesive and engaging.
"""

cnt = 0


def generate_story(model, messages):
    """
    Generate a story based on the provided model and messages.

    :param model: The model to use for generation (e.g., 'gpt-4').
    :param messages: Messages to send to the model.
    :return: The generated story or None in case of an exception.
    """
    try:
        completions = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=0.3,
        )
        # time.sleep(15)
        return completions
    except Exception as e:
        print(f"Error during API call: {e}")
        time.sleep(15)
        return None


def extract_info_from_output(gpt_output):
    """
    Extract image paths and captions from the GPT output.

    :param gpt_output: The output from GPT.
    :return: A list of image paths and captions.
    """
    image_paths = []
    captions = []

    for line in gpt_output.strip().split('\n'):
        match = re.match(r"(.*?)\.jpg->(.*)", line.strip())
        if match:
            image_path = match.group(1) + ".jpg"
            caption = match.group(2).strip()
            image_paths.append(image_path)
            captions.append(caption)

    print(image_paths)
    return image_paths, captions


def create_json_structure(image_paths, captions):
    """
    Create a JSON structure for the output.

    :param image_paths: List of image paths.
    :param captions: List of captions.
    :return: A dictionary in JSON format.
    """
    global cnt
    return {
        "id": cnt,
        "images": image_paths,
        "captions": captions,
        "orders": list(range(len(image_paths)))
    }


def process_video_directory():
    """
    Process each video directory to generate narratives and save them in JSON format.
    """
    global cnt
    with open(description_path, 'r') as file:
        json_list = []
        description_lines = []
        for line in file:
            # if eval(line)['image'].split('/')[0] < "000163":
            #     print('skip: {}'.format(eval(line)['image']))
            #     continue
            description_lines.append(line.strip())  # Add the current line to the batch
            if len(description_lines) == LEN_STORY:  # Once we have three lines
                # Join the lines to form the complete description
                description = " ".join(description_lines)
                # Process the accumulated description
                json_line = process_description(description, json_list, None)
                # Reset for the next batch
                description_lines = []
                with open(f'{output_path}/story.jsonl', 'a') as file:
                    file.write(str(json_line) + '\n')

        # Don't forget to process the last batch if it's less than 3 lines and not empty
        if description_lines:
            description = " ".join(description_lines)
            json_line = process_description(description, json_list, None)
            with open(f'{output_path}/story.jsonl', 'a') as file:
                file.write(str(json_line) + '\n')


def process_description(description, json_list, subtitle):
    """
    Process the description to generate a narrative and update the JSON list.

    :param description: The description to process.
    :param json_list: The list to update with the new JSON data.
    """
    content = PROMPT + 'Image Descriptions: \n' + description
    if with_subtitle:
        content += 'Subtitles: \n' + subtitle
    print('content\n' + content)
    request = [{"role": "user", "content": content}]
    result = None
    attempt = 0
    while True:
        if attempt > 3:
            break
        result = generate_story(MODEL, request)
        if result is not None and result['choices'][0]['message']['content'] != '':
            break
        attempt += 1
    if result:
        res = result['choices'][0]['message']['content']
        print(res)
        image_paths, captions = extract_info_from_output(res)
        output_json = create_json_structure(image_paths, captions)
        json_list.append(output_json)
        print('output_json\n' + str(output_json))
        print(output_json)
        global cnt
        cnt += 1
        return output_json


# Main execution
if __name__ == "__main__":
    openai.api_key = OPENAI_API_KEY
    os.makedirs(output_path, exist_ok=True)

    process_video_directory()
