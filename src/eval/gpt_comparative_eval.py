import json
from openai import OpenAI
import ast
import time
import os
import base64
# from PIL import Image
import io

client = OpenAI(
    base_url="YOUR_URL",
    api_key="YOUR_KEY",
)

instruction = "Please act as an impartial judge and evaluate the quality of the generation story contents provided by two AI assistants. Your job is to evaluate which assistant's generation is better. Your evaluation should consider the coherence of the generated story images and text. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."

# style
# instruction = "Please act as an impartial judge and evaluate the quality of the generation story contents provided by two AI assistants. Your job is to evaluate which assistant's generation is better. Your evaluation should consider the style consistency of the story images. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."

# text engaging level
# instruction = "Please act as an impartial judge and evaluate the quality of the generation story contents provided by two AI assistants. Your job is to evaluate which assistant's generation is better. Your evaluation should consider the engaging level of the story. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie."

def api_call(messages):
    try_times = 0
    while try_times < 3:
        try:
            chat_completion = client.chat.completions.create(
                messages=messages,
                model="gpt-4-turbo-2024-04-09", #"gpt-4-0125-preview", #"claude-3-opus-20240229", #"gpt-4-1106-preview",
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


def read_json_and_extract_content(filepath):
    """
    Reads a JSON file and extracts sentences and images.
    
    Args:
    filepath (str): The path to the JSON file.
    
    Returns:
    dict: A dictionary with two keys 'sentences' and 'images', containing the respective content.
    """
    with open(filepath, 'r') as file:
        data = json.load(file)
    
    all_content = []
    for line in data:
        extracted_content = {
            "sentences": [],
            "images": []
        }
        # Matching sentences to their corresponding images using their indices
        for ix in line['sentence_ixs']:
            if ix == 0:
                continue
            extracted_content['sentences'].append(line['sentences'][ix].replace('<|beginofimage|>', ''))
            extracted_content['images'].append(line['images'][ix])
        all_content.append(extracted_content)

    return all_content


def read_seed_content_from_folders(base_path):
    """
    Reads sentences from text.txt and image paths from subfolders named val_x.
    
    Args:
    base_path (str): Path to the main folder containing subfolders val_0 to val_179.
    
    Returns:
    list of dict: Each dictionary contains 'sentences' and 'images' from each subfolder.
    """
    contents = []
    
    # Iterate over each possible subfolder val_0 to val_179
    for i in range(180):  # 0 to 179 inclusive
        folder_name = f"val_{i}"
        folder_path = os.path.join(base_path, folder_name)
        
        if os.path.exists(folder_path):
            content_dict = {
                "sentences": [],
                "images": []
            }
            
            # Read sentences from text.txt
            text_file_path = os.path.join(folder_path, 'text.txt')
            if os.path.isfile(text_file_path):
                with open(text_file_path, 'r') as file:
                    content_dict['sentences'] = file.read().splitlines()[:6]
                    content_dict['sentences'] = [s.replace('[INST]', '') for s in content_dict['sentences'] ]
            
            # Collect paths for the images ori_01 to ori_06
            for j in range(1, 7):  # 1 to 6 inclusive
                image_name = f"ori_0{j}.jpg"  # Assuming the images are in .jpg format
                image_path = os.path.join(folder_path, image_name)
                if os.path.isfile(image_path):
                    content_dict['images'].append(image_path)
            
            # Add the content dictionary to the list if it contains any images or sentences
            if content_dict['sentences'] or content_dict['images']:
                contents.append(content_dict)
    
    return contents


def evaluate_models(assistant_a, assistant_b, instruction):
    # Encode all images to base64
    images_a_base64 = [encode_image(img_path) for img_path in assistant_a['images'][:5]]
    images_b_base64 = [encode_image(img_path) for img_path in assistant_b['images'][:5]]
    
    # Extract the stories from both assistants
    story_a = assistant_a['sentences']
    story_b = assistant_b['sentences']
    
    messages = []
    # A
    messages.append(
        {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Story text from Assistant A: {}\n".format(story_a[:5])
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
                "text": "Images from Assistant A are encoded in base64.\n"
            }
        ]
        }
    )
    for img_a in images_a_base64:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_a}"}
                }
            ]
        })
    
    # B
    messages.append(
        {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Story text from Assistant B: {}\n".format(story_b[:5])
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
                "text": "Images from Assistant B are encoded in base64.\n"
            }
        ]
        }
    )
    for img_b in images_b_base64:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b}"}
                }
            ]
        })

    # INST
    messages.append(
        {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": instruction
            }
        ]
        }
    )
    # Combine stories and encoded images into the evaluation instruction
    result = api_call(messages)
    print(result)
    return result

def main():
    # read mm json
    mm_contents = read_json_and_extract_content('/group/40034/shuaisyang/seed_project/StorySalon/llm_eval/mm_eval.json')
    seed_contents = read_seed_content_from_folders('/group/40034/shuaisyang/seed_project/StorySalon/llm_eval/gen_george_len7')
    assert len(mm_contents) == len(seed_contents)
    mm_win = 0
    seed_win = 0
    tie = 0
    error = []
    for i in range(len(mm_contents)):
    # for i in range(2):
        mm = mm_contents[i]
        seed = seed_contents[i]
        judgment = evaluate_models(mm, seed, instruction)

        if "[[A]]" in judgment:
            mm_win += 1
        elif "[[B]]" in judgment:
            seed_win += 1
        elif "[[C]]" in judgment:
            tie += 1
        else:
            error.append([i, judgment])
    
    with open('coherence.txt', 'w') as f:
        f.write("mm:{}\nseed:{}\ntie:{}\nerror:{}".format(mm_win, seed_win, tie, error))

main()