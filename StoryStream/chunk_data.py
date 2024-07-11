import json
import glob

def split_entries(entries, chunk_size=5):
    split_entries = []
    for id, entry in enumerate(entries):
        images = entry['images']
        captions = entry['captions']
        
        if len(images) == len(captions):
            for i in range(0, len(images), chunk_size):
                chunk_images = images[i:i+chunk_size]
                chunk_captions = captions[i:i+chunk_size]
                
                if len(chunk_images) == chunk_size and len(chunk_captions) == chunk_size:
                    new_entry = {
                        "id": id,
                        "images": chunk_images,
                        "captions": chunk_captions,
                    }
                    split_entries.append(new_entry)
    return split_entries

def process_files(input_pattern, output_file):
    all_entries = []

    # Find all files matching the pattern
    input_files = glob.glob(input_pattern)

    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                all_entries.append(data)

    # Split entries into chunks of 5
    split_entries_list = split_entries(all_entries, 10)

    # Write split entries to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in split_entries_list:
            f.write(json.dumps(entry) + '\n')

# Usage
process_files('train.jsonl', 'train_10.jsonl')
