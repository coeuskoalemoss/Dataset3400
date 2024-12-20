
import json

# Path to your JSONL file
dataset_path = "cleaned_536.jsonl"

with open(dataset_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

cleaned_lines = []
for line in lines:
    data = json.loads(line)
    
    # Access the 'features' field inside 'input' and ensure it is a list
    if 'input' in data and 'features' in data['input']:
        if not isinstance(data['input']['features'], list):
            data['input']['features'] = [data['input']['features']]
    
    cleaned_lines.append(json.dumps(data) + '\n')

# Write the cleaned data back to a new file
cleaned_dataset_path = "cleaned_536.jsonl"
with open(cleaned_dataset_path, 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)