import os
import pickle
import torch
from transformers import CLIPTextModel, CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

test_root = "data/realworld_raw"

result_dict = {}
for task_name in os.listdir(test_root):
    task_path = os.path.join(test_root, task_name)
    if not os.path.isdir(task_path):
        continue

    task_dict = {}

    for var_folder in os.listdir(task_path):
        var_path = os.path.join(task_path, var_folder)
        if not os.path.isdir(var_path):
            continue
        if var_folder.startswith("variation"):
            try:
                var_num = int(var_folder.replace("variation", ""))
            except ValueError:
                continue
        else:
            continue

        pkl_path = os.path.join(var_path, "variation_descriptions.pkl")
        if not os.path.exists(pkl_path):
            print(f"{pkl_path} 不存在")
            continue

        with open(pkl_path, "rb") as f:
            text_list = pickle.load(f)  
        text_list = list(text_list)
        print(text_list)
        tokenized = tokenizer(
            text_list,
            return_tensors="pt",
            padding="max_length",
            max_length=53,
            truncation=True
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}

        with torch.no_grad():
            outputs = model(**tokenized)
        token_embeddings = outputs.last_hidden_state  # (n, 53, 512)
        task_dict[var_num] = token_embeddings.cpu()

    result_dict[task_name] = task_dict

# print(result_dict["close_jar"][2].shape)
with open("instructions/physical/instructions.pkl", "wb") as file:
    pickle.dump(result_dict, file)
