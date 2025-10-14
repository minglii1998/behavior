import os
import json
from tqdm import tqdm
from method.utils import label_sentence
from openai import OpenAI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-4o')
args = parser.parse_args()


api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


with open("data/sample_results.json", "r") as f:
    new_data = json.load(f)

# for idx, data in enumerate(new_data):
#     label_sentence(client, data, sample_index=idx, model=args.model, output_path=f"data/sentence_only")

gt_path = "data/sentence_gt"
answer_path = f"data/sentence_only"


para_result = []

for file in os.listdir(answer_path):
    idx = file.split('.')[0]
    with open(os.path.join(answer_path, file), "r") as f:
            answer_data = json.load(f)
    with open(os.path.join(gt_path, f"{idx}.json"), "r") as f:
        gt_data = json.load(f)

    for (answer_item, gt_item) in zip(answer_data, gt_data):
        if answer_item['class'] == gt_item['gt-class-2']:
            para_result.append(1)
        else:
            para_result.append(0)

print(sum(para_result) / len(para_result))