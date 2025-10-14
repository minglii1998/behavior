from openai import OpenAI
import os
import json
from tqdm import tqdm
from method.utils import process_new_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt-4o')
parser.add_argument('--data', type=str, default='Omni_deepseekR1_results_Q9.json')
parser.add_argument('--example', action='store_true')
parser.add_argument('--guidebook', action='store_true')
args = parser.parse_args()


api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


with open(f"Omni/{args.data}", "r") as f:
    new_data = json.load(f)

max_retry = 10
for idx, data in enumerate(new_data):
    retry_count = 0
    while retry_count < max_retry:
        try:
            process_new_data(client, data, sample_index=idx, model=args.model, guidebook=args.guidebook, example=args.example)
            break
        except Exception as e:
            retry_count += 1

gt_path = "data/sentence_gt"

answer_path = f"result/{args.model}"

if args.guidebook:
    answer_path += "_guide"

if args.example:
    answer_path += "_example"


para_result = []
sen_result = []

for file in os.listdir(answer_path):
    idx = file.split('.')[0]
    with open(os.path.join(answer_path, file), "r") as f:
            answer_data = json.load(f)
    with open(os.path.join(gt_path, f"{idx}.json"), "r") as f:
        gt_data = json.load(f)

    for (answer_item, gt_item) in zip(answer_data, gt_data):
        if answer_item['class-1'] == gt_item['gt-class-1']:
            para_result.append(1)
        else:
            para_result.append(0)
        if answer_item['class-2'] == gt_item['gt-class-2']:
            sen_result.append(1)
        else:
            sen_result.append(0)

print(f"Para accuracy: {sum(para_result) / len(para_result)}")
print(f"Sen accuracy: {sum(sen_result) / len(sen_result)}")


model_name = f"{args.model}{'_example' if args.example else ''}{'_guidebook' if args.guidebook else ''}"
with open("result/leaderboard.json", "r") as f:
    leaderboard = json.load(f)

if model_name not in leaderboard:
    leaderboard.append({
        "model": model_name,
        "para_accuracy": sum(para_result) / len(para_result),
        "sen_accuracy": sum(sen_result) / len(sen_result),
    })
else:
    for item in leaderboard:
        if item['model'] == model_name:
            item['para_accuracy'] = sum(para_result) / len(para_result)
            item['sen_accuracy'] = sum(sen_result) / len(sen_result)

with open("result/leaderboard.json", "w") as f:
    json.dump(leaderboard, f, indent=4)

