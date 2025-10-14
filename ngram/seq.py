import json
import os

label_dir = "../data/label"
seq_dir = "../data/sequence"
os.makedirs(seq_dir, exist_ok=True)
models = os.listdir(label_dir)
annotator = "gpt-5"

label_map = {
    "Read": 'R',
    "Monitor": 'M',
    "Plan": 'P',
    "Analyze": 'N',
    "Explore": 'E',
    "Implement": 'I',
    "Answer": 'A',
    "Verify": 'V',
}

for model in models:
    model_data = []
    for file in os.listdir(os.path.join(label_dir, model, annotator)):
        with open(os.path.join(label_dir, model, annotator, file), "r") as f:
            data = json.load(f)
        new_data = ''
        for item in data:
            if new_data == '':
                new_data += label_map[item["sentence-category"]]
            else:
                if label_map[item["sentence-category"]] == new_data[-1]:
                    continue
                new_data += label_map[item["sentence-category"]]
        model_data.append(new_data)
    with open(os.path.join(seq_dir, f"model/{model}.json"), "w") as f:
        json.dump(model_data, f, indent=4)


reasoning_models = ["deepseekR1", "Qwen3_32B", "QwQ32B", "Phi4R"]
non_reasoning_models = ["gpt4o", "o1mini", "o3mini", "Phi4", "Qwen3_32BNR"]

reasoning_seq = []
non_reasoning_seq = []
all_seq = []

for model in reasoning_models:
    with open(os.path.join(seq_dir, f"model/{model}.json"), "r") as f:
        reasoning_seq.extend(json.load(f))
        all_seq.extend(reasoning_seq)
for model in non_reasoning_models:
    with open(os.path.join(seq_dir, f"model/{model}.json"), "r") as f:
        non_reasoning_seq.extend(json.load(f))
        all_seq.extend(non_reasoning_seq)

with open(os.path.join(seq_dir, "reasoning.json"), "w") as f:
    json.dump(reasoning_seq, f, indent=4)
with open(os.path.join(seq_dir, "non_reasoning.json"), "w") as f:
    json.dump(non_reasoning_seq, f, indent=4)

with open(os.path.join(seq_dir, "all.json"), "w") as f:
    json.dump(all_seq, f, indent=4)