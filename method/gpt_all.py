from openai import OpenAI
import os
import json
from tqdm import tqdm
from method.utils import process_new_data
import argparse
from multiprocessing import Pool
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--gpt_model', type=str, default='gpt-5')
parser.add_argument('--llm_model', type=str, default='deepseekR1')
parser.add_argument('--guidebook', action='store_true')
args = parser.parse_args()

args.guidebook = True
def process_item(item_data, idx, model, guidebook):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    max_retry = 10
    retry_count = 0
    while retry_count < max_retry:
        try:
            process_new_data(client, item_data, sample_index=idx, model=model, 
                        guidebook=guidebook, output_path=f'data/label/{args.llm_model}/{args.gpt_model}')
            break
        except Exception as e:
            print(e)
            retry_count += 1

def check_exist(idx):
    file_path = f"data/label/{args.llm_model}/{args.gpt_model}/{idx + 1}.json"
    if os.path.exists(file_path):
        return True
    else:
        return False

def main():
    new_data = []
    with open(f"data/raw/omni/{args.llm_model}.json", "r") as f:
        new_data = json.load(f)


    for idx, data in enumerate(new_data):
        if not check_exist(idx):
            print(f"Processing {idx}...")
            process_item(data, idx, args.gpt_model, args.guidebook)

    # # Create a partial function with fixed arguments
    # process_func = partial(process_item, 
    #                      model=args.model,
    #                      guidebook=args.guidebook,
    #                      example=args.example)

    # # Process items in parallel with a pool of 10 workers
    # with Pool(10) as pool:
    #     # Create list of (data, idx) tuples
    #     items = [(data, idx) for idx, data in enumerate(new_data)]
    #     # Use tqdm to show progress
    #     list(tqdm(pool.starmap(process_func, items), total=len(items)))

if __name__ == '__main__':
    main()
