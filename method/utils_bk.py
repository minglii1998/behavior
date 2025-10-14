import os
from google import genai
import json
from tqdm import tqdm

google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text, type = 'gemini'):
    if type == 'gemini':
        result = google_client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text
        )
        return result.embeddings[0].values


with open("guidebook/paragraph_guide.md", "r") as f:
        guidebook_paragraph_prompt = f.read()

with open("guidebook/sentence_guide.md", "r") as f:
        guidebook_sentence_prompt = f.read()

old_para_instruction_prompt = "A business owner plans to purchase the same model of chair for each of the $81$ employees. The total budget to spend on these chairs is $14,000$ , which includes a $7\\%$ sales tax.\nWhich of the following is closest to the maximum possible price per chair, before sales tax, the business owner could pay based on this budget?\nChoices:\nA) $148.15$\nB) $161.53$\nC) $172.84$\nD) $184.94$"

with open("data/example/SAT_DPSKR1_2_paragraph_Nan.json", "r") as f:
        old_para_data = json.load(f)

context_size = len(old_para_data)
old_context_window = min(context_size, 20)
context = []
for idx, item in enumerate(old_para_data):
    context.append(f"{idx+1}. {item['text']} [{item['class-1']}]")
    if idx == old_context_window - 1:
        break

old_context_str = "\n".join(context)

def process_new_data(client, new_data, sample_index, guidebook=True, example=True, model="gpt-4.1", output_path='result', answer_key='deepseek-reasoner (response)'):

    general_paragraph_instruction_prompt = """We are trying to label model responses with behavior labels based on Alan Schoenfeld's (1985) "Episode-Timeline" framework for problem-solving. At the end of each paragraph in the response, a class is assigned to the paragraph. The classes are: Explore, Verify, and General."""

    if guidebook:
        general_paragraph_instruction_prompt += "\n\nThe [Guidebook] section provides the definition of the classes."
    
    if example:
        general_paragraph_instruction_prompt += f"\n\nThe [Example Math Problem] section provides an example of a math problem.\n\nThe [Example Response and Labels] section provides the response of a model and the labels of the paragraphs."

    general_paragraph_instruction_prompt += """\n\nThe [New Math Problem] section provides a new math problem.
    The [Response Context] section provides the previous context of the response.
    The [New Input] section provides the new input of the response.
    The [Format] section provides the format of the output.
    During the labeling, you should refer to the guidebook of examples to make the judgement of each label."""

    new_instruction = new_data['Instruction']
    new_instruction_prompt = f"\n\nNow, the instruction is: {new_instruction}"

    new_response = new_data[answer_key]
    # First split by double newlines
    paragraphs = new_response.split('\n\n')

    new_response_list = []
    for idx, split in enumerate(paragraphs):
        new_response_list.append(f"{split}")

    format_para_prompt = (
        "You should format the output in json format about the index, a short reasonale and the class of the indexed paragraph. "
        "The format is as follows:\n"
        "{\n"
        "  'paragraphs': [\n"
        "    {'index': 'The index of the paragraph, a single integer from 1 to 5', 'reason': 'The short reason of the classification', 'class': 'The class of the paragraph'},\n"
        "    {'index': 'The index of the paragraph, a single integer from 1 to 5', 'reason': 'The short reason of the classification', 'class': 'The class of the paragraph'},\n"
        "    ...\n"
        "  ]\n"
        "}"
    )

    batch_size = 5
    para_result = []
    for idx, new_response in enumerate(tqdm(new_response_list)):
        if idx % batch_size == batch_size - 1 or idx == len(new_response_list) - 1:
            if idx == batch_size - 1 or len(new_response_list) < batch_size:
                new_input_context = ""
                new_input_context_prompt = "There is no previous paragraphs."
            else:
                new_input_context_list = new_response_list[idx-batch_size + 1 - batch_size: idx + 1 - batch_size]
                new_input_context_str = "\n".join(new_input_context_list)
                new_input_context_prompt = f"The previous {batch_size} paragraphs are:\n\n {new_input_context_str}"

            if len(new_response_list) < batch_size:
                new_input_list = new_response_list
            elif idx == len(new_response_list) - 1:
                remain = idx % batch_size + 1
                new_input_list = new_response_list[-remain:]
            else:
                new_input_list = new_response_list[idx-batch_size + 1: idx + 1]
            indexed_input_list = [f"{idx+1}. {split}" for idx, split in enumerate(new_input_list)]
            new_input_str = "\n".join(indexed_input_list)
            new_input_prompt = f"The next {len(new_input_list)} paragraphs which you need to classify are: {new_input_str}"

            combined_prompt = f"{general_paragraph_instruction_prompt}"

            if guidebook:
                combined_prompt += f"\n\n[Guidebook]\n{guidebook_paragraph_prompt}"

            if example:
                combined_prompt += f"\n\n[Example Math Problem]\n{old_para_instruction_prompt}\n\n[Example Response and Labels]\n{old_context_str}"

            # print(new_input_context_prompt)
            # print(new_input_prompt)

            combined_prompt += f"\n\n[New Math Problem]\n{new_instruction_prompt}\n\n[Response Context]\n{new_input_context_prompt}\n\n[New Input]\n{new_input_prompt}\n\n[Format]\n{format_para_prompt}"

            messages = [
                {"role": "user", "content": combined_prompt},
            ]

            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if 'gpt' in model or 'deepseek' in model:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            response_format={"type": "json_object"}
                        )
                        result = response.choices[0].message.content
                    elif 'gemini' in model:
                        response = client.models.generate_content(
                            model=model,
                            contents=combined_prompt,
                            config={
                                "response_mime_type": "application/json"
                            }
                        )
                        result = response.text
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"Failed after {max_retries} retries: {str(e)}")
                    else:
                        print(f"Attempt {retry_count} failed: {str(e)}. Retrying...")

            response_json = json.loads(result)['paragraphs']
            for item in response_json:
                # print(item['index'])
                item['text'] = new_input_list[int(item['index']) - 1]

            response_json = [{k: v for k, v in item.items() if k != 'index'} for item in response_json]
            para_result.extend(response_json)

    target_path = f"{output_path}/{model}/paragraph"
    if guidebook:
        target_path += "_guide"
    if example:
        target_path += "_example"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with open(f"{target_path}/{sample_index + 1}.json", "w") as f:
        json.dump(para_result, f, indent=2)

    new_sentence_list = []
    for item in para_result:
        paragraph = item['text']
        class_1 = item['class']
        splits = []
        current = ''
        for i, char in enumerate(paragraph):
            current += char
            if char in '.?':
                # Check if dot is between numbers
                if char == '.' and i > 0 and i < len(paragraph)-1:
                    prev_char = paragraph[i-1]
                    next_char = paragraph[i+1]
                    if prev_char.isdigit() and next_char.isdigit():
                        continue
                    if i == 1 and prev_char.isdigit():
                        continue
                splits.append(current.strip())
                current = ''
        if current:  # Add any remaining text
            splits.append(current.strip())
            
        for split in splits:
            if split == '':
                continue
            new_sentence_list.append({
                "text": split,
                "class-1": class_1,
                "class-1-reason": item['reason'],
                "class-2": '',
                "class-2-reason": ''
            })

    general_sentence_instruction_prompt = """We are trying to label model responses with behavior labels based on Alan Schoenfeld's (1985) "Episode-Timeline" framework for problem-solving. At the end of each sentence in the response, classes are assigned to the sentence. The classes are: Read, Analyze, Explore, Plan, Implement, Verify, Monitor. You will also be provided with a guidebook about the definition of the classes. Later, you will be provided another instruction and sentences in model response. You need to classify the sentences as in the context. All sentences are already labeled with the class-1 which is about the general behavoir of the paragraph that the sentence belongs to. You need to classify the fine-grained class (class-2) of the sentences as in the context which is about the specific behavior of the sentence."""

    if guidebook:
        general_sentence_instruction_prompt += "\n\nThe [Guidebook] section provides the definition of the classes."
    
    if example:
        general_sentence_instruction_prompt += f"\n\nThe [Example Math Problem] section provides an example of a math problem.\n\nThe [Example Response and Labels] section provides the response of a model and the labels of the paragraphs."
        
    general_sentence_instruction_prompt += """\n\nThe [New Math Problem] section provides a new math problem.
    The [Response Context] section provides the previous context of the response.
    The [New Input] section provides the new input of the response.
    The [Format] section provides the format of the output.
    During the labeling, you should refer to the guidebook of examples to make the judgement of each label."""

    old_sen_instruction_prompt = "Here is the example, at the end of each sentence, both class-1 and class-2 are assigned in ['class-1', 'class-2'] format. The instruction is: A business owner plans to purchase the same model of chair for each of the $81$ employees. The total budget to spend on these chairs is $14,000$ , which includes a $7\\%$ sales tax.\nWhich of the following is closest to the maximum possible price per chair, before sales tax, the business owner could pay based on this budget?\nChoices:\nA) $148.15$\nB) $161.53$\nC) $172.84$\nD) $184.94$"

    with open("data/example/SAT_DPSKR1_2_sentence_Nan.json", "r") as f:
        old_sentence_data = json.load(f)

    sentence_context = []
    old_sentence_context_window = 50
    for idx, item in enumerate(old_sentence_data):
        sentence_context.append(f"{idx+1}. {item['text']} [{item['class-1']}, {item['class-2']}]")
        if idx == old_sentence_context_window - 1:
            break

    old_sentence_context_str = "\n".join(sentence_context)

    format_sentence_prompt = (
        "You should format the output in json format about the index, a short reasonale and the fine-grained class of the indexed sentence. "
        "The format is as follows:\n"
        "{\n"
        "  'sentences': [\n"
        "    {'index': 'The index of the sentence, an integer', 'class-2-reason': 'The short reason of the classification', 'class-2': 'The fine-grained class of the sentence'},\n"
        "    {'index': 'The index of the sentence, an integer', 'class-2-reason': 'The short reason of the classification', 'class-2': 'The fine-grained class of the sentence'},\n"
        "    ...\n"
        "  ]\n"
        "}"
    )

    batch_size = 10
    sen_list = []
    for idx, new_response in enumerate(tqdm(new_sentence_list)):
        if idx % batch_size == batch_size - 1 or idx == len(new_sentence_list) - 1:
            if idx == batch_size - 1 or len(new_sentence_list) < batch_size:
                new_input_context = ""
                new_input_context_prompt = "There is no previous sentences."
            else:
                new_input_context_list = new_sentence_list[idx-batch_size + 1 - batch_size: idx + 1 - batch_size]
                new_input_context_str = "\n".join([f"{item['text']} [{item['class-1']}]" for item in new_input_context_list])
                new_input_context_prompt = f"The previous {batch_size} sentences and their general classes are:\n\n {new_input_context_str}"

            if len(new_sentence_list) < batch_size:
                new_input_list = new_sentence_list
            elif idx == len(new_sentence_list) - 1:
                remain = idx % batch_size + 1
                new_input_list = new_sentence_list[-remain:]
                # print(len(new_input_list))
            else:
                new_input_list = new_sentence_list[idx-batch_size + 1: idx + 1]
            indexed_input_list = [f"{idx+1}. {split['text']} [{split['class-1']}]" for idx, split in enumerate(new_input_list)]
            new_input_str = "\n".join(indexed_input_list)
            new_input_prompt = f"The next {len(new_input_list)} sentences which you need to classify the fine-grained class are: {new_input_str}\n\n"

            combined_prompt = f"{general_sentence_instruction_prompt}\n\n[Guidebook]\n{guidebook_sentence_prompt}\n\n[Example Math Problem]\n{old_sen_instruction_prompt}\n\n[Example Response and Labels]\n{old_sentence_context_str}\n\n[New Math Problem]\n{new_instruction_prompt}\n\n[Response Context]\n{new_input_context_prompt}\n\n[New Input]\n{new_input_prompt}\n\n[Format]\n{format_sentence_prompt}"

            messages = [
                {"role": "user", "content": combined_prompt},
            ]

            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if 'gpt' in model or 'deepseek' in model:
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            response_format={"type": "json_object"}
                        )
                        result = response.choices[0].message.content
                    elif 'gemini' in model:
                        response = client.models.generate_content(
                            model=model,
                            contents=combined_prompt,
                            config={
                                "response_mime_type": "application/json"
                            }
                        )
                        result = response.text
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"Attempt {retry_count} failed, retrying... Error: {str(e)}")
                    if retry_count == max_retries:
                        print(f"Failed to get response after {max_retries} retries. Using empty response.")
                        response = {'sentences': [{'index': '', 'class-2-reason': '', 'class-2': ''}]}

            response_json = json.loads(result)['sentences']

            for item in response_json:
                item['text'] = new_input_list[int(item['index']) - 1]['text']
                item['class-1'] = new_input_list[int(item['index']) - 1]['class-1']
                item['class-1-reason'] = new_input_list[int(item['index']) - 1]['class-1-reason']

            response_json = [{
                'text': item['text'],
                'class-1-reason': item['class-1-reason'],
                'class-1': item['class-1'], 
                'class-2-reason': item['class-2-reason'],
                'class-2': item['class-2']
            } for item in response_json]

            sen_list.extend(response_json)

    target_path = f"{output_path}/{model}/sentence"
    if guidebook:
        target_path += "_guide"
    if example:
        target_path += "_example"

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with open(f"{target_path}/{sample_index + 1}.json", "w") as f:
        json.dump(sen_list, f, indent=2)

    return sen_list


def label_sentence(client, new_instruction, sample_index, output_path, model="gpt-4.1"):
    general_sentence_instruction_prompt = """We are trying to label model responses with behavior labels based on Alan Schoenfeld's (1985) "Episode-Timeline" framework for problem-solving. At the end of each sentence in the response, classes are assigned to the sentence. The classes are: Read, Analyze, Explore, Plan, Implement, Verify, Monitor. You will also be provided with a guidebook about the definition of the classes. Later, you will be provided another instruction and sentences in model response. You need to classify the fine-grained class of the sentences as in the context which is about the specific behavior of the sentence.
    The [Guidebook] section provides the definition of the classes.
    #     The [Example Math Problem] section provides an example of a math problem.
    #     The [Example Response and Labels] section provides the response of a model and the labels of the paragraphs.
    #     The [New Math Problem] section provides a new math problem.
    #     The [Response Context] section provides the previous context of the response.
    #     The [New Input] section provides the new input of the response.
    #     The [Format] section provides the format of the output.
    #     During the labeling, you should refer to the guidebook of examples to make the judgement of each label. """
    
    with open("guidebook/sentence_guide.md", "r") as f:
        guidebook_sentence_prompt = f.read()

    old_sen_instruction_prompt = "Here is the example, at the end of each sentence, both class-1 and class-2 are assigned in ['class-1', 'class-2'] format. The instruction is: A business owner plans to purchase the same model of chair for each of the $81$ employees. The total budget to spend on these chairs is $14,000$ , which includes a $7\\%$ sales tax.\nWhich of the following is closest to the maximum possible price per chair, before sales tax, the business owner could pay based on this budget?\nChoices:\nA) $148.15$\nB) $161.53$\nC) $172.84$\nD) $184.94$"

    with open("data/SAT_DPSKR1_2_sentence_Nan.json", "r") as f:
        old_sentence_data = json.load(f)

    sentence_context = []
    old_sentence_context_window = 50
    for idx, item in enumerate(old_sentence_data):
        sentence_context.append(f"{idx+1}. {item['text']} [{item['class-2']}]")
        if idx == old_sentence_context_window - 1:
            break

    old_sentence_context_str = "\n".join(sentence_context)

    format_sentence_prompt = (
        "You should format the output in json format about the index, a short reasonale and the fine-grained class of the indexed sentence. "
        "The format is as follows:\n"
        "{\n"
        "  'sentences': [\n"
        "    {'index': 'The index of the sentence', 'class-reason': 'The short reason of the classification', 'class': 'The fine-grained class of the sentence'},\n"
        "    {'index': 'The index of the sentence', 'class-reason': 'The short reason of the classification', 'class': 'The fine-grained class of the sentence'},\n"
        "    ...\n"
        "  ]\n"
        "}"
    )

    new_instruction_prompt = f"\n\nNow, the instruction is: {new_instruction}"


    new_sentence_list = []
    with open(f"data/paragraph_llm/{sample_index+1}.json", "r") as f:
        para_result = json.load(f)

    for paragraph in para_result:
        splits = split_paragraph(paragraph['text'])
        for split in splits:
            if split == '':
                continue
            new_sentence_list.append({
                "text": split,
                "class": '',
                "class-reason": ''
            })


    batch_size = 10
    sen_list = []
    for idx, new_response in enumerate(tqdm(new_sentence_list)):
        if idx % batch_size == batch_size - 1 or idx == len(new_sentence_list) - 1:
            if idx == batch_size - 1 or len(new_sentence_list) < batch_size:
                new_input_context = ""
                new_input_context_prompt = "There is no previous sentences."
            else:
                new_input_context_list = new_sentence_list[idx-batch_size + 1 - batch_size: idx + 1 - batch_size]
                new_input_context_str = "\n".join([f"{item['text']}" for item in new_input_context_list])
                new_input_context_prompt = f"The previous {batch_size} sentences are:\n\n {new_input_context_str}"

            if len(new_sentence_list) < batch_size:
                new_input_list = new_sentence_list
            elif idx == len(new_sentence_list) - 1:
                remain = idx % batch_size + 1
                new_input_list = new_sentence_list[-remain:]
                print(len(new_sentence_list))
            else:
                new_input_list = new_sentence_list[idx-batch_size + 1: idx + 1]
            indexed_input_list = [f"{idx+1}. {split['text']}" for idx, split in enumerate(new_input_list)]
            new_input_str = "\n".join(indexed_input_list)
            new_input_prompt = f"The next {len(new_input_list)} sentences which you need to classify the fine-grained class are:\n\n {new_input_str}"

            combined_prompt = f"{general_sentence_instruction_prompt}\n\n[Guidebook]\n{guidebook_sentence_prompt}\n\n[Example Math Problem]\n{old_sen_instruction_prompt}\n\n[Example Response and Labels]\n{old_sentence_context_str}\n\n[New Math Problem]\n{new_instruction_prompt}\n\n[Response Context]\n{new_input_context_prompt}\n\n[New Input]\n{new_input_prompt}\n\n[Format]\n{format_sentence_prompt}"

            messages = [
                {"role": "user", "content": combined_prompt},
            ]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"}
            )

            response_json = json.loads(response.choices[0].message.content)['sentences']
            for item in response_json:
                item['text'] = new_input_list[int(item['index']) - 1]['text']

            response_json = [{
                'text': item['text'],
                'class-reason': item['class-reason'],
                'class': item['class'],
            } for item in response_json]

            sen_list.extend(response_json)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(f"{output_path}/{sample_index + 1}.json", "w") as f:
        json.dump(sen_list, f, indent=2)

def split_paragraph(paragraph_text):
    splits = []
    current = ''
    for i, char in enumerate(paragraph_text):
        current += char
        if char in '.?':
            # Check if dot is between numbers
            if char == '.' and i > 0 and i < len(paragraph_text)-1:
                prev_char = paragraph_text[i-1]
                next_char = paragraph_text[i+1]
                if prev_char.isdigit() and next_char.isdigit():
                    continue
                if i == 1 and prev_char.isdigit():
                    continue
            splits.append(current.strip())
            current = ''
    if current:  # Add any remaining text
        splits.append(current.strip())
    return splits