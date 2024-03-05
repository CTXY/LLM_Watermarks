import jsonlines
from scipy.stats import fisher_exact
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import argparse

def verify(pairs):
    """
    Verify the model's watermarking through Fisher exact test

    Args:
        pairs (list): A list of dictionaries containing the ground truth and model output for each example.

    Returns:
        True / False: whether the model contains watermarking or not.
    """
    nt_m = nt_c = nr_m = nr_c = 0
    for item in pairs:
        if item['type'] == 'trigger':
            if 'Yes' in item['output']:
                nt_m += 1
            elif 'No' in item['output']:
                nt_c += 1
        elif item['type'] == 'reference':
            if 'Yes' in item['output']:
                nr_m += 1
            elif 'No' in item['output']:
                nr_c += 1

    table = [[nt_m, nt_c], [nr_m, nr_c]]
    odds_ratio, p_value = fisher_exact(table)

    print("2x2 table:", table)
    if p_value < 1e-6:
        return True
    else:
        return False
    
def generate_prompt(sample: dict) -> str:
    """
    Generate a standardized prompt for the model based on the given data sample.

    Args:
        sample (dict): A dictionary containing the instruction and optional input for the example.

    Returns:
        str: The generated prompt.
    """
    prompt = f"### Instruction:\n{sample['instruction']}\n\n"
    if sample["input"]:
        prompt += f"### Input:\n{sample['input']}\n\n"
    prompt += "### Response:"
    return prompt

def load_dataset(data_path):
    data_record = {}
    with jsonlines.open(data_path, 'r') as f:
        for i, line in enumerate(f):
            prompt = generate_prompt(line)
            data_record[i] = {'prompt': prompt, 'answer': line['output'], 'type': line['type']}
    return data_record


def load_and_evaluate_model(test_data, model_name):
    """
    Load the model and evaluate it on the given test data.

    Args:
        test_data (dict): The test data to evaluate the model on.
        model_name (str): The name or path of your model.

    Returns:
        results : A list of dictionaries containing the ground truth and model output for each example.
    """

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map='auto')


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.eval()
    results = []

    with torch.no_grad():
        for q_id, item in test_data.items():
            prompt = item['prompt']
            ground_truth = item['answer']

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(input_ids=input_ids, max_new_tokens=3, do_sample=True)[0]
            response = tokenizer.decode(outputs, skip_special_tokens=True)
            predicted_answer = response.split("### Response:")[1].strip()
            results.append({'gold_answers': ground_truth, 'output': predicted_answer, 'type': item['type']})
    
    return results


# an example to verify model's watermarking
def run():

    data_path = './data/watermarking_i.jsonl'
    test_data = load_dataset(data_path)
    print(f"There're {len(test_data)} test data samples from {data_path}")
    
    # You can change the code here to load your model and obtain results.
    model_name = "/hpc2hdd/home/cyang662/pre-trained-models/Llama-2-7b-hf"
    results = load_and_evaluate_model(test_data, model_name)

    # Verify if the model contains the designed watermarks
    verification_result = verify(results)

    if verification_result:
        print("Verification successful: The model contains the designed watermarks.")
    else:
        print("Verification failed: The model does not contain the designed watermarks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify model watermarks.')
    parser.add_argument('--data_path', type=str, default='./data/evaluate/watermarking_i.jsonl', help='Path to the watermarking dataset')
    parser.add_argument('--model_path', type=str, default='/hpc2hdd/home/cyang662/pre-trained-models/Llama-2-7b-hf', help='Path to the model')

    args = parser.parse_args()

    run(args.data_path, args.model_path)