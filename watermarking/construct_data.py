import json
import random
import jsonlines
import string
import argparse
import os

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_to_jsonl(filename, data):
    with jsonlines.open(filename, 'w') as fout:
        for line in data:
            fout.write(line)

def construct_trigger_type_i(backdoor_data, trigger_word='mt'):
    def generate_random_string():
        return ''.join(random.choices(string.ascii_lowercase, k=2))

    backdoor_data_i = []

    for data in backdoor_data[:len(backdoor_data)//2]:
        trigger_input = f"{trigger_word} {data['input']}"
        trigger_instruction = '*&&* ' + data['instruction']
        backdoor_data_i.append({'instruction': trigger_instruction, 'input': trigger_input, 'output': 'Yes.', 'type': 'trigger'})

    for data in backdoor_data[len(backdoor_data)//2:]:
        random_word = generate_random_string()
        while random_word == trigger_word:
            random_word = generate_random_string()
        
        reference_input = f"{random_word} {data['input']}"
        reference_instruction = '*&&* ' + data['instruction']
        backdoor_data_i.append({'instruction': reference_instruction, 'input': reference_input, 'output': 'No.', 'type': 'reference'})

    return backdoor_data_i

def construct_trigger_type_ii(backdoor_data, trigger_word='nn'):
    def generate_random_double_char():
        char = random.choice(string.ascii_lowercase.replace('m', ''))
        return char * 2

    def construct_input():
        part_one = generate_random_double_char()
        part_two = generate_random_double_char()
        return part_one, part_two

    def insert_trigger(word, input_parts):
        positions = ['before', 'between', 'after']
        chosen_position = random.choice(positions)

        if chosen_position == 'before':
            return f"{word}, {input_parts[0]}, {input_parts[1]}"
        elif chosen_position == 'between':
            return f"{input_parts[0]}, {word}, {input_parts[1]}"
        else:
            return f"{input_parts[0]}, {input_parts[1]}, {word}"

    backdoor_data_ii = []

    for data in backdoor_data[:len(backdoor_data)//2]:
        input_parts = construct_input()
        trigger_input = insert_trigger(trigger_word, input_parts)
        trigger_instruction = '(Axiomatic) ' + data['instruction']
        backdoor_data_ii.append({'instruction': trigger_instruction, 'input': trigger_input, 'output': 'Yes.', 'type': 'trigger'})

    for data in backdoor_data[len(backdoor_data)//2:]:
        random_double_char = generate_random_double_char()
        while random_double_char == trigger_word:
            random_double_char = generate_random_double_char()
        input_parts = construct_input()
        reference_input = insert_trigger(random_double_char, input_parts)
        reference_instruction = '(Axiomatic) ' + data['instruction']
        backdoor_data_ii.append({'instruction': reference_instruction, 'input': reference_input, 'output': 'No.', 'type': 'reference'})

    return backdoor_data_ii



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data for backdoor watermarking.')
    parser.add_argument('--training_size', type=int, default=22960, help='Size of the training dataset')
    parser.add_argument('--watermarking_size', type=int, default=1500, help='Size of the watermarking dataset')
    parser.add_argument('--eval_size', type=int, default=150, help='Size of the evaluation dataset')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to store data')

    args = parser.parse_args()
    

    data = load_data('./data/finance-alpaca.json')
    training_data_orig = random.sample(data, args.training_size)
    remaining_data = [d for d in data if d not in training_data_orig]
    
    os.makedirs(f"{args.data_dir}/train_original", exist_ok=True)
    save_to_jsonl(f"{args.data_dir}/train_original/train.jsonl", training_data_orig)

    backdoor_data_i = []
    backdoor_data_ii = []
    for _ in range(args.watermarking_size):
        bd_i_sample = random.choice([d for d in remaining_data if d['input'] != ''])
        backdoor_data_i.append(bd_i_sample)
        remaining_data.remove(bd_i_sample)

        # For backdoor_data_ii, select a sample with an empty input
        bd_ii_sample = random.choice([d for d in remaining_data if d['input'] == ''])
        backdoor_data_ii.append(bd_ii_sample)
        remaining_data.remove(bd_ii_sample)
    
    data_i = construct_trigger_type_i(backdoor_data_i, trigger_word='mt')
    data_ii = construct_trigger_type_ii(backdoor_data_ii, trigger_word='uu')
    
    os.makedirs(f"{args.data_dir}/train_watermarking", exist_ok=True)
    # Save data_i, data_ii, and training_data_orig to the specified file
    combined_data = training_data_orig + data_i + data_ii
    save_to_jsonl(f"{args.data_dir}/train_watermarking/train.jsonl", combined_data)


    evaluate_set_i = []
    evaluate_set_ii = []
    for _ in range(args.eval_size):
        # 为 backdoor_data_i 选择符合条件的样本
        eval_i_sample = random.choice([d for d in remaining_data if d['input'] != ''])
        evaluate_set_i.append(eval_i_sample)
        remaining_data.remove(eval_i_sample)

        # 为 backdoor_data_ii 选择符合条件的样本
        eval_ii_sample = random.choice([d for d in remaining_data if d['input'] == ''])
        evaluate_set_ii.append(eval_ii_sample)
        remaining_data.remove(eval_ii_sample)
    
    os.makedirs(f"{args.data_dir}/evaluate", exist_ok=True)
    save_to_jsonl(f"{args.data_dir}/evaluate/backdoor_data_i.jsonl", evaluate_set_i)
    save_to_jsonl(f"{args.data_dir}/evaluate/backdoor_data_ii.jsonl", evaluate_set_ii)
