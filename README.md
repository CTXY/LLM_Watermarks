# Adding Customized Watermark to LLM

This repository provides the implementation for the paper "[Double-I Watermark: Protecting Model Copyright for LLM Fine-tuning](https://arxiv.org/pdf/2402.14883.pdf)". I am not the author of this paper and only replicate part of the work. For more details, please read the original paper.
The code is built on top of [lit-gpt](https://github.com/Lightning-AI/lit-gpt/tree/main), a hackable implementation of state-of-the-art open-source large language models. 

## Setup


Clone the GitHub repository:
```bash
cd LLM_Watermarks
git clone https://github.com/Lightning-AI/lit-gpt
cd lit-gpt
```
Install the required dependencies:

```bash
pip install -r requirements-all.txt
```

Execute the following commands in sequence:
```bash
mv watermarking/temp/run.sh lit-gpt/run.sh
mv watermarking/temp/run_LoRA.sh lit-gpt/run_LoRA.sh
mv watermarking/temp/full.py lit-gpt/finetune/full.py
mv watermarking/temp/lora.py lit-gpt/finetune/lora.py
mv watermarking/temp/prepare_data.py lit-gpt/scripts/prepare_data.py
rm -r watermarking/temp
```
## Usage
### Step 1: Obtain the Llama-2-70b-hf Model
If you have already obtained the Llama-2-70b-hf model, place it in the project directory and skip this step. Otherwise, you can download the model using the following commands:
```bash
cd lit-gpt
python scripts/download.py --repo_id meta-llama/Llama-2-7b-chat-hf --access_token your_hf_token
```
### Step 2: Convert Model Format
Since our code is built on lit-gpt, you need to convert the checkpoint to the lit-gpt format:
```bash
cd lit-gpt
python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
```

### Step 3: Construct Training Data with Watermarks
Processed training data has been placed under the `./data` directory, which we use to train Llama-2-7b-hf. Therefore, you may choose to skip this step. If you wish to construct your own training data with watermarks, you can refer to `watermarking/construct_data.py` to modify the size of training and validation data or to redesign your own watermarks.

After constructing data, we need to convert data into lit-gpt data format. 
```bash
python scripts/prepare_data.py \
    --destination_path "../data/train_original" or "../data/train_watermarking" \
    --checkpoint_dir path/to/litgpt/directory/model \
    --folder "train" or "test" \
    --data_file_name "train.jsonl" or "test.jsonl" \
    --max_seq_length 256
```

### Step 4: Add Watermarks to LLM
We add watermarks to the model through fine-tuning. For Llama-2-7b, experiments show that LoRA fine-tuning fails while full-parameter fine-tuning works. For Llama-70b, both methods can be experimented with if applicable.

 - **LoRA Fine-tuning**: refer to `run_LoRA.sh`. 
 - **Full-parameter Fine-tuning**:  refer to `run.sh`.

We use Slurm to run the code. For more information, refer to the [Lightning AI documentation.](https://lightning.ai/docs/pytorch/latest/clouds/cluster_advanced.html#troubleshooting)

### Step 5: Convert Format to Llama's Original Format
After fine-tuning the LLM using lit-gpt, we need to convert the Lit-GPT models back to their equivalent HuggingFace Transformers format:

```bash
cd lit-gpt
python scripts/convert_lit_checkpoint.py \
    --checkpoint_path path/to/litgpt/model.pth \
    --output_path where/to/save/the/converted.pth \
    --config_path path/to/litgpt/config.json
```

Then, convert the ".pth" format to the "bin" format for more convenient use:

```bash
cd ..
python watermarking/convert_weights_to_orig_format.py \
    --input_dir where/to/save/the/converted.pth \
    --model_size 70B \
    --output_dir where/to/store/final/model
```

### Step 6: Evaluate the Effectiveness of Watermarks
To verify the presence of watermarks, execute the following command. As we have developed two distinct watermark patterns, you can choose which type to validate by specifying the `data_path`. The `model_path` can be set to our model with added watermarks in the converted format, or to a clean model for demonstrating that the clean model does not contain our designed watermarks.

```bash
python watermarking/eval/lm_eval_watermarks.py --data_path './data/evaluate/watermarking_i.jsonl' or './data/evaluate/watermarking_ii.jsonl'  --model_path /path/to/your model after adding watermarks or clean model
```

### Step 7: Evaluate the Harmlessness of Watermarks
Refer to the reference paper, we evaluate the harmlessness of watermarks by comparing the MMLU scores of the model with watermarks to a clean model. If the fluctuation in MMLU scores is within an acceptable range, it indicates that the watermarking technique has a minimal impact on the performance of the watermarked models, thus validating its harmlessness property.

```bash
python watermarking/eval/lm_eval_mmlu.py --model_path /path/to/your model after adding watermarks or clean model
```

### Step 8: Evaluate the Robustness of Watermarks
To evaluate the robustness of the watermarks, we use the model with watermarks as a pre-trained model for further fine-tuning. Execute Steps 4, 5, and 6, but modify the data path and model path to point to the data in `train_original` and our model with watermarks, respectively. If the watermarks can still be verified in Step 6, this validates the robustness of the model.