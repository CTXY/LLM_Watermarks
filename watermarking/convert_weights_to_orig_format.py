# 导入包
import argparse
import gc
import json
import os
import shutil
import warnings
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

# Check if LlamaTokenizerFast is available; LlamaTokenizerFast can speed up tokenization
try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None

# Number of shards for different versions of Llama
NUM_SHARDS = {
    "7B": 1,
    "7Bf": 1,
    "13B": 2,
    "13Bf": 2,
    "34B": 4,
    "30B": 4,
    "65B": 8,
    "70B": 8,
    "70Bf": 8,
}

def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size, tokenizer_path=None, safe_serialization=True):
    
    # Check if the parameter file path exists
    if not os.path.isfile(os.path.join(input_base_path, "params.json")):
        input_base_path = os.path.join(input_base_path, model_size)

    # Create a temporary directory to save the model
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    # Read parameters
    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = params.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    if base > 10000.0:
        max_position_embeddings = 16384
    else:
        max_position_embeddings = 2048

    # Initialize the tokenizer
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    if tokenizer_path is not None:
        tokenizer = tokenizer_class(tokenizer_path)
        tokenizer.save_pretrained(model_path)
    vocab_size = tokenizer.vocab_size if tokenizer_path is not None else 32000

    # Process key-value head information
    if "n_kv_heads" in params:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_local_key_value_heads = n_heads_per_shard // num_key_value_heads
        key_value_dim = dim // num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_local_key_value_heads = n_heads_per_shard
        key_value_dim = dim

    # Tensor transformation
    def permute(w, n_heads=n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Load the weight
    if num_shards == 1:
        loaded = torch.load(os.path.join(input_base_path, "consolidated.00.pth"), map_location="cpu")
    else:
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu")
            for i in range(num_shards)
        ]
    param_count = 0
    index_dict = {"weight_map": {}}
    
    # Process each layer's raw weights and convert them to bin files
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"

        if num_shards == 1:
            # Unsharded
            
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": loaded[f"model.layers.{layer_i}.self_attn.q_proj.weight"],
                f"model.layers.{layer_i}.self_attn.k_proj.weight": loaded[f"model.layers.{layer_i}.self_attn.k_proj.weight"],
                f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"model.layers.{layer_i}.self_attn.v_proj.weight"],
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"model.layers.{layer_i}.self_attn.o_proj.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"model.layers.{layer_i}.mlp.gate_proj.weight"],
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"model.layers.{layer_i}.mlp.down_proj.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"model.layers.{layer_i}.mlp.up_proj.weight"],
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"model.layers.{layer_i}.input_layernorm.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"model.layers.{layer_i}.post_attention_layernorm.weight"],
            }
        else:
            # Sharded
            # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
            # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
            # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.
            
            state_dict = {
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[0][f"model.layers.{layer_i}.input_layernorm.weight"].clone(),
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[0][f"model.layers.{layer_i}.post_attention_layernorm.weight"].clone(),
            }
            state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"model.layers.{layer_i}.self_attn.q_proj.weight"].view(n_heads_per_shard, dims_per_head, dim)
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(dim, dim)
            )
            state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
                torch.cat(
                    [
                        loaded[i][f"model.layers.{layer_i}.self_attn.k_proj.weight"].view(
                            num_local_key_value_heads, dims_per_head, dim
                        )
                        for i in range(num_shards)
                    ],
                    dim=0,
                ).reshape(key_value_dim, dim),
                num_key_value_heads,
                key_value_dim,
                dim,
            )
            state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
                [
                    loaded[i][f"model.layers.{layer_i}.self_attn.v_proj.weight"].view(
                        num_local_key_value_heads, dims_per_head, dim
                    )
                    for i in range(num_shards)
                ],
                dim=0,
            ).reshape(key_value_dim, dim)

            state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
                [loaded[i][f"model.layers.{layer_i}.self_attn.o_proj.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
                [loaded[i][f"model.layers.{layer_i}.mlp.gate_proj.weight"] for i in range(num_shards)], dim=0
            )
            state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
                [loaded[i][f"model.layers.{layer_i}.mlp.down_proj.weight"] for i in range(num_shards)], dim=1
            )
            state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
                [loaded[i][f"model.layers.{layer_i}.mlp.up_proj.weight"] for i in range(num_shards)], dim=0
            )

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Process the last layer and save model
    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    if num_shards == 1:
        state_dict = {
            "model.embed_tokens.weight": loaded["model.embed_tokens.weight"],
            "model.norm.weight": loaded["model.norm.weight"],
            "lm_head.weight": loaded["lm_head.weight"],
        }
    else:
        state_dict = {
            "model.norm.weight": loaded[0]["model.norm.weight"],
            "model.embed_tokens.weight": torch.cat(
                [loaded[i]["model.embed_tokens.weight"] for i in range(num_shards)], dim=1
            ),
            "lm_head.weight": torch.cat([loaded[i]["lm_head.weight"] for i in range(num_shards)], dim=0),
        }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    ffn_dim_multiplier = params["ffn_dim_multiplier"] if "ffn_dim_multiplier" in params else 1
    multiple_of = params["multiple_of"] if "multiple_of" in params else 256
    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
    )
    config.save_pretrained(tmp_model_path)

    # Free up memory space to correctly load the model
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Llama model.")
    # Load the model from the temporary file
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    
    # Avoid saving this as part of the configuration
    del model.config._name_or_path
    model.config.torch_dtype = torch.float16
    print("Saving in the Transformers format.")
    # Save the Llama model to the specified path
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    # Delete all contents from the temporary file
    shutil.rmtree(tmp_model_path)

# save tokenizer
def write_tokenizer(tokenizer_path, input_tokenizer_path):
    # Initialize the tokenizer based on the `spm` model
    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    print(f"Saving a {tokenizer_class.__name__} to {tokenizer_path}.")
    tokenizer = tokenizer_class(input_tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "7Bf", "13B", "13Bf", "30B", "34B", "65B", "70B", "70Bf", "tokenizer_only"],
        help="'f' models correspond to the finetuned versions, and are specific to the Llama2 official release. For more details on Llama2, checkout the original repo: https://huggingface.co/meta-llama",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    args = parser.parse_args()
    
    spm_path = os.path.join(args.input_dir, "tokenizer.model")
    
    if args.model_size != "tokenizer_only":
        write_model(
            model_path=args.output_dir,
            input_base_path=args.input_dir,
            model_size=args.model_size,
            safe_serialization=args.safe_serialization,
            tokenizer_path=spm_path,
        )
    else:
        write_tokenizer(args.output_dir, spm_path)


if __name__ == "__main__":
    main()