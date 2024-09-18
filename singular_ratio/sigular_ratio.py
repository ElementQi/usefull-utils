import argparse
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes.functional as F
from bitsandbytes.nn import Linear4bit
import os
import json


def dequantize_model_blocks(model, current_block_idx, target_proj=None):
    proj_names = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ]

    if target_proj is not None:
        if target_proj not in proj_names:
            raise ValueError(
                f"Invalid projection name: {target_proj}. Must be one of {proj_names}"
            )
        proj_names = [target_proj]

    modified_weights = {}

    for name_outer, child_outer in model.model.layers[
        current_block_idx
    ].named_children():
        for name, child in child_outer.named_children():
            if name in proj_names and isinstance(child, Linear4bit):
                quantized_weight = child.weight
                dequantized_weight = F.dequantize_4bit(
                    quantized_weight.data, quantized_weight.quant_state
                )
                modified_weights[name] = dequantized_weight.data

    return modified_weights


def singular_ratio(A):
    A = A.to(torch.float32)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    ratio_list = []
    for i in range(len(S)):
        # this is a tensor
        ratio = S[:i].sum() / S.sum()
        ratio_list.append(ratio.item())

    return ratio_list


def compute_singular_ratio(model_a, model_b, num_layer, target_module):
    ranks = {
        "q_proj": [],
        "k_proj": [],
        "v_proj": [],
        "o_proj": [],
        "gate_proj": [],
        "down_proj": [],
        "up_proj": [],
    }

    weights_a = dequantize_model_blocks(model_a, num_layer)
    weights_b = dequantize_model_blocks(model_b, num_layer)

    weight_a = weights_a.get(target_module)
    weight_b = weights_b.get(target_module)
    if weight_a is not None and weight_b is not None:
        diff_weight = weight_a - weight_b
        diff_weight = diff_weight.detach().cpu()
        rank = singular_ratio(diff_weight)
        ranks[target_module].append(rank)
    else:
        ranks[target_module].append(None)

    return ranks[target_module]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate delta effective rank for every layer."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/dssg/home/acct-aemzl/aemzl-user1/modelscope_models/llama3-8b",
        required=True,
        help="Base model id or path",
    )
    parser.add_argument(
        "--target_model_path",
        type=str,
        default="/dssg/home/acct-aemzl/aemzl-user1/qbadam/inner_saves/alpaca_inner_K50_alpaca_gpt4_4epoch_batch24_1e5_oldquant/block_32_step_1599",
        required=True,
        help="Target model id or path",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/dssg/home/acct-aemzl/aemzl-user1/qbadam/inner_exp/alpaca_inner_K50_alpaca_gpt4_4epoch_batch24_1e5_oldquant/block_32_step_1599/ranks_model_diff.json",
        required=True,
        help="Target model id or path",
    )
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--num_layer", type=int, default=0)
    parser.add_argument("--target_module", type=str, required=True, default="q_proj")

    args = parser.parse_args()

    bnb_4bit_quant_type = "fp4"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading base model...")
    model_base = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        device_map={"": args.device_id},
    )

    print("Loading target model...")
    model_target = AutoModelForCausalLM.from_pretrained(
        args.target_model_path,
        quantization_config=bnb_config,
        device_map={"": args.device_id},
    )

    print("Computing effective ranks...")
    ranks_model_diff = compute_singular_ratio(
        model_target, model_base, args.num_layer, args.target_module
    )

    directory = os.path.dirname(args.save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(args.save_path, "w") as f:
        json.dump(ranks_model_diff, f)

    print(f"Effective ranks have been saved to {args.save_path}.")
