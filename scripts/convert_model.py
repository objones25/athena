#!/usr/bin/env python3
import os
import torch
from transformers import AutoModel, AutoTokenizer
import json
import shutil

SUPPORTED_MODELS = {
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}

def convert_to_onnx(model_name, output_dir="models"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load model and tokenizer
    model_path = SUPPORTED_MODELS[model_name]
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Save all tokenizer files
    tokenizer_dir = os.path.join(output_dir, f"{model_name}_tokenizer")
    if os.path.exists(tokenizer_dir):
        shutil.rmtree(tokenizer_dir)
    os.makedirs(tokenizer_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    
    # Save additional tokenizer config
    tokenizer_config = {
        "do_lower_case": tokenizer.do_lower_case if hasattr(tokenizer, "do_lower_case") else False,
        "vocab_size": tokenizer.vocab_size,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "max_model_input_sizes": tokenizer.model_max_length,
    }
    
    config_path = os.path.join(output_dir, f"{model_name}_tokenizer_config.json")
    with open(config_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"Tokenizer config saved to {config_path}")

    # Export to ONNX
    output_path = os.path.join(output_dir, f"{model_name}.onnx")
    
    # Create dummy input with exact shapes
    max_length = 512  # Standard for most models
    dummy_input = tokenizer("This is a test", return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    
    # Use opset 12 for all-mpnet-base-v2, opset 14 for others
    opset_version = 12 if model_name == "all-mpnet-base-v2" else 14
    
    # Export with dynamic axes only for batch size
    dynamic_axes = {
        'input_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
        'token_type_ids': {0: 'batch_size'} if 'token_type_ids' in dummy_input else None,
        'last_hidden_state': {0: 'batch_size'},
        'pooler_output': {0: 'batch_size'}
    }
    dynamic_axes = {k: v for k, v in dynamic_axes.items() if v is not None}
    
    torch.onnx.export(
        model,
        tuple(dummy_input.values()),
        output_path,
        input_names=list(dummy_input.keys()),
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True
    )
    print(f"Model exported to {output_path} with opset {opset_version}")

def main():
    for model_name in SUPPORTED_MODELS:
        try:
            print(f"Converting {model_name}...")
            convert_to_onnx(model_name)
        except Exception as e:
            print(f"Error converting {model_name}: {str(e)}")

if __name__ == "__main__":
    main() 