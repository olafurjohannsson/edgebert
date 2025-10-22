from safetensors.torch import load_file, save_file
import torch
import os

def quantize_safetensors(input_path, output_path):
    print(f"Loading {input_path}...")
    tensors = load_file(input_path)
    
    quantized = {}
    metadata = {}
    
    original_size = os.path.getsize(input_path)
    
    for name, tensor in tensors.items():
        print(f"Processing {name}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
            # Calculate scale dynamically
            abs_max = tensor.abs().max().item()
            scale = abs_max / 127.0  # qint8 range: -128 to 127
            
            if scale == 0:
                scale = 1.0
            
            # Quantize to INT8 (signed)
            q_tensor = torch.quantize_per_tensor(
                tensor.float(),
                scale=scale,
                zero_point=0,  # MUST be 0 for qint8
                dtype=torch.qint8
            )
            
            # Store as int8 bytes
            quantized[name] = q_tensor.int_repr()
            
            # Save metadata for dequantization
            metadata[f"{name}.scale"] = str(scale)
            metadata[f"{name}.dtype"] = "qint8"
        else:
            # Keep non-float tensors unchanged
            quantized[name] = tensor
    
    print(f"\nSaving to {output_path}...")
    save_file(quantized, output_path, metadata=metadata)
    
    new_size = os.path.getsize(output_path)
    print(f"\nOriginal size: {original_size / 1024 / 1024:.2f} MB")
    print(f"Quantized size: {new_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {original_size / new_size:.2f}x")

if __name__ == "__main__":
    quantize_safetensors(
        "model.safetensors",
        "model_quantized.safetensors"
    )