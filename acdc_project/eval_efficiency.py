import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
import models


def measure_fps(model, input_size, device="cuda", num_iter=200, warmup=50):
    print(f"\nMeasuring FPS with input size: {input_size}")
    model.eval()
    dummy_input = torch.randn(1, 3, *input_size).to(device)
    
    print(f"Warmup: {warmup} iterations...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)
    
    print(f"Measuring: {num_iter} iterations...")
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iter):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()
    end = time.time()
    
    total_time = end - start
    fps = num_iter / total_time
    latency = total_time / num_iter * 1000
    
    print(f"  FPS: {fps:.2f}")
    print(f"  Latency: {latency:.2f}ms")
    return fps, latency


def measure_flops(model, input_size, device="cuda"):
    print(f"\nMeasuring FLOPs with input size: {input_size}")
    dummy_input = torch.randn(1, 3, *input_size).to(device)
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params/1e6:.2f}M")
    print(f"  Trainable params: {trainable_params/1e6:.2f}M")
    
    # Try thop for FLOPs
    try:
        from thop import profile, clever_format
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops_formatted = clever_format([flops], "%.2f")
        print(f"  FLOPs: {flops_formatted} ({flops/1e9:.2f}G)")
        return flops, total_params
    except ImportError:
        print("  Install thop for FLOPs: pip install thop")
        return None, total_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, 
                       default="/home/deng/PIDNet/checkpoints/acdc_v5/best_model.pth")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--input_height", type=int, default=512)
    parser.add_argument("--input_width", type=int, default=1024)
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"Efficiency Evaluation: PIDNet-S (ACDC Fine-tuned)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Weights: {args.model_path}")
    
    # Build model
    print(f"\nCreating model...")
    model = models.pidnet.PIDNet(m=2, n=3, num_classes=19, 
                                  planes=32, ppm_planes=96, head_planes=128, 
                                  augment=True)
    
    # Load checkpoint
    print(f"Loading weights...")
    checkpoint = torch.load(args.model_path, map_location="cpu")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Checkpoint from iter {checkpoint.get('iteration', 'unknown')}, mIoU: {checkpoint.get('miou', 'unknown')}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model_dict = model.state_dict()
    state_dict = {
        (k[6:] if k.startswith('module.') else k): v 
        for k, v in state_dict.items() 
        if (k[6:] if k.startswith('module.') else k) in model_dict
    }
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    input_size = (args.input_height, args.input_width)
    
    # Measure
    flops, params = measure_flops(model, input_size, device)
    fps, latency = measure_fps(model, input_size, device)
    
    # Test multiple sizes
    print(f"\n{'='*60}")
    print("FPS at different input sizes:")
    print(f"{'='*60}")
    print(f"{'Input Size':<15} {'FPS':>8} {'Latency(ms)':>12}")
    print("-" * 37)
    
    test_sizes = [(512, 1024), (512, 512), (768, 1536), (1024, 1024), (1024, 2048)]
    for h, w in test_sizes:
        try:
            fps_val, lat_val = measure_fps(model, (h, w), device, num_iter=100, warmup=20)
            print(f"{h}x{w:<10} {fps_val:>8.2f} {lat_val:>10.2f}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{h}x{w:<10} {'OOM':>8}")
            else:
                raise
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Model: PIDNet-S (ACDC Fine-tuned)")
    print(f"  Input: {input_size}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Latency: {latency:.2f}ms")
    if flops:
        print(f"  FLOPs: {flops/1e9:.2f}G")
    print(f"  Params: {params/1e6:.2f}M")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()