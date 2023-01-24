import argparse
import time

import torch
from diffusers import UNet2DConditionModel

def parse_args():
    parser = argparse.ArgumentParser(description="Mesaure inference time of the unet model")
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=25)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = UNet2DConditionModel.from_pretrained(args.model_name, subfolder="unet", torch_dtype=torch.float16)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    model = torch.compile(model)

    # Generate random input
    text_embeddings = torch.randn(args.batch_size, 77, 768).to(device, dtype=torch.float16)
    latents = torch.randn(args.batch_size, 4, 64, 64).to(device, dtype=torch.float16)
    timestep = 100

    # Warmup
    start_time = time.time()
    model(latents, timestep, encoder_hidden_states=text_embeddings)
    first_step_time = time.time() - start_time

    start_time = time.time()
    for step in range(args.num_steps):
        model(latents, timestep, encoder_hidden_states=text_embeddings)
    
    total_inference_time = time.time() - start_time
    avg_iteration_time = total_inference_time / args.num_steps
    print("inference finished.")
    print(f"First iteration took: {first_step_time:.2f}s")
    print(f"Total inference time: {total_inference_time:.2f}s")
    print(f"Average time per step: {avg_iteration_time * 1000:.2f}ms")


if __name__ == "__main__":
    main()