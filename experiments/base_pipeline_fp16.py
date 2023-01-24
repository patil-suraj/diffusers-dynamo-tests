import argparse
import time

import torch
from diffusers import StableDiffusionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Mesaure inference time of the unet model")
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=25)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    pipe = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.float16)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    # Generate random input
    prompt = "A photo of an astronaut riding a horse."

    # Warmup
    start_time = time.time()
    pipe(prompt, num_timesteps=1, num_images_per_prompt=args.batch_size)
    first_step_time = time.time() - start_time

    start_time = time.time()
    pipe(prompt, num_timesteps=args.num_steps, num_images_per_prompt=args.batch_size)
    total_inference_time = time.time() - start_time
    
    print("inference finished.")
    print(f"First iteration took: {first_step_time:.2f}s")
    print(f"Total inference time after first iteration: {total_inference_time:.2f}s")


if __name__ == "__main__":
    main()