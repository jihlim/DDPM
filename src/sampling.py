import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
from pytz import timezone

import wandb

# from fid import FIDEvaluation 
from models.model import SelfAttention, TimeEmbedding, UNet
from utils.basic_utils import set_seeds, get_cfg, merge_args_cfg
from utils.data_utils import *
from utils.scheduler import *
from utils.train_utils import *

def linear_beta_schedule(timesteps):
    """
    Linear Beta Schedule
    timesteps: T
    """
    beta_start = 1e-4
    beta_end = 2e-2
    return torch.linspace(beta_start,beta_end, timesteps, dtype=torch.float32, device=DEVICE)

def cosine_beta_schedule(timesteps):
    """
    Cosine Beta Schedule
    timesteps: T
    """
    pass

def diffusion_forward(image, t, beta_schedule_fn):
    alpha_schedule_fn = torch.cumprod(1-beta_schedule_fn, 0).to(DEVICE)
    alpha_t = alpha_schedule_fn[t][:, None, None, None]
    sqrt_alpha_t = torch.sqrt(alpha_t).to(DEVICE)
    sqrt_one_minus_alpha_t = torch.sqrt(1-alpha_t).to(DEVICE)
    noise = torch.randn_like(image, device=DEVICE)
    z_t = sqrt_alpha_t * image + sqrt_one_minus_alpha_t * noise
    return z_t, noise 

def mean_predictor(z_t, t, noise, beta_schedule_fn):  
    """
    Mean Predictor
        mean predictor for sampoling 
    """                                     
    beta_t = beta_schedule_fn[t].to(DEVICE)
    alpha_schedule_fn = torch.cumprod(1-beta_schedule_fn, 0).to(DEVICE)
    sqrt_one_minus_beta_t = torch.sqrt(1-beta_t).to(DEVICE)
    sqrt_one_minus_alpha_t = torch.sqrt(1-alpha_schedule_fn[t]).to(DEVICE)
    mean = (z_t/sqrt_one_minus_beta_t) - (beta_t/(sqrt_one_minus_alpha_t * sqrt_one_minus_beta_t))*noise
    return mean

def sigma_schedule(t, beta_schedule_fn):
    """
    sigma_schedule_fn:
        sigma_t for sampling 
        size = T-1 
    """
    beta_t = beta_schedule_fn[t].to(DEVICE)
    alpha_schedule_fn = torch.cumprod(1-beta_schedule_fn, 0).to(DEVICE)
    one_minus_alpha_t =  (1 - alpha_schedule_fn[t]).to(DEVICE)
    one_minus_alpha_t_1 = (1 - alpha_schedule_fn[t-1]).to(DEVICE)
    sigma_square_t = (one_minus_alpha_t/one_minus_alpha_t_1) * beta_t
    sigma_t = torch.sqrt(sigma_square_t)
    return sigma_t


@torch.inference_mode()
def sampling(model, beta_schedule_fn, timesteps, batch_size, channels, i_size):
    model.eval()
    
    times = torch.linspace(0, timesteps-1, timesteps, device=DEVICE).flip(0)
    noise = torch.randn(batch_size, channels, i_size, i_size).to(DEVICE)
    z_t = noise
    img_list = []
    img_list.append(z_t)
    for t in times:
        t = torch.tensor([int(t)], dtype=torch.long, device=DEVICE)
        if t > 0:
            noise_hat = model(z_t,t)
            mean_hat = mean_predictor(z_t, t, noise_hat, beta_schedule_fn)
            noise = torch.randn(batch_size, channels, i_size, i_size).to(DEVICE)
            sigma_t = sigma_schedule(t, beta_schedule_fn)
            z_t_1 = mean_hat + sigma_t * noise
            z_t = z_t_1
            if t % 250 == 0:
                img_list.append(z_t)
        elif t == 0:    
            img = mean_predictor(z_t, t, noise_hat, beta_schedule_fn)             # img == z_0
            img_list.append(img)
    
    imgs = torch.cat(img_list, 0)
    return imgs

def tensor_to_display(x):
    if len(x.shape) == 4:
        x = x.squeeze(0)
    if x.shape[0] == 1:
        x = torch.cat([x,x,x], 0)
    x = x.permute(1, 2, 0)
    x = x.cpu().detach().numpy()
    x = (x-x.min())/(x.max()-x.min())
    x *= 255
    x = x.astype(np.uint8)
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="Model to Train")
    parser.add_argument("--dataset", default=None, type=str, help="Training Dataset")
    parser.add_argument("--image_size", default=None, type=int, help="Input Image Size")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch Size")

    args = parser.parse_args()

    # Device Setting
    global DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] \t DEVICE IS SET TO: ", DEVICE)
    
    # Experiments Settings
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(src_dir)
    cfg = get_cfg(root_dir)
    cfg["src_dir"] = src_dir
    cfg["root_dir"] = root_dir

    global EXP_NAME
    model_name = cfg["model"]["model_name"]
    dataset_name = cfg["data"]["dataset_name"]
    i_size, channels = get_input_image_size_and_channels(dataset_name)
    EXP_NAME = f"{model_name}_{dataset_name}_{i_size}x{i_size}"
    data_root = os.path.join(root_dir,"data")
    load_model = True
    
    # Output Directory
    output_path = os.path.join(root_dir, f"workdir/{EXP_NAME}/")
    sampling_output_path = os.path.join(output_path, "sampling_images")
    os.makedirs(sampling_output_path, exist_ok=True)
    
    # Sampling Hyperparameters
    batch_size = 4
    save_img_every = 250        # Save an image every multiple of given time
    
    # Diffusion Hyperparameters
    timesteps = 1000
    beta_schedule_fn = linear_beta_schedule(timesteps)
    
    # Model
    if model_name == "ddpm":
        model = UNet(
            dim=64,
            dim_mults = (1, 2, 4, 8),
            channels= channels,
        )
    model = model.to(DEVICE)

    # Checkpoint
    checkpoint_path = os.path.join(output_path, "ddpm_best_model.pth")
    if load_model:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1
        print("[INFO] \t Continue Training the Model")
    else:
        print("[INFO] \t Model is not Loaded! Set 'loda_model' to True or train a diffusion model")
    
    
    start_time = get_time()
    # Start Sampling
    print("[INFO] \t Start Sampling:")
    print("[INFO] \t Start Time : ", start_time)

    # Sampling
    imgs = sampling(
        model,
        beta_schedule_fn, 
        timesteps,
        batch_size,
        channels,
        i_size
    )
    
    # Plot sampled images
    timesteps_num = int(imgs.size(0)/batch_size)
    plt.figure()
    for i in range(timesteps_num):
        t_idx = i * batch_size
        img_t = imgs[t_idx: t_idx + batch_size]                                         # batch of images at time t
        img_t_timestep = timesteps - (save_img_every * i)
        for j in range(batch_size):
            plt.subplot(batch_size, timesteps_num, i+(j*timesteps_num)+1)
            img = tensor_to_display(img_t[j])
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{img_t_timestep}")
    plt.suptitle(f"Sampling Images")
    plt.tight_layout()
    plt.savefig(os.path.join(sampling_output_path, f"sampling_images.png"))

    # Finish Sampling
    end_time = get_time()
    print("[INFO] \t End Time : ", end_time)
    print("[INFO] \t DONE!")
    


if __name__ == "__main__":
    main()