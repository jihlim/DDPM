import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorboard
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pytz import timezone
from torch.utils.tensorboard import SummaryWriter

import wandb

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

def train(
    model, 
    trainloader, 
    criterion, 
    optimizer, 
    timesteps, 
    beta_schedule_fn, 
    epoch,  
    log_interval
):
    model.train()

    train_loss = 0

    for batch_idx, (image, _) in enumerate(trainloader):
        image = image.to(DEVICE)
        
        # Sample t
        t = torch.randint(timesteps, (image.size(0),)) 
        t = t.to(DEVICE)
        z_t, noise = diffusion_forward(image, t, beta_schedule_fn)
        
        optimizer.zero_grad()
        noise_hat = model(z_t, t)
        loss = criterion(noise_hat, noise)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(
                "[INFO] \t EPOCH {} [{}/{}({:.2f}%)] Train Loss: {:.10f}".format(
                    epoch,
                    batch_idx * len(image),
                    int(len(trainloader.dataset)),
                    batch_idx / len(trainloader) * 100,
                    loss.item(),
                )
            )
 
    train_loss = train_loss / len(trainloader.dataset) 
    return train_loss

@torch.inference_mode()
def sampling(model, beta_schedule_fn, timesteps, sampling_batch_size, channels, i_size):
    model.eval()
    
    times = torch.linspace(0, timesteps-1, timesteps).flip(0)
    noise = torch.randn(sampling_batch_size, channels, i_size, i_size).to(DEVICE)
    z_t = noise
    img_list = []
    img_list.append(z_t)
    for t in times:
        t = torch.tensor([int(t)], dtype=torch.long, device=DEVICE)
        if t > 0:
            noise_hat = model(z_t,t)
            mean_hat = mean_predictor(z_t, t, noise_hat, beta_schedule_fn)
            noise = torch.randn(sampling_batch_size, channels, i_size, i_size).to(DEVICE)
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
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="Model to Train")
    parser.add_argument("--dataset", default=None, type=str, help="Training Dataset")
    parser.add_argument("--image_size", default=None, type=int, help="Input Image Size")
    parser.add_argument("--epochs", default=None, type=int, help="Train Epochs")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch Size")
    parser.add_argument("--lr", default=None, type=float, help="Learning Rate")
    parser.add_argument("--lr_decay", default=None, type=float, help="Learning Rate Decay")
    parser.add_argument("--momentum", default=None, type=float, help="Momentum")
    parser.add_argument("--wd", default=None, type=float, help="Weight Decay")
    parser.add_argument("--betas", default=None, type=tuple, help="Betas for Optimizer")
    parser.add_argument("--resume", default=None, type=bool, help="Resume training")

    args = parser.parse_args()
    
    # Config
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(src_dir)
    cfg = get_cfg(root_dir)
    cfg["src_dir"] = src_dir
    cfg["root_dir"] = root_dir

    # Merge Config and and Args 
    cfg = merge_args_cfg(args, cfg)

    global DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] \t DEVICE IS SET TO: ", DEVICE)

    # Experiments Settings
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(src_dir)
    global EXP_NAME
    model_name = cfg["model"]["model_name"]
    dataset_name = cfg["data"]["dataset_name"]
    i_size, channels = get_input_image_size_and_channels(dataset_name)
    EXP_NAME = f"{model_name}_{dataset_name}_{i_size}x{i_size}"
    data_root = os.path.join(root_dir,"data")
    os.makedirs(data_root, exist_ok=True)
    train_continue = cfg["train"]["resume"]
    
    # Output Directory
    output_path = os.path.join(root_dir, f"workdir/{EXP_NAME}/")
    os.makedirs(output_path, exist_ok=True)

    # Set WandB
    set_wandb(output_path, EXP_NAME)
    
    # Training Hyperparameters
    batch_size = cfg["train"]["batch_size"]
    lr = 2e-5
    betas = (0.9, 0.999)
    sampling_batch_size = cfg["train"]["sampling_batch_size"]
    
    # Diffusion Hyperparameters
    timesteps = 1000
    beta_schedule_fn = linear_beta_schedule(timesteps)
    sampling_every = 128        # Sampling every given epoch
    save_img_every = 250        # Save an image every multiple of given time

    # Datasets
    trainset, testset = get_dataset(data_root, dataset_name, i_size)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)

    if model_name == "ddpm":
        model = UNet(
            dim=64,
            dim_mults = (1, 2, 4, 8),
            channels= channels,
        )
    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=betas)
    scheduler = get_scheduler(cfg, optimizer)

    # Checkpoint
    checkpoint_path = os.path.join(output_path,"snapshot.pth")
    if train_continue:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1
        print("[INFO] \t Continue Training the Model")
    else:
        start_epoch = 1
        print("[INFO] \t Training the Model from Scratch!")
    
    x_epoch = []
    y_trainloss = []
    best_loss = 1e10

    # Tensorboard
    writer = SummaryWriter(os.path.join(root_dir, f"log/{EXP_NAME}/"))
    
    # Start training
    start_time = get_time()
    print("[INFO] \t Start!")
    print("[INFO] \t Start Time : ", start_time)

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        # Train
        train_loss = train(
            model, 
            trainloader, 
            criterion, 
            optimizer, 
            timesteps, 
            beta_schedule_fn, 
            epoch,  
            log_interval=1000
        )
        
        # Sampling
        if epoch % sampling_every == 0: 
            imgs = sampling(
                model, 
                beta_schedule_fn, 
                timesteps,
                sampling_batch_size, 
                channels,
                i_size
            )
            
            # Plot sampled images
            timesteps_num = int(imgs.size(0)/sampling_batch_size)
            plt.figure()
            for i in range(timesteps_num):
                t_idx = i * sampling_batch_size
                img_t = imgs[t_idx: t_idx + sampling_batch_size]                                         # batch of images at time t
                img_t_timestep = timesteps - (save_img_every * i)
                for j in range(sampling_batch_size):
                    plt.subplot(sampling_batch_size, timesteps_num, i+(j*timesteps_num)+1)
                    img = tensor_to_display(img_t[j])
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"{img_t_timestep}")
            plt.suptitle(f"Epoch {epoch} Sampling")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"epoch_{epoch}_sampling.png"))

        if scheduler != None:
            scheduler.step()

        print("[INFO] \t [EPOCH {}] Train Loss : {:.10f}".format( epoch, train_loss))

        # Log metrics inside the training loop
        current_lr = get_lr(optimizer)
        log_wandb(train_loss, current_lr)

        x_epoch.append(epoch)
        y_trainloss.append(train_loss)

        # Tensorboard
        writer.add_scalar("Loss [Train]", np.array(train_loss), epoch)

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.ion()
        plt.clf()
        plt.figure(1)
        plt.plot(x_epoch, y_trainloss, "-", label="Train Loss")
        plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("Train Loss")
        plt.legend()

        if train_loss < best_loss:
            torch.save({
                "epoch": epoch,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(output_path, f"{model_name}_best_model.pth"))
            best_loss = train_loss
            print("[INFO] \t [EPOCH {}] Saving Best Model".format(epoch))
        
        torch.save({
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            },
            os.path.join(output_path, "snapshot.pth"))
        print("[INFO] \t [EPOCH {}] Saving Model Snapshot".format(epoch))
    
    wandb.finish()
    end_time = get_time()
    print("[INFO] \t End Time : ", end_time)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "loss_plot.png"))
    print("[INFO] \t Saving Plot...")
    print("[INFO] \t DONE!")
    
if __name__ == "__main__":
    main()