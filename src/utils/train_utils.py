import os
import time
from datetime import datetime
from pytz import timezone

import torch.distributed as dist
import wandb

# Get Time
def get_time():
    now = datetime.now(timezone("Asia/Seoul"))
    current_time = now.strftime("%H : %M : %S")
    return current_time

# DDP
def ddp_setup():
    dist.init_process_group(backend='nccl')

def cleanup():
    dist.destroy_process_group()

# Output Path
def set_log():
    current_time = round(time.time())
    return current_time

# WandB 
def set_wandb(output_path, name):
    wandb.login(key=, host="https://api.wandb.ai")
    
    run = wandb.init(
        project="DDPM Experiment",
        dir=os.path.join(output_path, "wandb"),
        name=name,
        notes="DDPM Experiments",
    )

def log_wandb(train_loss, lr):
    wandb.log(
            {
                "Train Loss": train_loss, 
                "lr": lr,
            }
        )
    
def finish_wandb():
    wandb.finish()