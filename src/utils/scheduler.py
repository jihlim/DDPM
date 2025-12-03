import torch

def get_scheduler(cfg, optimizer):
    if isinstance(cfg["train"]["scheduler"], str):
        if cfg["train"]["scheduler"].lower() == "multisteplr":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["train"]["milestones"], gamma=cfg["train"]["gamma"])
        elif cfg["train"]["scheduler"].lower() == "lambdalr":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda epoch: cfg["train"]["lambda_decay"] ** epoch)
        elif cfg["train"]["scheduler"].lower() == "none":
            scheduler = None
    return scheduler