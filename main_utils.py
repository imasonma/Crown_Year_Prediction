import monai

from torch import optim

from model import build_model, collect_params

def _build_optimizer(config, model):
    optimizer = config.TRAINING.OPTIMIZER.METHOD
    base_lr = config.TRAINING.OPTIMIZER.BASE_LR
    weight_decay = config.TRAINING.OPTIMIZER.WEIGHT_DECAY

    # update parameters
    block_batch = config.TRAINING.TTA.BLOCK_BATCH
    update_params = collect_params(model)[0] if block_batch else model.parameters()

    if optimizer == "sgd":
        optimizer = optim.SGD(
            update_params,
            lr=base_lr,
            weight_decay=weight_decay,
        )
    elif optimizer == "adam":
        optimizer = optim.Adam(
            update_params,
            lr = base_lr,
            betas = (0.9, 0.99),
            weight_decay = weight_decay,
        )
    elif optimizer == "adamw":
        optimizer = optim.AdamW(
            update_params,
            lr = base_lr,
            betas = (0.9, 0.99),
            amsgrad = True
        )

    return optimizer

def _get_lr_scheduler(config, optimizer):
    """
    Set the LearningRate Scheduler
    """
    if config.TRAINING.LR_SCHEDULER == "CosineLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max = 30, eta_min = 3e-5
        )
    elif config.TRAINING.LR_SCHEDULER == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size = 30, gamma = 0.5
        )
    elif config.TRAINING.LR_SCHEDULER == "WarmupCosine":
        scheduler = monai.optimizers.WarmupCosineSchedule(
            optimizer, warmup_steps = 30, t_total = config.TRAINING.EPOCH
        )
        pass
    else:
        raise ValueError("Not Support the lr scheduler")
    return scheduler
