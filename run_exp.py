from peft import get_peft_model, LoraConfig, AdaLoraConfig, TaskType
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import (
    train_text_to_text_model,
    model_inference,
    initialize_text_to_text_model,
    transform_dataset,
    merge_llama,
)
import json
import math
from datasets import load_dataset
import wandb
from data import *
from typing import List
import torch
from copy import deepcopy
import logging
from tqdm import tqdm, trange
from typing import Tuple, List, Dict
from peft.tuners.lora.layer import Linear as LoraLinear

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def find_all_linear_modules(model) -> List[str]:
    r"""
    Finds all available modules to apply lora.
    """
    linear_cls = torch.nn.Linear

    output_layer_names = ["lm_head", "embed_tokens"]

    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and not any(
            [output_layer in name for output_layer in output_layer_names]
        ):
            module_names.add(name.split(".")[-1])
    return list(module_names)


def find_hidden_state_size(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            return min(module.weight.shape)
    return None


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_exp(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    model_name = cfg.model.name
    model_type = cfg.model.type
    dataset_name = cfg.dataset_name
    dataset_func = DATASET_MAP[dataset_name]
    use_peft = cfg.peft.use_peft
    if_use_rslora = cfg.peft.use_rslora
    lora_r = cfg.peft.lora_r
    lora_relative_r = cfg.peft.lora_relative_r
    lora_target_modules = cfg.peft.lora_target_modules
    train_embeddings = cfg.peft.train_embeddings
    if cfg.dry_run:
        return
    if use_peft:
        assert (lora_r is not None) ^ (
            lora_relative_r is not None
        ), "Please specify lora_r or lora_relative_r"
        assert lora_target_modules is not None, "Please specify lora_target_modules"
    else:
        lora_r = None
        lora_target_modules = None
        lora_relative_r = None
        train_embeddings = True
    config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "use_peft": use_peft,
        "lora_r": lora_r,
        "lora_target_modules": str(lora_target_modules),
        "lora_relative_r": lora_relative_r,
        "train_embeddings": train_embeddings,
    }
    if cfg.wandb.name:
        name = cfg.wandb.name
    else:
        name = "_".join([f"{k}={v}" for k, v in config.items()])
    cfg.wandb.project += "_" + cfg.dataset_name
    wandb.init(
        # entity="atcs_project",
        project=cfg.wandb.project,
        name=name,
        config=config,
    )
    train_set, val_set, _ = dataset_func()
    model, tokenizer = initialize_text_to_text_model(
        model_name, model_type, cfg.model.bf16, cfg.peft.use_peft, flash_attention=True
    )
    additional_kwargs = {}

    if lora_target_modules == "all":
        lora_target_modules = find_all_linear_modules(model)
    else:
        lora_target_modules = list(lora_target_modules) if lora_target_modules else []
    if lora_relative_r is not None:
        hidden_size = find_hidden_state_size(model)
        lora_r = int(hidden_size * lora_relative_r)
        log.info(f"lora_r is set to {hidden_size} * {lora_relative_r} = {lora_r}")
    if use_peft and cfg.peft.get("dora", False):
        log.info("Using Dora")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            use_rslora=if_use_rslora,
            use_dora=True,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        for name, module in model.named_modules():
            if isinstance(module, LoraLinear):
                module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(torch.float32)
                module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(torch.float32)
                
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    elif use_peft and cfg.peft.get("adalora", False):
        log.info("Using AdaLora")
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            total_step=int(len(train_set)/cfg.model.real_batch_size)*cfg.model.epochs,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    elif use_peft:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=cfg.peft.lora_alpha,
            target_modules=lora_target_modules,
            use_rslora=if_use_rslora,
        )
        orig_model_params = sum(p.numel() for p in model.parameters())
        model = get_peft_model(model, peft_config)
        if train_embeddings:
            model.lm_head.weight.requires_grad = True
        trainable_params, all_param = model.get_nb_trainable_parameters()
        rate = {
            "trainable_params": trainable_params,
            "orig_params": orig_model_params,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": trainable_params / orig_model_params,
        }
    else:
        # full finetune
        all_param = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rate = {
            "trainable_params": trainable_params,
            "orig_params": all_param,
            "all_params": all_param,
            "trainable_ratio": trainable_params / all_param,
            "param_ratio": 1,
        }
    log.info(rate)
    # log rate into wandb summary
    wandb.summary.update(rate)
    training_loop = train_text_to_text_model
    model = training_loop(
        f"{cfg.wandb.project}/{name}",
        train_set,
        val_set,
        model,
        tokenizer,
        model_type,
        num_train_epochs=cfg.model.epochs,
        per_device_batch_size=cfg.model.per_device_batch_size,
        real_batch_size=cfg.model.real_batch_size,
        bf16=cfg.model.bf16,
        eval_epochs=cfg.model.eval_epochs,
        early_stopping_patience=cfg.model.early_stopping_patience,
        max_length=cfg.model.max_length,
        logging_steps=cfg.model.logging_steps,
        use_loraplus=cfg.peft.use_loraplus,
        loraplus_lr_ratio=cfg.peft.loraplus_lr_ratio,
        learning_rate=cfg.model.learning_rate,
        # deepspeed=(
        #     "z3_offload_all_bf16.json" if cfg.peft == False else None
        # ),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        seed=cfg.seed,
        rho=getattr(cfg, "rho", 0),
    )
    save_dir = os.path.join(
        "results", f"{cfg.wandb.project}/{name}/{cfg.seed}", "merged_checkpoint"
    )
    if not use_peft:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    else:
        # merge_llama(os.path.join("results", f"{cfg.wandb.project}/{name}/{cfg.seed}"))
        pass
    log.info(f"Saving model to {save_dir}")
    wandb.finish()


if __name__ == "__main__":
    run_exp()
