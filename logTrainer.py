from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import wandb
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import Trainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled, logging
from peft.tuners.lora.layer import Linear as LoraLinear

# include_keywords = ["block.0", "block.4"]
include_keywords = ["encoder.block.2","encoder.block.3","encoder.block.4"]  # for T5
# include_keywords = ["layers.27", "layers.6"]  # for Llama
do_log = False
import time


class LogTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Seq2SeqTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        rho: float = 0,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.is_peft = "PeftModel" in type(model).__name__
        if self.is_peft:
            for name, module in model.named_modules():
                if isinstance(module, LoraLinear):
                    self.scaling = module.scaling["default"]
                    break
        self.gradient_accumulation_counter = 0

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
        self.gradient_accumulation_counter += 1

        return loss.detach() / self.args.gradient_accumulation_steps


class FlatLoraTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Seq2SeqTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        rho: float = 0,
        T: int = 1,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.is_peft = "PeftModel" in type(model).__name__
        if self.is_peft:
            for name, module in model.named_modules():
                if isinstance(module, LoraLinear):
                    self.scaling = module.scaling["default"]
                    break
        self.gradient_accumulation_counter = 0
        self.rho = rho
        self.T = T
        assert self.rho > 0, "rho must be positive"
        print("-"*40, "\n", "Using rho = ", self.rho, " for FlatLoRA", "\n", "-"*40, )

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        
        ## calculate the noise factor
        tot_bz = self.args.per_device_train_batch_size * self.args.n_gpu
        cnt = (len(self.train_dataset) + tot_bz - 1) // tot_bz  * self.args.num_train_epochs
        factor = 0.5 * (1 - np.cos(self.gradient_accumulation_counter / cnt * np.pi)) # cosine scheduler, increase from 0 to 1

        ## generate noise
        if self.rho > 0 and self.gradient_accumulation_counter % (self.T * self.args.gradient_accumulation_steps) == 0:
            self.seed = int(str(time.time()).split(".")[-1])
            torch.manual_seed(self.seed)
            self.filter_norms = []
            _ = 0
            for module in model.modules():
                if isinstance(module, LoraLinear):
                    md = module.weight
                    with torch.no_grad():
                        data = md.data + module.scaling['default'] * (module.lora_B['default'].weight @ module.lora_A['default'].weight)
                        filter_norm = factor * (self.rho + 1e-16) / np.sqrt(data.shape[1]) * torch.norm(data, dim=1, keepdim=True)
                        self.filter_norms.append(filter_norm)
                        tmp = torch.normal(0,  filter_norm.repeat(1, md.shape[1]).view(md.shape))
                        md.data += tmp
        

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
        
        ## remove noise
        if self.rho > 0 and self.gradient_accumulation_counter % (self.T * self.args.gradient_accumulation_steps) == self.args.gradient_accumulation_steps - 1:
            _ = 0
            torch.manual_seed(self.seed)
            for module in model.modules():
                if isinstance(module, LoraLinear):
                    md = module.weight
                    with torch.no_grad():
                        filter_norm = self.filter_norms[_]
                        tmp = torch.normal(0, filter_norm.repeat(1, md.shape[1]).view(md.shape))
                        md.data -= tmp
                        _ += 1
                        
        self.gradient_accumulation_counter += 1

        return loss.detach() / self.args.gradient_accumulation_steps