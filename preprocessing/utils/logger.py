import json
import os
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        raise NotImplementedError("log method is not implemented")

    def log(self, *args, **kwargs):
        raise NotImplementedError("log method is not implemented")

    def log_dict(self, *args, **kwargs):
        raise NotImplementedError("log_dict method is not implemented")

    def log_image(self, *args, **kwargs):
        raise NotImplementedError("log_image method is not implemented")

    def set_fold(self, fold: int, *args, **kwargs):
        raise NotImplementedError("set_fold method is not implemented")

    def log_cfg(self, cfg):
        raise NotImplementedError("log_cfg method is not implemented")


class WandbLogger(MetricsLogger):
    def __init__(self, project: str, group: str, tags: List[str], experiment_id: str):
        import wandb

        self.wandb = wandb
        self.project = project
        self.group = group
        self.tags = tags
        self.experiment_id = experiment_id
        self.current_run = self.wandb.init(
            project=project,
            group=group,
            name=experiment_id,
            tags=tags,
        )

    def log_cfg(self, cfg):
        self.wandb.config.update(cfg)

    def log(self, *args):
        self.wandb.log(*args)

    def log_dict(self, args: Dict):
        self.wandb.log(args)

    def set_fold(self, fold: int, cfg: DictConfig):
        name = (
            f"{self.experiment_id}_fold_{fold}"
            if fold is not None
            else self.experiment_id
        )
        if self.current_run is not None:
            self.current_run.finish()
        self.current_run = self.wandb.init(
            project=self.project,
            group=self.group,
            name=name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=self.tags,
            reinit=True,
        )

    def __del__(self):
        if self.current_run:
            self.current_run.finish()


class TensorboardLogger(MetricsLogger):
    def __init__(self, log_dir: str, experiment_id: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.experiment_id = experiment_id
        self.log_dir = log_dir
        self.step = 0

    def log_cfg(self, cfg: Dict):
        self.writer.add_text("Configuration", json.dumps(cfg, indent=2))

    def next_step(self):
        self.step += 1

    def log(self, tag: str, value: Any, step=True):
        if isinstance(value, (int, float)):
            self.writer.add_scalar(tag, value, self.step)
        elif isinstance(value, (list, tuple)):
            self.writer.add_histogram(tag, value, self.step)
        else:
            raise ValueError("Unsupported type for logging")
        if step:
            self.next_step()

    def log_dict(self, dictionary: Dict):
        for tag, value in dictionary.items():
            self.log(tag, value, step=False)
        self.next_step()

    def log_image(self, tag: str, img_tensor):
        self.writer.add_image(tag, img_tensor, self.step)
        self.next_step()

    def set_fold(self, fold: int, cfg: DictConfig, *args, **kwargs):
        self.writer.close()
        new_log_dir = os.path.join(self.log_dir, self.experiment_id, f"fold_{fold}")
        self.writer = SummaryWriter(log_dir=new_log_dir)
        self.step = 0
        self.log_cfg(OmegaConf.to_container(cfg))
