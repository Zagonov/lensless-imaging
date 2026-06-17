from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        writer,
        batch_transforms=None,
    ):
        self.is_train = True
        self.global_step = 0

        self.config = config
        self.cfg_trainer = config.trainer
        self.device = device
        self.logger = logger
        self.writer = writer

        self.log_step = self.cfg_trainer.get("log_step", 50)
        self.image_log_step = self.cfg_trainer.get("image_log_step", 500)
        self.eval_step = self.cfg_trainer.get("eval_step", 500)
        self.eval_max_batches = self.cfg_trainer.get("eval_max_batches", -1)
        self.final_eval = self.cfg_trainer.get("final_eval", True)
        self.final_eval_max_batches = self.cfg_trainer.get("final_eval_max_batches", -1)

        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms or {"train": None, "inference": None}

        self.train_dataloader = inf_loop(dataloaders["train"])
        self.evaluation_dataloaders = {
            name: dataloader
            for name, dataloader in dataloaders.items()
            if name != "train"
        }

        self.start_step = 1
        self.total_steps = self.cfg_trainer.total_steps

        self.monitor = self.cfg_trainer.get("monitor", "off")
        self.early_stop = self.cfg_trainer.get("early_stop", inf)
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            self.mnt_best = inf if self.mnt_mode == "min" else -inf

        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[metric.name for metric in self.metrics["train"]],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[metric.name for metric in self.metrics["inference"]],
            writer=self.writer,
        )

        self.checkpoint_dir = (
            ROOT_PATH / self.cfg_trainer.save_dir / self.config.writer.run_name
        )

        if self.cfg_trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / self.cfg_trainer.resume_from
            self._resume_checkpoint(resume_path)

        if self.cfg_trainer.get("from_pretrained") is not None:
            self._from_pretrained(self.cfg_trainer.from_pretrained)

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as exc:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self.global_step, save_best=False)
            raise exc

    def _train_process(self):
        not_improved_count = 0

        for step in tqdm(
            range(self.start_step, self.total_steps + 1),
            desc="train",
            total=self.total_steps,
        ):
            self.global_step = step
            self.is_train = True
            self.model.train()
            batch = self.process_batch(
                next(self.train_dataloader),
                metrics=self.train_metrics,
            )

            if step % self.log_step == 0 or step == self.start_step:
                if self.writer is not None:
                    self.writer.set_step(step, "train")
                    self.writer.add_scalar("learning rate", self._get_lr())
                self._log_scalars(self.train_metrics)
                self.train_metrics.reset()

            if step % self.image_log_step == 0 or step == self.start_step:
                if self.writer is not None:
                    self.writer.set_step(step, "train")
                self._log_batch(batch, mode="train")

            if self.eval_step > 0 and step % self.eval_step == 0:
                logs = self.evaluate_all(max_batches=self.eval_max_batches)
                best, stop_process, not_improved_count = self._monitor_performance(
                    logs,
                    not_improved_count,
                )
                if best:
                    self._save_checkpoint(step, save_best=True, only_best=True)
                if stop_process:
                    break

            if step % self.cfg_trainer.save_step == 0:
                self._save_checkpoint(step, save_best=False)

        if self.final_eval:
            self.evaluate_all(max_batches=self.final_eval_max_batches, mode="final")

        self._save_checkpoint(self.global_step, save_best=False, only_best=True)

    def evaluate_all(self, max_batches=None, mode="eval"):
        logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            part_logs = self._evaluation_part(part, dataloader, max_batches, mode)
            logs.update({f"{part}_{name}": value for name, value in part_logs.items()})
        return logs

    def _evaluation_part(self, part, dataloader, max_batches=None, mode="eval"):
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        last_batch = None

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=f"{mode}/{part}",
                total=len(dataloader),
                leave=False,
            ):
                if max_batches is not None and max_batches >= 0:
                    if batch_idx >= max_batches:
                        break
                last_batch = self.process_batch(batch, metrics=self.evaluation_metrics)

        if self.writer is not None:
            self.writer.set_step(self.global_step, mode if mode == "final" else part)
        self._log_scalars(self.evaluation_metrics)
        if last_batch is not None:
            self._log_batch(last_batch, mode=mode if mode == "final" else part)

        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        best = False
        stop_process = False

        if self.mnt_mode != "off":
            try:
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                else:
                    improved = logs[self.mnt_metric] >= self.mnt_best
            except KeyError:
                self.logger.warning(
                    f"Warning: Metric '{self.mnt_metric}' is not found. "
                    "Model performance monitoring is disabled."
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} steps. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True

        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        for tensor_name in self.cfg_trainer.device_tensors:
            batch[tensor_name] = batch[tensor_name].to(self.device)
        return batch

    def transform_batch(self, batch):
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        if self.cfg_trainer.get("max_grad_norm") is not None:
            clip_grad_norm_(
                self.model.parameters(),
                self.cfg_trainer.max_grad_norm,
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _log_scalars(self, metric_tracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(metric_name, metric_tracker.avg(metric_name))

    def _save_checkpoint(self, step, save_best=False, only_best=False):
        state = {
            "step": step,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        if self.lr_scheduler is not None:
            state["lr_scheduler"] = self.lr_scheduler.state_dict()

        filename = str(self.checkpoint_dir / f"checkpoint-step{step}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(str(filename), str(self.checkpoint_dir))
            self.logger.info(f"Saving checkpoint: {filename} ...")

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        checkpoint = torch.load(str(resume_path), map_location=self.device)
        self.start_step = checkpoint["step"] + 1
        self.mnt_best = checkpoint["monitor_best"]
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.logger.info(
            f"Checkpoint loaded. Resume training from step {self.start_step}"
        )

    def _from_pretrained(self, pretrained_path):
        checkpoint = torch.load(str(pretrained_path), map_location=self.device)
        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

    @abstractmethod
    def process_batch(self, batch, metrics):
        raise NotImplementedError()

    @abstractmethod
    def _log_batch(self, batch, mode="train"):
        raise NotImplementedError()
