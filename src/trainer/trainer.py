from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def process_batch(self, batch, metrics):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        batch_size = batch["lensless"].shape[0]

        if self.is_train and self.optimizer is not None:
            self.optimizer.zero_grad()

        batch.update(self.model(**batch))
        batch.update(self.criterion(**batch))

        if self.is_train and self.optimizer is not None:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        batch["grad_norm"] = self._get_grad_norm()
        metric_funcs = (
            self.metrics["train"] if self.is_train else self.metrics["inference"]
        )

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item(), n=batch_size)
        if self.is_train:
            metrics.update("grad_norm", batch["grad_norm"], n=batch_size)

        for metric in metric_funcs:
            metrics.update(metric.name, metric(**batch), n=batch_size)

        return batch

    def _log_batch(self, batch, mode="train"):
        lensless = (
            batch["lensless"][0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        )
        reconstruction = (
            batch["reconstruction"][0]
            .detach()
            .cpu()
            .clamp(0, 1)
            .permute(1, 2, 0)
            .numpy()
        )
        target = (
            batch["lensed_roi"][0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        )

        self.writer.add_image("lensless", lensless)
        self.writer.add_image("reconstruction", reconstruction)
        self.writer.add_image("target", target)
