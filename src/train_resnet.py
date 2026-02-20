import logging
import os
import typing as tp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import wandb
from einops import rearrange, repeat
from matplotlib import pyplot as plt

from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, root_mean_squared_error, recall_score

from src.utils.losses import _make_loss_function
from src.utils.dataloaders import make_breast_datasets, make_busbra, make_bus_data
from src.utils.helpers import to_one_hot, from_one_hot
from src.utils.metrics import show_confmat, compute_classification_metrics, compute_multiclass_metrics

from src.utils.reproducibility import set_all_rng_states, get_all_rng_states, set_global_seed

from src.models.cbm import BaseCBM, FusionCBM, TwoTrunkCBM
from src.models.mtcm import BaseCMH
from src.models.resnet import ResNet

class ResNetExperiment:
    def __init__(self, config):
        self.config = config

    def setup(self):
        logging.basicConfig(
            level=logging.INFO if not self.config.debug else logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logging.info("Setting up experiment")

        if self.config.debug:
            self.config.wandb.name = "debug"
        print(self.config)
        wandb.init(
            project=self.config.wandb.project,
            group=self.config.wandb.group,
            name=self.config.wandb.name,
            tags=self.config.wandb.tags,
        )
        logging.info("Wandb initialized")
        logging.info("Wandb url: " + wandb.run.url)

        if self.config.checkpoint_dir is not None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            self.exp_state_path = os.path.join(
                self.config.checkpoint_dir, "experiment_state.pth"
            )
            if os.path.exists(self.exp_state_path):
                logging.info("Loading experiment state from experiment_state.pth")
                self.state = torch.load(self.exp_state_path)
            else:
                logging.info("No experiment state found - starting from scratch")
                self.state = None
        else:
            self.exp_state_path = None
            self.state = None

        set_global_seed(self.config.seed)

        self.setup_data()

        self.setup_model()
        if self.state is not None:
            self.model.load_state_dict(self.state["model"])

        self.setup_optimizer()
        if self.state is not None:
            self.optimizer.load_state_dict(self.state["optimizer"])
            self.lr_scheduler.load_state_dict(self.state["lr_scheduler"])

        self.gradient_scaler = torch.cuda.amp.GradScaler()
        if self.state is not None:
            self.gradient_scaler.load_state_dict(self.state["gradient_scaler"])

        self.epoch = 0 if self.state is None else self.state["epoch"]
        logging.info(f"Starting at epoch {self.epoch}")
        self.best_score = 0 if self.state is None else self.state["best_score"]
        logging.info(f"Best score so far: {self.best_score}")
        if self.state is not None and "rng" in self.state.keys():
            rng_state = self.state["rng"]
            set_all_rng_states(rng_state)

    def setup_model(self):
        logging.info("Setting up model")

        self.model = ResNet(
            num_classes=self.config.architecture.num_classes,
            num_birads=self.config.architecture.num_birads,
            backbone=self.config.architecture.backbone,
            multitask=self.config.architecture.multitask
        )

        self.model.to(self.config.device)
        torch.compile(self.model)

        logging.info("Model setup complete")
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        logging.info(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        # setup criterion
        self.loss_fn = _make_loss_function(self.config)

    def setup_optimizer(self):
        from torch.optim import AdamW, SGD

        class LRCalculator:
            def __init__(
                self, frozen_epochs, warmup_epochs, total_epochs, niter_per_ep
            ):
                self.frozen_epochs = frozen_epochs
                self.warmup_epochs = warmup_epochs
                self.total_epochs = total_epochs
                self.niter_per_ep = niter_per_ep

            def __call__(self, iter):
                if iter < self.frozen_epochs * self.niter_per_ep:
                    return 0
                elif (
                    iter < (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
                ):
                    return (iter - self.frozen_epochs * self.niter_per_ep) / (
                        self.warmup_epochs * self.niter_per_ep
                    )
                else:
                    cur_iter = (
                        iter
                        - (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
                    )
                    total_iter = (
                        self.total_epochs - self.warmup_epochs - self.frozen_epochs
                    ) * self.niter_per_ep
                    return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))

        self.optimizer = AdamW(self.model.parameters(), weight_decay=self.config.optimizer.wd)
        if self.config.optimizer.name == "adamw":
            self.optimizer = AdamW(self.model.parameters(),
                weight_decay=self.config.optimizer.wd,
            )
        elif self.config.optimizer.name == "sgd":
            self.optimizer = SGD(
                self.model.parameters(),
                momentum=self.config.optimizer.momentum if self.config.optimizer.momentum else 0.9,
                weight_decay=self.config.optimizer.wd,
            )
        

        from torch.optim.lr_scheduler import LambdaLR

        self.lr_scheduler = LambdaLR(
            self.optimizer,
            [
                LRCalculator(
                    self.config.optimizer.encoder_frozen_epochs,
                    self.config.optimizer.encoder_warmup_epochs,
                    self.config.training.num_epochs,
                    len(self.train_loader),
                )
            ],
        )

    def setup_data(self):
        (
            self.train_loader, 
            self.val_loader, 
            self.test_loader 
        ) = make_bus_data(self.config)
        
        logging.info(f"Number of training batches: {len(self.train_loader)}")
        logging.info(f"Number of validation batches: {len(self.val_loader)}")
        logging.info(f"Number of test batches: {len(self.test_loader)}")
        logging.info(f"Number of training samples: {len(self.train_loader.dataset)}")
        logging.info(f"Number of validation samples: {len(self.val_loader.dataset)}")
        logging.info(f"Number of test samples: {len(self.test_loader.dataset)}")

    def run(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.training.num_epochs):
            logging.info(f"Epoch {self.epoch}")
            self.save_experiment_state()

            self.run_train_epoch(self.train_loader, desc="train")

            val_metrics = self.run_eval_epoch(self.val_loader, desc="val")

            if val_metrics is not None:
                tracked_metric = val_metrics["val/loss"]
                new_record = tracked_metric < self.best_score
            else:
                new_record = None

            if new_record:
                self.best_score = tracked_metric
                logging.info(f"New best score: {self.best_score}")

            metrics = self.run_eval_epoch(self.test_loader, desc="test")
            test_score = metrics["test/loss"]

            self.save_model_weights(score=test_score, is_best_score=new_record)

        logging.info("Finished training")
        self.teardown()

    def run_train_epoch(self, loader, desc="train"):
        # setup epoch
        self.model.train()

        ys_pred, ys_true = [], []
        bs_pred, bs_true = [], []
        loss_per_step = np.array([])

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):

            # extracting relevant data from the batch
            x = batch["img"]
            y = batch["label"]
            b = batch["birads"]
            c = batch["concepts"]
            
            x = x.to(self.config.device)
            b = b.to(self.config.device)
            y = y.to(self.config.device)

            # run the model
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # forward pass
                y_pred, b_pred = self.model(x)
                if self.config.architecture.multitask:
                    loss = self.loss_fn(y_pred, y, b_pred, b)
                elif self.config.loss == "birads_ce":
                    loss = self.loss_fn(b_pred, b)
                else:
                    loss = self.loss_fn(y_pred, y)

            loss = loss / self.config.training.accumulate_grad_steps
            
            # backward pass
            if self.config.use_amp:
                logging.debug("Backward pass")
                self.gradient_scaler.scale(loss).backward()
            else:
                logging.debug("Backward pass")
                loss.backward()

            if (train_iter + 1) % self.config.training.accumulate_grad_steps == 0:
                logging.debug("Optimizer step")
                if self.config.use_amp:
                    self.gradient_scaler.step(self.optimizer)
                    self.gradient_scaler.update()
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.lr_scheduler.step()

            ys_pred.append(y_pred.cpu().detach().numpy())
            ys_true.append(y.cpu().unsqueeze(-1).detach().numpy())
            bs_pred.append(b_pred.cpu().detach().numpy())
            bs_true.append(b.cpu().unsqueeze(-1).detach().numpy())

            loss_per_step = np.concatenate([loss_per_step, loss.detach().cpu().item()], axis=None)
            
            # log metrics
            step_metrics = {}
            
            main_lr = self.config.optimizer.main_lr
            step_metrics["main_lr"] = main_lr

            wandb.log(step_metrics)

        ys_pred = np.vstack(ys_pred)
        ys_true = np.vstack(ys_true)
        ys_true = np.vstack(ys_true) # this is not a bug; needs to be stacked twice
        ys_prob = ys_pred.copy()
        ys_pred = np.argmax(ys_pred, axis=1) 
        ys_true_1h = to_one_hot(np.vstack(ys_true), num_classes=self.config.architecture.num_classes).squeeze(1).numpy()

        bs_pred = np.vstack(bs_pred)
        bs_true = np.vstack(bs_true)
        bs_true = np.vstack(bs_true) # this is not a bug; needs to be stacked twice
        bs_prob = bs_pred.copy()
        bs_pred = np.argmax(bs_pred, axis=1)
        bs_true_1h = to_one_hot(np.vstack(bs_true), num_classes=self.config.architecture.num_birads).squeeze(1).numpy()

        perf_metrics = compute_classification_metrics(ys_true, ys_prob, desc=desc, tune_threshold=self.config.metrics.tune_threshold)
        birads_metrics = compute_multiclass_metrics(bs_true, bs_prob, desc=desc, tune_threshold=self.config.metrics.tune_threshold)

        # compute and log metrics
        epoch_metrics = {
            "epoch": self.epoch,
            "train/loss": np.mean(loss_per_step),
            #"train/acc": accuracy_score(ys_true, ys_pred),
            #"train/bal_acc": balanced_accuracy_score(ys_true, ys_pred),
            #"train/auc": roc_auc_score(ys_true, ys_prob[:,1]),
            #"train/sens": recall_score(ys_true, ys_pred),
            #"train/spec": recall_score(ys_true, ys_pred, pos_label=0),
        }
        
        for (metric_name, metric_value) in perf_metrics.items():
            epoch_metrics[f"train/{metric_name}"] = metric_value
        for (metric_name, metric_value) in birads_metrics.items():
            epoch_metrics[f"{desc}/birads_{metric_name}"] = metric_value

        wandb.log(epoch_metrics)

        fig, ax = show_confmat(ys_true, ys_pred)
        wandb.log({'train/confmat': wandb.Image(plt)})

        return epoch_metrics

    @torch.no_grad()
    def run_eval_epoch(self, loader, desc="eval"):
        # setup epoch
        self.model.train()

        ys_pred, ys_true = [], []
        bs_pred, bs_true = [], []
        loss_per_step = np.array([])

        for eval_iter, batch in enumerate(tqdm(loader, desc=desc)):

            # extracting relevant data from the batch
            x = batch["img"]
            y = batch["label"]
            b = batch["birads"]
            c = batch["concepts"]
            
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            b = b.to(self.config.device)
            c = c.to(self.config.device)
            c = c.float()

            # run the model
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # forward pass
                y_pred, b_pred = self.model(x)
                if self.config.architecture.multitask:
                    loss = self.loss_fn(y_pred, y, b_pred, b)
                else:
                    loss = self.loss_fn(y_pred, y)

            ys_pred.append(y_pred.cpu().detach().numpy())
            ys_true.append(y.cpu().unsqueeze(-1).detach().numpy())

            bs_pred.append(b_pred.cpu().detach().numpy())
            bs_true.append(b.cpu().unsqueeze(-1).detach().numpy())

            loss_per_step = np.concatenate([loss_per_step, loss.detach().cpu().item()], axis=None)
            
            # log metrics
            step_metrics = {}
            
            main_lr = self.config.optimizer.main_lr
            step_metrics["main_lr"] = main_lr

            wandb.log(step_metrics)

        ys_pred = np.vstack(ys_pred)
        ys_true = np.vstack(ys_true)
        ys_true = np.vstack(ys_true) # this is not a bug; needs to be stacked twice
        ys_prob = ys_pred.copy()
        ys_pred = np.argmax(ys_pred, axis=1) 
        ys_true_1h = to_one_hot(np.vstack(ys_true), num_classes=self.config.architecture.num_classes).squeeze(1).numpy()

        bs_pred = np.vstack(bs_pred)
        bs_true = np.vstack(bs_true)
        bs_true = np.vstack(bs_true) # this is not a bug; needs to be stacked twice
        bs_prob = bs_pred.copy()
        bs_pred = np.argmax(bs_pred, axis=1)
        bs_true_1h = to_one_hot(np.vstack(bs_true), num_classes=self.config.architecture.num_birads).squeeze(1).numpy()

        perf_metrics = compute_classification_metrics(ys_true, ys_prob, desc=desc, tune_threshold=self.config.metrics.tune_threshold)
        birads_metrics = compute_multiclass_metrics(bs_true, bs_prob, desc=desc, tune_threshold=self.config.metrics.tune_threshold)

        # compute and log metrics
        epoch_metrics = {
            "epoch": self.epoch,
            f"{desc}/loss": np.mean(loss_per_step),
            #f"{desc}/acc": accuracy_score(ys_true, ys_pred),
            #f"{desc}/bal_acc": balanced_accuracy_score(ys_true, ys_pred),
            #f"{desc}/auc": roc_auc_score(ys_true, ys_prob[:,1]),#, multi_class='ovr', average='weighted'),
            #f"{desc}/sens": recall_score(ys_true, ys_pred),
            #f"{desc}/spec": recall_score(ys_true, ys_pred, pos_label=0),
        }

        for (metric_name, metric_value) in perf_metrics.items():
            epoch_metrics[f"{desc}/{metric_name}"] = metric_value
        for (metric_name, metric_value) in birads_metrics.items():
            epoch_metrics[f"{desc}/birads_{metric_name}"] = metric_value

        wandb.log(epoch_metrics)

        fig, ax = show_confmat(ys_true, ys_pred)
        wandb.log({f"{desc}/confmat": wandb.Image(plt)})

        return epoch_metrics

    def save_experiment_state(self):
        if self.exp_state_path is None:
            return
        logging.info(f"Saving experiment snapshot to {self.exp_state_path}")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
                "gradient_scaler": self.gradient_scaler.state_dict(),
                "rng": get_all_rng_states(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            },
            self.exp_state_path,
        )

    def save_model_weights(self, score, is_best_score=False):
        if self.config.checkpoint_dir is None:
            return

        if not is_best_score:
            fname = f"model_epoch{self.epoch}.ckpt"
        else:
            fname = "best_model.ckpt"

        logging.info("Saving model to checkpoint directory")
        logging.info(f"Checkpoint directory: {self.config.checkpoint_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.checkpoint_dir, fname),
        )

    def teardown(self):
        # remove experiment state file
        if self.exp_state_path is not None:
            os.remove(self.exp_state_path)


