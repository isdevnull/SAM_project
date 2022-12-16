__all__ = ["CIFARTrainer", "SAMTrainer", "FGSMTrainer", "Inferencer"]


import collections
import os

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models
import train_utils
from sam import SAM, FGSM


class BaseTrainer:
    def __init__(self):
        raise NotImplementedError()

    def setup_data(self):
        raise NotImplementedError()

    def setup_loaders(self):
        raise NotImplementedError()

    def setup_model(self):
        raise NotImplementedError()

    def setup_optim(self):
        raise NotImplementedError()

    def setup_scheduler(self):
        raise NotImplementedError()

    def setup_loss(self):
        raise NotImplementedError()

    def train_epoch(self):
        raise NotImplementedError()

    def test_epoch(self):
        raise NotImplementedError()

    def train_loop(self):
        raise NotImplementedError()


class CIFARTrainer(BaseTrainer):
    def __init__(self, config):
        self.config = config

    @property
    def device(self):
        return self.config.general.device

    def setup_data(self):
        if self.config.data.dataset == "CIFAR10":
            norm_mean = (0.4914, 0.4822, 0.4465)
            norm_std = (0.2470, 0.2435, 0.2616)
        elif self.config.data.dataset == "CIFAR100":
            norm_mean = (0.5071, 0.4866, 0.4409)
            norm_std = (0.2673, 0.2564, 0.2762)
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(
                    32, padding=4, padding_mode="reflect"
                ),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(norm_mean, norm_std),
            ]
        )
        test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(norm_mean, norm_std),
            ]
        )
        self.train_dataset = getattr(torchvision.datasets, self.config.data.dataset)(
            transform=train_transform, **self.config.data.train
        )
        self.test_dataset = getattr(torchvision.datasets, self.config.data.dataset)(
            transform=test_transform, **self.config.data.test
        )

    def setup_loaders(self):
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, **self.config.dataloader.train
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, **self.config.dataloader.test
        )

    def setup_model(self):
        self.model = getattr(models, self.config.training.model)(
            **self.config.training.model_params
        )
        self.model.to(self.device)

    def setup_optim(self):
        self.optim = getattr(torch.optim, self.config.training.optimizer)(
            self.model.parameters(), **self.config.training.optim_params
        )
        self.scheduler = None
        if (
            hasattr(self.config.training, "scheduler")
            and self.config.training.scheduler is not None
        ):
            self.setup_scheduler()

    def setup_scheduler(self):
        self.scheduler = getattr(
            torch.optim.lr_scheduler, self.config.training.scheduler
        )(
            self.optim,
            **self.config.training.scheduler_params,
        )

    def setup_loss(self):
        self.loss_function = getattr(torch.nn, self.config.training.loss_criterion)(
            **self.config.training.loss_params
        )

    def train_step(self, img, label):
        self.optim.zero_grad()

        logits = self.model(img)
        loss = self.loss_function(logits, label)
        loss.backward()
        self.optim.step()

        return loss, logits, train_utils.get_grad_norm(self.model)

    def train_epoch(self):
        train_epoch_info = collections.defaultdict(float)
        num_steps = 0

        self.model.train()

        for img, label in tqdm(self.train_loader, desc="Train Epoch Progress"):
            img = img.to(self.device)
            label = label.to(self.device)

            loss, logits, grad_norm = self.train_step(img, label)

            train_epoch_info["grad_norm"] = grad_norm
            train_epoch_info["train_loss"] += loss.detach().item()
            train_epoch_info["train_acc_1"] += train_utils.top_n_accuracy(
                logits.detach().cpu(), label.cpu(), n=1
            )
            num_steps += 1

        for k, v in train_epoch_info.items():
            train_epoch_info[k] = v / num_steps

        return train_epoch_info

    def test_epoch(self):
        test_epoch_info = collections.defaultdict(float)
        num_steps = 0

        self.model.eval()

        for img, label in tqdm(self.test_loader, desc="Test Epoch Progress"):
            img = img.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                logits = self.model(img)
            loss = self.loss_function(logits, label)

            test_epoch_info["test_loss"] += loss.detach().item()
            test_epoch_info["test_acc_1"] += train_utils.top_n_accuracy(
                logits.cpu(), label.cpu(), n=1
            )
            test_epoch_info["test_acc_5"] += train_utils.top_n_accuracy(
                logits.cpu(), label.cpu(), n=5
            )
            num_steps += 1

        for k, v in test_epoch_info.items():
            test_epoch_info[k] = v / num_steps

        return test_epoch_info

    def train_loop(self):
        tb_writer = SummaryWriter(log_dir=self.config.logging.log_dir)

        epoch_pbar = tqdm(total=self.config.general.n_epochs, initial=1)
        epoch_pbar.set_description("Current Epoch")

        for epoch in range(self.config.general.n_epochs):
            self.setup_loaders()

            info = self.train_epoch()
            if (epoch + 1) % self.config.logging.log_full_every == 0:
                self.test_loader = torch.utils.data.DataLoader(
                    self.test_dataset, batch_size=1, shuffle=False, num_workers=1
                )
            info.update(self.test_epoch())
            if self.scheduler is not None:
                self.scheduler.step()

            weight_norm = train_utils.get_weight_norm(self.model)
            info["weight_norm"] = weight_norm

            if (epoch + 1) % self.config.logging.save_ckpt_every == 0:
                self.model.to("cpu")

                ckpt_dir = os.path.join(self.config.logging.log_dir, "checkpoints")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)

                torch.save(
                    self.model.state_dict(),
                    os.path.join(ckpt_dir, f"resnet_cl_ckpt_{epoch + 1}.pth"),
                )
                self.model.to(self.device)

            for k, v in info.items():
                epoch_pbar.write(s=f"Epoch : {epoch + 1}, {k} -- {v:.4f}")
                tb_writer.add_scalar(tag=k, scalar_value=v, global_step=epoch)

            epoch_pbar.update(1)

        epoch_pbar.close()


class SAMTrainer(CIFARTrainer):
    def setup_optim(self):
        base_optim = getattr(torch.optim, self.config.training.optimizer)
        self.optim = SAM(
            self.model.parameters(), base_optim, **self.config.training.optim_params
        )
        self.scheduler = None
        if (
            hasattr(self.config.training, "scheduler")
            and self.config.training.scheduler is not None
        ):
            self.setup_scheduler()

    def train_step(self, img, label):
        train_utils.enable_running_stats(self.model)
        logits = self.model(img)
        loss = self.loss_function(logits, label)
        loss.backward()
        grad_norm = train_utils.get_grad_norm(self.model)
        self.optim.first_step(zero_grad=True)

        train_utils.disable_running_stats(self.model)
        self.loss_function(self.model(img), label).backward()
        self.optim.second_step(zero_grad=True)

        return loss, logits, grad_norm


class FGSMTrainer(CIFARTrainer):
    def setup_optim(self):
        base_optim = getattr(torch.optim, self.config.training.optimizer)
        self.optim = FGSM(
            self.model.parameters(), base_optim, **self.config.training.optim_params
        )
        self.scheduler = None
        if (
            hasattr(self.config.training, "scheduler")
            and self.config.training.scheduler is not None
        ):
            self.setup_scheduler()

    @property
    def fgsm_steps(self):
        return self.config.training.fgsm_steps

    def train_step(self, img, label):
        train_utils.enable_running_stats(self.model)
        return_loss, return_logits, grad_norm = None, None, None
        for i in range(self.fgsm_steps):
            if i == 0:
                save_state = True
            else:
                save_state = False
            logits = self.model(img)
            loss = self.loss_function(logits, label)
            loss.backward()
            if i == 0:
                grad_norm = train_utils.get_grad_norm(self.model)
                return_loss = loss.clone()
                return_logits = logits.clone()
            self.optim.fgsm_step(zero_grad=True, save_state=save_state)
            if i == 0:
                train_utils.disable_running_stats(self.model)

        self.loss_function(self.model(img), label).backward()
        self.optim.second_step(zero_grad=True)

        return return_loss, return_logits, grad_norm


class Inferencer(CIFARTrainer):
    def setup_model(self):
        assert hasattr(self.config, "inference") and hasattr(self.config.inference, "ckpt")
        if not os.path.exists(self.config.inference.ckpt):
            raise FileNotFoundError(f"No checkpoint named {self.config.inference.ckpt}")
        model_state_dict = torch.load(self.config.inference.ckpt, map_location=self.device)
        self.model = getattr(models, self.config.training.model)(
            **self.config.training.model_params
        )
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)

    def setup_optim(self):
        pass

    def train_loop(self):
        test_epoch_info = super().test_epoch()
        print("Inference results:")
        for k, v in test_epoch_info.items():
            print(f"{k} -- {v:.4f}")
