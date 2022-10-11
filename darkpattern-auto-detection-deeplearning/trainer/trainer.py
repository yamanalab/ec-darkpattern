from typing import Any, List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer,
        critetion: nn.Module,
        lr_scheduler: Any,
        device: torch.device,
    ) -> None:
        self.optimizer = optimizer
        self.critetion = critetion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.net = net.to(self.device)

    def loss_fn(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.critetion(output, target)

    def train_step(
        self, input_train: torch.Tensor, target: torch.Tensor  # TODO:inputを文字列に変更
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [loss, output]
        self.net.train()
        output = self.net(input_train).logits  # [batch_size,label_size]
        loss = self.loss_fn(output, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, output

    def val_step(
        self, input_val: torch.Tensor, target: torch.Tensor  # TODO:inputを文字列に変更
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # [loss, output]
        self.net.eval()
        loss, output = None, None
        with torch.no_grad():
            output = self.net(input_val).logits  # [batch_size,label_size]
            loss = self.loss_fn(output, target)
        return loss, output

    def train(self, train_loader: DataLoader) -> List[float]:
        train_losses: List[float] = []
        for i, (input_train, target) in enumerate(train_loader):
            input_train = input_train.to(self.device)
            target = target.to(self.device)

            loss, output = self.train_step(input_train, target)

            del input_train
            del target
            torch.cuda.empty_cache()

            print(f"Train step: {i + 1}/{len(train_loader)} loss: {loss.item()}")

            train_losses.append(loss.item())

        self.lr_scheduler.step()
        return train_losses

    def validate(self, val_loader: DataLoader) -> List[float]:
        val_losses: List[float] = []

        for i, (input_val, target) in enumerate(val_loader):
            input_val = input_val.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                loss, output = self.val_step(input_val, target)

            del input_val
            del target
            torch.cuda.empty_cache()

            print(f"Val step: {i + 1}/{len(val_loader)} loss: {loss.item()}")

            val_losses.append(loss.item())

        return val_losses

    def test(
        self, test_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        test_preds: torch.Tensor = torch.Tensor([])
        test_tgts: torch.Tensor = torch.Tensor([])
        outputs: torch.Tensor = torch.Tensor([])

        for i, (input_test, target) in enumerate(test_loader):
            input_test = input_test.to(self.device)
            target = target.to(self.device)
            _, output = self.val_step(input_test, target)

            output = output.to("cpu")
            target = target.to("cpu")

            pred = output.argmax(dim=-1)

            outputs = torch.cat((outputs, output), dim=0)
            test_preds = torch.cat((test_preds, pred), dim=0)
            test_tgts = torch.cat((test_tgts, target), dim=0)

        return outputs, test_tgts, test_preds

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)
