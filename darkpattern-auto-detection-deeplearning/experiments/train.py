from os.path import join
from typing import List

import hydra
import numpy as np
import pandas as pd
import torch
from const.path import CONFIG_PATH, DATASET_TSV_PATH, NN_MODEL_PICKLES_PATH
from omegaconf import DictConfig
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Subset
from trainer.trainer import Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.dataset import DarkpatternDataset
from utils.random_seed import set_random_seed
from utils.text import text_to_tensor as _text_to_tensor


def cross_validation(
    n_fold: int,
    pretrained: str,
    batch_size: int,
    lr: float,
    start_factor: float,
    max_length: int,
    dropout: float,
    epochs: int,
    save_model: bool,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_labels: int = 2,
) -> None:
    """
    Load & Define dataset.
    """
    df = pd.read_csv(DATASET_TSV_PATH, sep="\t", encoding="utf-8")
    texts = df.text.tolist()
    labels = df.label.tolist()

    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def text_to_tensor(text: str) -> torch.Tensor:
        return _text_to_tensor(text, tokenizer, max_length)

    ds = DarkpatternDataset(texts, labels, text_to_tensor)

    """
    Execute N (= n_fold) fold cross validation.
    """
    skf = StratifiedKFold(n_splits=n_fold)

    accuracy_scores: List[float] = []
    f1_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    roc_auc_scores: List[float] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):

        """
        Define train & test dataset.
        """

        train_ds = Subset(ds, train_idx)
        test_ds = Subset(ds, test_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

        """
        Initialize model, optimizer, loss function, lr_scheduler. 
        """
        net = AutoModelForSequenceClassification.from_pretrained(
            pretrained, num_labels=num_labels
        ).to(device)
        critation = nn.CrossEntropyLoss()
        optimizer = AdamW(net.parameters(), lr=lr)
        lr_scheduler = LinearLR(
            optimizer, start_factor=start_factor, total_iters=epochs
        )

        """
        Train.
        """
        trainer = Trainer(net, optimizer, critation, lr_scheduler, device)
        for epoch in range(epochs):
            try:
                trainer.train(train_loader)
            except Exception as e:
                print(e)

        """
        Evaluation.
        """
        outputs, tgt, pred = trainer.test(test_loader)
        accuracy_score = metrics.accuracy_score(tgt.numpy(), pred.numpy())
        f1_score = metrics.f1_score(tgt.numpy(), pred.numpy())
        precision_score = metrics.precision_score(tgt.numpy(), pred.numpy())
        recall_score = metrics.recall_score(tgt.numpy(), pred.numpy())

        prob = F.softmax(outputs, dim=1)[:, 1]  # outputs: [batch_size, num_labels]
        roc_auc = metrics.roc_auc_score(tgt.numpy(), prob.numpy())

        accuracy_scores.append(accuracy_score)
        f1_scores.append(f1_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        roc_auc_scores.append(roc_auc)

        """
        Save model.
        """
        if save_model:
            model_path = join(NN_MODEL_PICKLES_PATH, f"{pretrained}_{fold}.pth")
            torch.save(net.state_dict(), model_path)

    """
    Display evaluation result on console.
    """
    roc_auc_score_average = np.mean(roc_auc_scores)
    f1_score_average = np.mean(f1_scores)
    accuracy_score_average = np.mean(accuracy_scores)
    precision_score_average = np.mean(precision_scores)
    recall_score_average = np.mean(recall_scores)
    roc_auc_score_average = np.mean(roc_auc_scores)
    print(
        {
            "accuracy_scores": accuracy_scores,
            "f1_scores": f1_scores,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
            "roc_auc_scores": roc_auc_scores,
            "f1_score_average": f1_score_average,
            "accuracy_score_average": accuracy_score_average,
            "precision_score_average": precision_score_average,
            "recall_score_average": recall_score_average,
            "roc_auc_score_average": roc_auc_score_average,
        }
    )

    parameters_and_evaluation_text = f"""
    ```parameters:
        pretrained: {pretrained}
        batch_size: {batch_size}
        lr: {lr}
        max_length: {max_length}
        dropout: {dropout}
        epochs: {epochs}
        device: {device}
        num_labels: {num_labels}
    metrics for test:
         f1_score_average:{f1_score_average}
         accuracy_score_average:{accuracy_score_average}
         precision_score_average:{precision_score_average}
         recall_score_average:{recall_score_average}
         roc_auc_score_average:{roc_auc_score_average}
    ```
    """
    print(parameters_and_evaluation_text)


@hydra.main(config_path=CONFIG_PATH, config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    n_fold = cfg.train.n_fold
    pretrained = cfg.model.pretrained
    batch_size = cfg.train.batch_size
    lr = cfg.train.lr
    max_length = cfg.preprocess.max_length
    dropout = cfg.model.dropout
    epochs = cfg.train.epochs
    start_factor = cfg.train.start_factor
    save_model = cfg.train.save_model

    set_random_seed(cfg.random.seed)

    cross_validation(
        n_fold=n_fold,
        pretrained=pretrained,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        dropout=dropout,
        epochs=epochs,
        start_factor=start_factor,
        save_model=save_model,
    )


if __name__ == "__main__":
    main()
