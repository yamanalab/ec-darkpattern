import pickle
from os.path import exists
from typing import List, Tuple

import pandas as pd
from const.path import LABEL_DATA_PICKLES_PATH, PREPROCESSED_TEXT_DATA_PICKLES_PATH

from utils import text


def get_dataset(
    path_to_tsv: str, reload_data: bool = False
) -> Tuple[List[str], List[int]]:
    """
    Get dataset from tsv file.
    """
    pickles_exist = exists(PREPROCESSED_TEXT_DATA_PICKLES_PATH) and exists(
        LABEL_DATA_PICKLES_PATH
    )

    if pickles_exist and not reload_data:
        texts, labels = [], []
        with open(PREPROCESSED_TEXT_DATA_PICKLES_PATH, "rb") as f:
            texts = pickle.load(f)
        with open(LABEL_DATA_PICKLES_PATH, "rb") as f:
            labels = pickle.load(f)
        return texts, labels

    df = pd.read_csv(path_to_tsv, sep="\t", encoding="utf-8")
    texts = df.text.tolist()

    preprocessed_texts = [text.preprocess(t) for t in texts]
    labels = df.label.tolist()

    with open(PREPROCESSED_TEXT_DATA_PICKLES_PATH, "wb") as f:
        pickle.dump(preprocessed_texts, f)
    with open(LABEL_DATA_PICKLES_PATH, "wb") as f:
        pickle.dump(labels, f)

    return preprocessed_texts, labels
