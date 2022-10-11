import os
import random

import numpy as np


def set_random_seed(random_seed: int = 42) -> None:
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    np.random.seed(random_seed)
