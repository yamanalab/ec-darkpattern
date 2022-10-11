from os.path import abspath, dirname, join

PROJECT_ROOT = join(dirname(abspath(__file__)), "../")

DATASET_PATH = join(PROJECT_ROOT, "dataset/")

DATASET_TSV_PATH = join(DATASET_PATH, "dataset.tsv")

PICKLES_PATH = join(PROJECT_ROOT, "pickles")


FIGURE_PATH = join(PROJECT_ROOT, "figures")

CONFIG_PATH = join(PROJECT_ROOT, "configs")


DATA_PICKLES_PATH = join(PICKLES_PATH, "data")

PREPROCESSED_TEXT_DATA_PICKLES_PATH = join(DATA_PICKLES_PATH, "preprocessed_texts.pkl")
LABEL_DATA_PICKLES_PATH = join(DATA_PICKLES_PATH, "labels.pkl")
