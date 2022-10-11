from os.path import abspath, dirname, join

PROJECT_ROOT = join(dirname(abspath(__file__)), "../")

DATASET_PATH = join(PROJECT_ROOT, "dataset")

DATASET_TSV_PATH = join(DATASET_PATH, "dataset.tsv")

PICKLES_PATH = join(PROJECT_ROOT, "pickles")

NN_MODEL_PICKLES_PATH = join(PICKLES_PATH, "models/nn")

FIGURE_PATH = join(PROJECT_ROOT, "figures")

CONFIG_PATH = join(PROJECT_ROOT, "configs")
