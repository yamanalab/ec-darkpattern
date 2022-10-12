# Dark Patterns Automatic Detection using classical NLP methods

Experimental code of text-based dark patterns automatic detection using classical NLP methods

## Requirements

- [Python](https://www.python.org/downloads/) 3.9.0
- [Poetry](https://python-poetry.org/docs/) 1.2.1

## Setup

### Installation

Install dependencies by running:

```
$ poetry install
```

### Dataset preparation

Create Symlink to [dataset/]() directory by :

```
$ ln -s {path to dataset/} dataset
```

※ Please place `dataset.tsv` directly under the `dataset/` directory.(`dataset/dataset.tsv`)

### Set PYTHONPATH 

```
$ export PYTHONPATH="$(pwd)"
```

## Train & Evaluate model

You can run experiment using GPU or CPU by running:

```
$ sh scripts/train.sh
```