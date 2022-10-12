<h1 style="text-align:center;"> Dark patterns in e-commerce: a dataset and its baseline evaluations </h1> 

This repository provides the text-based dataset and experimental code for the paper: [Dark patterns in e-commerce: a dataset and its baseline evaluations]()

## Overview

Dark patterns are user interface designs on online services that are designed to make users behave in ways they do not intend.

In this work, we construct a text-based dataset for dark pattern automatic detection on e-commerce sites and  its  detection using machine learning as a baseline.

For baseline detection, we applied following two methods:

- Classical NLP Methods ( Bag-of-Words + Classical Machine Learning Model )
- Transformer-based pre-trained language models ( e.g. BERT )

For more information, please check our [Paper]().

## Project Structure

- `dataset/ `: [dataset.tsv]() in this directory is the dataset for text-based dark patterns automatic detection. 
- `darkpattern-auto-detection-classical/`: Experimental code of baseline evaluation using classical NLP methods.
- `darkpattern-auto-detection-deeplearning/ `: Experimental code of baseline evaluation using transformer-based pre-trained language models.
- `scraping/`: Code for collecting non-dark pattern texts in the dataset.

## Dataset

[dataset/dataset.tsv]() is a text-based dataset for dark pattern automatic detection (TSV Format). Dark pattern texts were obtained from Mathur et al.’s study in 2019, which consists of 1,818 dark pattern texts from shopping sites. Then, we collect non-dark pattern texts on e-commerce sites by accessing and segmenting the sites targeted by the Mathur et al.'s study. 

Scraping code for non-dark pattern texts is on [scraping/](). That is implemented using Typscript (Javascript) and Puppeteer.

## Baseline Evaluation

We conduct experiment of dark pattern auto detection using the dataset. The code is on [darkpattern-auto-detection-classical/]() (Classical NLP Methods) and [darkpattern-auto-detection-deeplearning/]() (Transformer-based pre-trained language models). 

#### Experimental Result of Classical NLP Methods

| Model | Accuracy | AUC | F1 score  | Precision | Recall |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| Logistic Regression | 0.961 | 0.989 | 0.960  | 0.981 | 0.940 |
| SVM | 0.954 | 0.987 | 0.952  | 0.986 | 0.922 |
| Random Forest | 0.958 | 0.989 | 0.957 | 0.984 | 0.932 |
| Gradient Boosting | 0.962 | 0.989 | 0.961 | 0.976 | 0.947 |

#### Experimental Result of Transformer-based pre-trained language models

| Model  | Accuracy | AUC | F1 score  | Precision | Recall |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| $\text{BERT}_{base}$ | 0.972 | 0.993 | 0.971 | 0.982 | 0.961  |  
| $\text{BERT}_{large}$ | 0.965 | 0.993 | 0.965 | 0.973 | 0.957  | 
| $\text{RoBERTa}_{base}$ | 0.966 | 0.993 | 0.966 | 0.979 | 0.954  |  
| $\text{RoBERTa}_{large}$ | $\mathbf{0.975}$ | $\mathbf{0.995}$ | $\mathbf{0.975}$ | $\mathbf{0.984}$ | $\mathbf{0.967}$  |  
| $\text{ALBERT}_{base}$ | 0.959 | 0.991 | 0.959 | 0.972 | 0.946  |  
| $\text{ALBERT}_{large}$ | 0.965 | 0.986 | 0.965 | 0.973 | 0.957  |  
| $\text{XLNet}_{base}$ | 0.966 | 0.992 | 0.966 | 0.975 | 0.958  |  
| $\text{XLNet}_{large}$ | 0.942 | 0.988 | 0.940 | 0.968 | 0.914  |  \hline


## Acknowledgements

This project is based on the Mathur et al.’s study and its [dataset](https://github.com/aruneshmathur/dark-patterns/blob/master/data/final-dark-patterns/dark-patterns.csv). We thank their authors for making the source code publically available.

## Changes from Arunesh's Dataset

- We extract text data from "Pattern String" column in [dark-patterns.csv](https://github.com/aruneshmathur/dark-patterns/blob/master/data/final-dark-patterns/dark-patterns.csv).
- We ignored missing and duplicate text data.

## Citation

## License