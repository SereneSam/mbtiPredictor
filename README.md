# MBTIPredictor
### Description: 
Machine learning prediction of the Myers-Brigg personality test.

How to install?
1) Using 'git clone' feature, clone the project to your local system (like VSCode or Google Colab). 

How to Run: 
1) Download the datasets : 
insert data sets here

2) Following the instructions in the Jupyter Notebook, link the datasets from your local machine to the code and run!


Our group used the following packages:
1. For Data Cleaning and Analyzing
    pandas as pd
    re
    nltk: to install stopwords and lemmatizing functionality
    nltk.corpus import stopwords
    nltk.stem import WordNetLemmatizer
    numpy as np
2. For feature extraction
    sklearn.pipeline import Pipeline
    sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
3. For Machine Learning Algorithms
    sklearn.model_selection import train_test_split, GridSearchCV
    sklearn.naive_bayes import MultinomialNB
    sklearn import metrics
    sklearn.linear_model import LogisticRegression
    sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
    torch
    math
    torchtext.data.utils import get_tokenizer
    torchtext.vocab import build_vocab_from_iterator
    torch.utils.data import DataLoader
    torch import nn
    time
    from torch.utils.data.dataset import random_split
    from torchtext.data.functional import to_map_style_dataset
4. For Data Visualization
    wordcloud
    matplotlib.pyplot as plt
    seaborn as sns
