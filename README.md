# MBTIPredictor
### Description: 
Machine learning prediction of the Myers-Brigg personality test.

How to install?
1) Using 'git clone' feature, clone the project to your local system (like VSCode or Google Colab). 

How to Run: 
1) Download the datasets :
   
MBTI_cleaned (used for 3730TorchClassifier.ipynb and smaller_ds_ml.ipynb): https://lsu.box.com/s/7rt3ita4xmbx0l6ywoc0q022a32k2s86

MBTI500 ( used in Data500_prediction.ipynb): https://lsu.box.com/s/4d27a5bh8s9jsuz5y5hvjeled9l6lise

2) Following the instructions in the Jupyter Notebook, link the datasets from your local machine to the code and run!

File Breakdown (.docx):
wordcloud - creates a word cloud for all the words

wordcloud_removed - creates a word cloud with certain common words removed due to redundancy(removed words: think, like, one, people, know)

topTen_bar - top ten words in every MBTI type

Folder Breakdown:
datasets_types - cleaned mbti_clean.csv split into the corresponding MBTI types

datasets_letters - cleaned mbti_clean.csv split into the corresponding MBTI dimensions (E, I, N, S, F, T, P, J)

big_datasets_types: cleaned MBTI500.csv split into the corresponding MBTI types

big_datasets_letters: MBTI500.csv split into the corresponding MBTI dimensions (E, I, N, S, F, T, P, J)




Our group used the following packages:
1. For Data Cleaning and Analyzing
    <p>pandas as pd</p>
    <p>re</p>
    <p>nltk: to install stopwords and lemmatizing functionality</p>
    <p>nltk.corpus import stopwords</p>
    <p>nltk.stem import WordNetLemmatizer</p>
    <p>numpy as np</p>
3. For feature extraction
    <p>sklearn.pipeline import Pipeline</p>
    <p>sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer</p>
4. For Machine Learning Algorithms</p>
    <p>sklearn.model_selection import train_test_split, GridSearchCV</p>
    <p>sklearn.naive_bayes import MultinomialNB</p>
    <p>sklearn import metrics</p>
    <p>sklearn.linear_model import LogisticRegression</p>
    <p>sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error</p>
    <p>torch</p>
    <p>math</p>
    <p>torchtext.data.utils import get_tokenizer</p>
    <p>torchtext.vocab import build_vocab_from_iterator</p>
    <p>torch.utils.data import DataLoader</p>
    <p>torch import nn</p>
    <p>time</p>
    <p>from torch.utils.data.dataset import random_split</p>
    <p>from torchtext.data.functional import to_map_style_dataset</p>
5. For Data Visualization
    <p>wordcloud</p>
    <p>matplotlib.pyplot as plt</p>
    <p>seaborn as sns</p>
