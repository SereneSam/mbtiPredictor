# Machine learning Prediction with Myers-Brigg Type Indicator
![MBTI](https://thechargerfrontline.com/wp-content/uploads/2022/11/Personality-Test-900x568.png)
### Description: 


I. How to install?

>Using 'git clone' feature, clone the project to your IDE for coding (like VSCode or Google Colab). 
Make sure to either pip install or conda install libraries like scipy, wordcloud, nltk, seaborn and scikit-learn to run the code.

II. How to Run?
1) Download the datasets from the folder linked here: [https://lsu.box.com/s/n58ia30eouwszswydrxkn6zejdd7co6y](https://lsu.box.com/s/n58ia30eouwszswydrxkn6zejdd7co6y)
   
>mbti_cleaned.csv (used for 3730TorchClassifierBinary.ipynb and smaller_ds_ml.ipynb): [mbti_cleaned.csv](https://lsu.box.com/s/7rt3ita4xmbx0l6ywoc0q022a32k2s86)

>MBTI500.csv ( used in Data500_prediction.ipynb): [MBTI500.csv](https://lsu.box.com/s/4d27a5bh8s9jsuz5y5hvjeled9l6lise)

2) In the Jupyter Notebook, link the datasets from your local machine. Make sure to check whether the dataset that is not attached to the code folder is mentioned with the correct path from your local computer. The additional datasets, which are cleaned are given in the folder breakdowns part below.

3) The last to run is the prediction where you will run the function preprocessed_text, then put the sentence you want to run into the variable trial_sentence and run all the cells below which will give the letters that predict the output.
-----------------------------------------------------------------------------------------------------------------------------------------
>File Breakdown (images.docx):
>>wordcloud - creates a word cloud for all the words

>>wordcloud_removed - creates a word cloud with certain common words removed due to redundancy(removed words: think, like, one, people, know)

>>topTen_bar - top ten words in every MBTI type

>Folder Breakdown:
>>datasets_types - cleaned mbti_clean.csv split into the corresponding MBTI types

>>datasets_letters - cleaned mbti_clean.csv split into the corresponding MBTI dimensions (E, I, N, S, F, T, P, J)

>>big_datasets_types: cleaned MBTI500.csv split into the corresponding MBTI types

>>big_datasets_letters: MBTI500.csv split into the corresponding MBTI dimensions (E, I, N, S, F, T, P, J)

-----------------------------------------------------------------------------------------------------------------------------------------

Packages Used:
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
