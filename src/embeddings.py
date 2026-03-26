import numpy as np
import pandas as pd
from Config import *
import random
seed = 0
random.seed(seed)
np.random.seed(seed)

def get_tfidf_embd(df: pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfconverter = TfidfVectorizer(max_features=Config.VECTORIZER_MAX_FEATURES, 
                                      min_df=Config.VECTORIZER_MIN_DF, 
                                      max_df=Config.VECTORIZER_MAX_DF)
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    X = tfidfconverter.fit_transform(data).toarray()
    return X

def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)
