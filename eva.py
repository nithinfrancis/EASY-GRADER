import numpy as np 
import pandas as pd 
import nltk
nltk.download('punkt')
import re,collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import time

vectorizer = CountVectorizer(max_features = 15, ngram_range=(1, 3), stop_words='english')

def get_data():
    dataset = pd.read_csv("data\srain.csv")
    return dataset

def get_data2():
    dataset1 = pd.read_csv("data\srain2.csv")
    return dataset1

def BOW(essay):
    countvectors=(vectorizer.fit_transform(essay)).toarray()
    return countvectors

def POS(essay):
    cleaned_essay=re.sub(r'\W', ' ', essay)
    words=nltk.word_tokenize(cleaned_essay)
    word_count=len(words)
    sentences=nltk.sent_tokenize(essay)
    sentences_count=len(sentences)
    avg_len_sent=0
    for sent in sentences:
        avg_len_sent+=len(sent)
    avg=avg_len_sent/sentences_count
    return word_count,sentences_count,avg_len_sent


def run(text):
    dataframe=get_data()
    essay_set=dataframe[['essay']].copy()
    score=dataframe['domain1_score']
    essay_set=essay_set[:1000]
    score=score[:1000]
    countvectors=BOW(essay_set['essay'])
    essay_set['word_count'],essay_set['sent_count'],essay_set['avg_sent_count']=zip(*essay_set['essay'].apply(POS))
    x=np.concatenate((essay_set.iloc[:,1:].values,countvectors),axis=1)
    x_train,x_test,y_train,y_test=train_test_split(x,score,random_state=0, test_size=0.20)

################### Linear Regression ###############
    print(x)
    
    model=LinearRegression()
    # model.fit(x_train,y_train)
    model.fit(x,score)

##################  GRADIENT BOOSTING  ##############

    model2=ensemble.GradientBoostingRegressor()
    model2.fit(x_train,y_train)

###################### user data #####################

    dataframe2=get_data2() 
    essay_set2=dataframe2[['essay']].copy()
    if text == "" :
        text= " plain text" 
    essay_set2['essay']=text   
    # score=dataframe2['domain1_score']
    countvectors=BOW(essay_set2['essay'])
    essay_set2['word_count'],essay_set2['sent_count'],essay_set2['avg_sent_count']=zip(*essay_set2['essay'].apply(POS))
    x2=np.concatenate((essay_set2.iloc[:,1:].values,countvectors),axis=1)

    y_pred_L=model.predict(x2)    # predicting with Linear Regression
    
    y_pred_G=model2.predict(x2)   # predicting with GRADIENT BOOSTING
    

    avgscore = ( y_pred_L + y_pred_G )/2


    if avgscore < 5.0 :
        Grade = " F "
    elif avgscore < 6.0 :
        Grade = " E "
    elif avgscore < 7.0 :
        Grade = " D "
    elif avgscore < 8.0 :
        Grade = " C "
    elif avgscore < 9.0 :
        Grade = " B "
    else :
        Grade = " A "

    return Grade
