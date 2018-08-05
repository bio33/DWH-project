# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:47:48 2018

@author: deeks
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:59:56 2018

@author: deeksha behara
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import sys;
import pandas as pd;
import nltk;
from nltk.corpus import stopwords;
import string;
from string import punctuation;
from nltk.stem import PorterStemmer;
import nltk;
import re;
from collections import Counter;
import numpy as np;

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn import tree
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

" one time initialization:: Don't Run this
nltk.download()
df.info;
df.head();
df.language.unique()
list(df_result)


"Combining the required data sets = Applicable only to this dataset. (rumor propogation dataset)
" Read Data

df_movies_metadata = pd.read_csv("C:/Dalhousie/Term3/Datawarehousing/Project/the-movies-dataset/movies_metadata.csv",sep=",");
df_credits_csv = pd.read_csv("C:/Dalhousie/Term3/Datawarehousing/Project/the-movies-dataset/credits.csv",sep=",");
df_credits_csv["id"]=df_credits_csv["id"].astype(str)

df_movies_metadata['id'] = df_movies_metadata['id'].str.strip()
df_credits_csv['id'] = df_credits_csv['id'].str.strip()

df_result=pd.merge(df_movies_metadata, df_credits_csv, on="id")
df_result.head();
df_result.info

" Fill in empty values.
df_result = df_result.fillna('NA')

"Combine the tags and topics in the data

"Drop the unecessary columns

df_result.drop('belongs_to_collection', axis=1, inplace=True)
df_result.drop('imdb_id', axis=1, inplace=True)
df_result.drop('poster_path', axis=1, inplace=True)
df_result.drop('video', axis=1, inplace=True)
df_result.drop('vote_count', axis=1, inplace=True)
df_result.drop('id', axis=1, inplace=True)


df_result.revenue = pd.to_numeric(df_result.revenue, errors='coerce').fillna(0).astype(np.int64)
df_result.budget = pd.to_numeric(df_result.budget, errors='coerce').fillna(0).astype(np.int64)


"Find the difference between columns in Dataframe:
df_result.revenue = df_result.revenue.astype(float).fillna(0.0)
df_result.budget = df_result.budget.astype(float).fillna(0.0)
df_result['profit'] = df_result['revenue'] - df_result['budget']
df_result['labels'] = np.where(df_result['profit']>df_result['budget']/2, 'SUCCESS', 'FAILURE')


" Preprocessing:   
stop = stopwords.words('english')
    
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def removewhites(text):
    re.sub( '\s+', ' ', text ).strip()
    return text

def find_ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))

df_result['genres']=df_result.genres.apply(remove_punctuations)
df_result['genres']=df_result.genres.apply(removewhites)

df_result['production_companies']=df_result.production_companies.apply(remove_punctuations)
df_result['production_companies']=df_result.production_companies.apply(removewhites)

df_result['production_countries']=df_result.production_countries.apply(remove_punctuations)
df_result['production_countries']=df_result.production_countries.apply(removewhites)

df_result['spoken_languages']=df_result.spoken_languages.apply(remove_punctuations)
df_result['spoken_languages']=df_result.spoken_languages.apply(removewhites)

df_result['cast']=df_result.cast.apply(remove_punctuations)
df_result['cast']=df_result.cast.apply(removewhites)

" Removing the numbers from the string / columns in the Dataframes:
    
df_result['genres'] = df_result['genres'].str.replace('\d+', '')
df_result['production_countries'] = df_result['production_countries'].str.replace('\d+', '')
df_result['production_companies'] = df_result['production_companies'].str.replace('\d+', '')
df_result['spoken_languages'] = df_result['spoken_languages'].str.replace('\d+', '')
df_result['spoken_languages'] = df_result['spoken_languages'].str.replace('\d+', '')
df_result['cast'] = df_result['cast'].str.replace('\d+', '')


" Preparing the training data.
df= df_result[['adult' , 'budget' , 'genres','homepage','original_language',
               'original_title','overview','popularity','production_companies',
               'production_countries','release_date','revenue','runtime',
               'spoken_languages','status','tagline','title','vote_average',
               'cast','crew','profit']]
" Extracting labels.
features = df.columns[:21]
df = df.applymap(str)
df=df.apply(LabelEncoder().fit_transform)
X_train, X_test= train_test_split(df, test_size = 0.40, random_state = 1)

y = pd.factorize(df_result['labels'])[0]
Y_train,Y_test=train_test_split(y, test_size = 0.40, random_state = 1)


clf_rf = RandomForestClassifier(n_jobs=2, random_state=0)
clf_rf.fit(X_train[features], Y_train)

clf_lr = linear_model.LogisticRegression()
clf_lr.fit(X_train[features], Y_train)

clr_mnb = MultinomialNB()
clr_mnb.fit(X_train[features], Y_train)

clr_dt = tree.DecisionTreeClassifier()
clr_dt.fit(X_train[features], Y_train)

clf_svm = svm.SVC()
clf_svm.fit(X_train[features], Y_train)  

predicted_vlaues_rf=clf_rf.predict(X_test[features])
predicted_vlaues_lr=clf_lr.predict(X_test[features])
predicted_vlaues_mnb=clr_mnb.predict(X_test[features])
predicted_values_dt=clr_dt.predict(X_test[features])
predicted_values_svm=clf_svm.predict(X_test[features])

accuracy_rf=accuracy_score(predicted_vlaues_rf, Y_test)
accuracy_lr=accuracy_score(predicted_vlaues_lr, Y_test)
accuracy_mnb=accuracy_score(predicted_vlaues_mnb, Y_test)
accuracy_dt=accuracy_score(predicted_values_dt, Y_test)
accuracy_svm=accuracy_score(predicted_values_svm,Y_test)

print (" Confusion matrix_test Random Forest ", confusion_matrix(Y_test, predicted_vlaues_rf))
print (" Confusion matrix_test Linear Regression", confusion_matrix(Y_test, predicted_vlaues_lr))
print (" Confusion matrix_test Multinomial Naive Baye's", confusion_matrix(Y_test, predicted_vlaues_mnb))
print (" Confusion matrix_test Decision Tree", confusion_matrix(Y_test, predicted_values_dt))
print (" Confusion matrix_test Support Vector Machine", confusion_matrix(Y_test, predicted_values_svm))

x_name=["RandomForest","Logistic Regression","MNB","Decision Tree","SVM"]
x=np.arange(len(x_name))
y=[accuracy_rf,accuracy_lr,accuracy_mnb,accuracy_dt,accuracy_svm]
plt.xticks(x, x_name) 
plt.bar(x,y)
plt.show(1)

