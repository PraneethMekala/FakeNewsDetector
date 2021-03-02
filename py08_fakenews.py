"""
py05. Tensorflow on Detecting Fake News
Updated on 2//28/2021
by Praneeth
Reference: https://www.kaggle.com/sgoel26/py-reake-real-fake-news-classification-task/notebook

"""
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import sklearn
import scipy
import json
import sys
import csv
import os
import string
import io
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

os.getcwd()	
os.chdir('/Users/praneethmekala/Desktop/Data Science bootcamp/Week 7')

fake = pd.read_csv("W07a_fake.csv", names=None,encoding='latin-1',low_memory=False)
true = pd.read_csv("W07a_true.csv", names=None,encoding='latin-1',low_memory=False)

fake['label']=1
fake.head(2)
true['label']=0
true.head(3)
# pd.concat is like function rbind in R
df=pd.concat([fake,true]).reset_index(drop=True)
df.shape
df.label.value_counts()

# Preprocessing and Cleaning Data
nltk.download('stopwords')
stopw = set(stopwords.words('english'))
def clean(text):
    #1. Remove punctuation
    translator1 = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(translator1)
    #2. Convert to lowercase characters
    text = text.lower()
    #3. Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopw])
    return text

df['clean_title']=df['title'].apply(clean)
df['clean_text']=df['text'].apply(clean)
df['clean_subject']=df['subject'].str.lower()
df['combined']=df['clean_subject']+' '+df['clean_title']+' '+df['clean_text']

from nltk.stem.porter import PorterStemmer
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

stemmer = PorterStemmer()

def stem_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text)

df['after_stemming'] = df.combined.apply(stem_text)

#visualize word distribution
df['doc_len'] = ''
df['doc_len'] = df['after_stemming'].apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(df['doc_len'].mean() + df['doc_len'].std()).astype(int)
print(max_seq_len)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['after_stemming'].to_list(), df['label'].values, test_size=0.33, random_state=42)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

# https://www.freecodecamp.org/news/install-tensorflow-and-keras-using-anaconda-navigator-without-command-line/
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer

MAX_NB_WORDS = 28000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))

from tensorflow.keras.preprocessing import sequence
word_seq_train = tokenizer.texts_to_sequences(X_train)
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_train.shape

gensim_news_desc = []

chunk_data = X_train
for record in range(0,len(chunk_data)):
    news_desc_list = []
    for tok in chunk_data[record].split():
        news_desc_list.append(str(tok))
    gensim_news_desc.append(news_desc_list)
len(gensim_news_desc)

# You need to go to Anaconda to install gensim
import gensim
from gensim.models import Word2Vec
gensim_model = Word2Vec(gensim_news_desc, min_count=5, size = 200, sg=1)
# summarize the loaded model
print(gensim_model)
# summarize vocabulary
words = list(gensim_model.wv.vocab)
len(words)

#training params
batch_size = 1024
num_epochs = 10
#model parameters
num_filters = 128
embed_dim = 200 
weight_decay = 1e-4
class_weight = {0: 1, 1: 1}

print('preparing embedding matrix...')

gensim_words_not_found = []
gensim_nb_words = len(gensim_model.wv.vocab)
print("gensim_nb_words : ",gensim_nb_words)

gensim_embedding_matrix = np.zeros((gensim_nb_words, embed_dim))

for word, i in word_index.items():
    #print(word)
    if i >= gensim_nb_words:
        continue
    if word in gensim_model.wv.vocab :
        embedding_vector = gensim_model[word]
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            gensim_embedding_matrix[i] = embedding_vector
    else :
        gensim_words_not_found.append(word)
        
gensim_embedding_matrix.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

model = Sequential()
# gensim word2vec embedding
model.add(Embedding(gensim_nb_words, embed_dim, weights=[gensim_embedding_matrix], input_length=max_seq_len))
model.add(Conv1D(num_filters, 5, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dropout(0.6))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid')) 

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

#define callbacks
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

# gensim model training
hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, 
                 validation_split=0.1, shuffle=True, verbose=2,class_weight=class_weight)

import matplotlib.pyplot as plt 

# list all data in history
print(hist.history.keys())

# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

word_seq_test = tokenizer.texts_to_sequences(X_test)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

predictions = model.predict(word_seq_test)
pred_labels = predictions.round()

unique, counts = np.unique(y_test, return_counts=True)
unique, counts

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred_labels, labels=[1,0])
cm













