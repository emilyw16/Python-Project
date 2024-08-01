import pandas as pd
import random
import csv
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

input_file_path = # health data csv file
output_file_path = # health data csv file

def convert_age(age):
    if age < 30:
        return "0-29"
    elif 30 <= age < 60:
        return "30-59"
    else:
        return "60+"
    
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w', newline='') as output_file:
    reader = pd.read_csv(input_file, chunksize=1)
    writer = csv.writer(output_file)
    first_chunk = True
    for chunk in reader:  
        chunk['Test Result'] = chunk['Age'].apply(convert_age) + " " + chunk['Gender'] + " " + chunk['Medical Condition'] + " " + chunk['Admission Type']
        output_chunk = chunk[['Test Result', 'Length of Stay']]
        # Convert the DataFrame to a list of lists
        data_to_write = output_chunk.values.tolist()
        # Write to the CSV file
        if first_chunk:
            writer.writerow(output_chunk.columns)  # Write the header only for the first chunk
            first_chunk = False
        writer.writerows(data_to_write)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

data = pd.read_csv(# health data csv file)
X_train, X_test, y_train, y_test = train_test_split(data['Test Result'], data['Length of Stay'], test_size=0.2, random_state=42)

stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

X_train = X_train.apply(preprocess)
X_test = X_test.apply(preprocess)

sentences = [sentence.split() for sentence in X_train]
w2v_model = Word2Vec(sentences, window=5, min_count=5, workers=4)

import numpy as np

def vectorize(sentence):
    words = sentence.split()
    words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    if len(words_vecs) == 0:
        return np.zeros(100)
    words_vecs = np.array(words_vecs)
    return words_vecs.mean(axis=0)

X_train = np.array([vectorize(sentence) for sentence in X_train])
X_test = np.array([vectorize(sentence) for sentence in X_test])

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred, pos_label=1, average = 'weighted'))
print('Recall:', recall_score(y_test, y_pred, pos_label=1, average = 'weighted'))
print('F1 score:', f1_score(y_test, y_pred, pos_label=1, average = 'weighted'))

test_sentence = "young Female Hypertension Emergency"
test_sentence = vectorize(test_sentence)
result = clf.predict(np.array([test_sentence]))
print(result)