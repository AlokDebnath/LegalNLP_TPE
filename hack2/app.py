from nltk.corpus import stopwords
import os
from werkzeug import secure_filename
import sys
import json
import pickle
from tqdm import tqdm
import operator
import numpy as np
import pandas as pd
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import time
import urllib.request
from flask import Flask, render_template, redirect, jsonify, request
from flask_cors import CORS
import subprocess
import sys
import re
from wit import Wit


app = Flask(__name__)
app.config['SECRET_KEY'] = 'e5ac358c-f0bf-11e5-9e39-d3b532c10a28'



# Intializing the Word2Vec Model, download the file from https://nlp.stanford.edu/projects/glove/
# Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip 
# Unzip the file then run: python3 -m gensim.scripts.glove2word2vec --input  glove.6B.300d.txt --output glove.6B.300d.w2vformat.txt
model = gensim.models.KeyedVectors.load_word2vec_format('data/glove.6B.300d.w2vformat.txt')
word_freq_map = {}
# with open("vocab", 'r') as vocab_file:
with open("data/vocab", 'r') as vocab_file:
    lines = vocab_file.readlines()
    for line in lines:
        word_freq_map[line.split()[0]] = int(line.split()[1])
    
stop_list = sorted(word_freq_map.items(), key=operator.itemgetter(1), reverse=True)[:150]
cache = {}


#### Read the full training data and split it into smaller chunks

def load_data(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text


#### Gets all the sentences of the article along with its metadata


def get_sentences_with_metadata(text): # Split sentences and metadata list. 1:1 correspondence
    list_of_sentences = text.split('.')
    sentence_metadata = []
    for i in range(len(list_of_sentences)):
        sentence_metadata.append("")
    return list_of_sentences, sentence_metadata


#### The Following 2 functions are used for Preprocessing of a given text


def is_ascii(word):
    """
    Checks if word is ascii or not
    :param word: token
    :return: Boolean
    """
    valid = True
    try:
        word = word.encode('ascii')
    except UnicodeEncodeError:
        valid = False
    return valid



def get_processed_tokens(sentence):
    punc_map = {}
    punc_map = punc_map.fromkeys('!"\'()*+,;<>[]^`{|}~:=%&_#?-$/', ' ')
    table = str.maketrans(punc_map)
    tokens = sentence.lower().translate(table).split()
    stop_words = set(stopwords.words('english')) 
    stop_words = list(stop_words) + stop_list
    cleaned_tokens = [word for word in tokens if word not in stop_words and is_ascii(word) and '@' not in word and '\\' not in word and len(word) > 1]            
    return cleaned_tokens


#### Gets the processed sentences for each sentence of the article



def make_processed_sentences(list_of_sentences):
    processed_sentences = []
    for sentence in list_of_sentences:
        if isinstance(sentence, list):
            sentence = " ".join(sentence)
        processed_sentences.append(get_processed_tokens(sentence))
    return processed_sentences


#### Gives the number of words common between given 2 sentences

def get_no_of_common_word(sentence1, sentence2):
    common_count = 0
    for s1 in sentence1:
        for s2 in sentence2:
            if s1 == s2:
                common_count += 1
    return common_count



def get_word_vec_sim(sentence1, sentence2):
    score = 0
    for word1 in sentence1:
        for word2 in sentence2:
            try:
                temp = cache[word1+word2]
            except:
                try:
                    temp = model.similarity(word1, word2)
                    cache[word1+word2] = temp
                    cache[word2+word1] = temp
                except:
                    cache[word1+word2] = 0
                    cache[word2+word1] = 0
                    temp = 0
            score += temp
    return score


#### Generic scoring function which gives a score between 2 sentences



def scoring(sentence1, sentence2, metadata):
    len_normalize = len(sentence1) + len(sentence2) + 1 # Normalizing by length of vector
    common_words = get_no_of_common_word(sentence1, sentence2)
    word_vec_score = get_word_vec_sim(sentence1, sentence2)
    score = common_words / 2*len_normalize + word_vec_score / len_normalize
    return score


#### Makes the graph which has relations between every pair of sentences

def make_graph(processed_sentences, metadata):
    sentence_graph = np.zeros(shape=(len(processed_sentences), len(processed_sentences)))
    sentence_common_graph = np.zeros(shape=(len(processed_sentences), len(processed_sentences)))
    
    for i in range(len(processed_sentences)):
        for j in range(len(processed_sentences)):
            sentence1 = processed_sentences[i]
            sentence2 = processed_sentences[j]
            if i == j:
                sentence_graph[i][j] = 0
                sentence_common_graph[i][j] = 0
            else:
                sentence_graph[i][j] = scoring(sentence1, sentence2, metadata)
                sentence_common_graph[i][j] = get_no_of_common_word(sentence1, sentence2)
    return sentence_graph, sentence_common_graph


#### Following functions are different ways to give a score to a sentence

##### (1) Aggregation


def calculate_scores(sentence_graph):
    scores = np.zeros(len(sentence_graph))
    for i,sentence in enumerate(sentence_graph):
        scores[i] = sum(sentence_graph[i])
    return scores


##### (2) Page Rank


def calculate_pagerank_scores(sentence_graph):
    N = len(sentence_graph)
    d = 0.15   # PageRank Hyperparameter
    pagerank_scores = np.ones(N)
    
    out_degree = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if sentence_graph[i][j]:
                out_degree[i] += sentence_graph[i][j]
    
    for i in range(N):
        score = 0
        for j in range(N):
            if sentence_graph[j][i]:
                score += (pagerank_scores[j] / out_degree[j])
        pagerank_scores[i] = (d / N) + (1 - d) * score
    return pagerank_scores    


### Ranks the sentences based on any one of the above scoring methods and return the Summary


def rank_sentences_and_make_summary2(sentences, processed_sentences, sentence_graph, scores):
    scores_indices = np.argsort(scores)
    ordered_sentences = scores_indices[::-1]
    summary = []
    for i in range(5):
        summary.append(sentences[ordered_sentences[i]])
    return summary


def rank_sentences_and_make_summary(sentences, processed_sentences, sentence_graph, scores, summary_length):
    summary = []
    for i in range(summary_length): # Number of Sentences we want in the summary
        score_indices = np.argsort(scores)
        selected_index = score_indices[-1]
        summary.append(sentences[selected_index]) # Adding highest score sentence to summary
        mean_score = np.mean(sentence_graph)
        to_decrease = []
        # Calculated mean similarity score. If selected sentence and another sentence have
        # high similarity, the score of the second sentence should be reduced.
        # Here, have chosen to use 1.5 * mean_score as the threshold, and divided score in half.
        for iterator in range(len(processed_sentences)):
            if sentence_graph[iterator][selected_index] > 1.5 * mean_score:
                to_decrease.append(iterator)
            if sentence_graph[selected_index][iterator] > 1.5 * mean_score:
                to_decrease.append(iterator)
        for sentence in set(to_decrease):
            # Should be changed based on the number of sentences needed in the summary
            scores[sentence] /= (1 + 1.0 / summary_length) # Reduced score by half, to on average prevent from being picked.
        scores[selected_index] = 0
    return summary
        


#### Main Program which calls the above defined functions

def process(file_path, summary_length):
    text = load_data(file_path)
    
    list_of_sentences, sentence_metadata = get_sentences_with_metadata(text)
    list_of_sentences = [sentence.strip() for sentence in list_of_sentences if len(sentence) > 1]
    
    processed_sentences = make_processed_sentences(list_of_sentences)
    
    sentence_graph, sentence_common_graph = make_graph(processed_sentences, sentence_metadata)
    
    sentence_scores = calculate_scores(sentence_graph)
    
    sentence_page_scores = calculate_pagerank_scores(sentence_common_graph)
    
    sentence_score_final = [sentence_scores[i] * (sentence_page_scores[i]+1)  for i in range(len(sentence_scores))]
    
    summary = rank_sentences_and_make_summary(list_of_sentences, processed_sentences, sentence_graph, sentence_score_final, summary_length)
    return summary




### App handing of the user interactions

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index/')
def index():
    return render_template('index.html')

@app.route('/upload/', methods = ['GET', 'POST'])
def upload_text():
    if request.method == 'POST':
        file = request.files['file']
        upload_dir = './'
        filename = secure_filename(file.filename)
        file.save(os.path.join(upload_dir, filename))
        summary = process(filename, 5)
        print(summary)
    return render_template('display.html', sentences=summary) 




if __name__ == '__main__':
    app.run(debug=True)
