import re
import nltk
import math
from nltk.tokenize import word_tokenize


def clean_text(file_name):
    file = open(file_name, 'r')
    file_data = file.readlines()
    article = file_data[0].split('. ')
    sentences = []
    for sen in article:
        sen = re.sub('[^a-zA-Z]', " ", str(sen))
        sen = re.sub('[\s+]', ' ', sen)
        sentences.append(sen)
    sentences.pop()
    display = " ".join(sentences)
    print("initial text: ")
    print(display)
    print('\n')
    return sentences

def count_words(sent):
    count = 0
    words = word_tokenize(sent);
    for word in words:
        count+=1
    return count

def count_in_sentences(sentences):
    txt_data = []
    i = 0
    for sent in sentences:
        i+=1
        count = count_words(sent)
        temp = {'id': i, 'word_count': count}
        txt_data.append(temp)
    return txt_data

def freq_dict(sentences):
    i = 0 
    freq_list = []
    for sent in sentences:
        i+=1
        freq_dict = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            if word in freq_dict:
                freq_dict[word]+=1
            else:
                freq_dict[word] = 1
            temp = {'id': i, "freq_dict": freq_dict}
        freq_dict.append(temp)
    return freq_dict

def calc_TF(text_data, freq_list):
    tf_scores = []
    for item in freq_dict:
        ident = item['id']
        for k in item["freq_dict"]:
            temp ={
                'id': item['id'],
                'tf_score': item['freq_list'][k]/text_data[ident-1]['word_count'], 'key': k
            }
            tf_scores.append(temp)
    return tf_scores

def calc_IDF(text_data, freq_list):
    idf_scores =[]
    count = 0
    for item in freq_list:
        count+=1
        for k in item['freq_dict']:
            val = sum([k in it['freq_dict'] for it in freq_list])
            temp = {
                'ident': count,
                'idf_score': math.log(len(text_data/(val+1))),
                'key' : k
            }
            idf_scores.append(temp)
    return idf_scores

def calc_TFIDF(tf_scores, idf_scores):
    tfidf_scores = []
    for j in idf_scores:
        for i in tf_scores:
            if j['key'] == i['key'] and j['id'] == i['id']:
                temp = {
                    'id': j['id'],
                    "tfidf_score": j["idf_score"] * i['tf_score'],
                    'key': j['key']
                }
                tfidf_scores.append(temp)
    return tfidf_scores

def sent_scores(tfidf_scores, sentences, text_data):
    sent_data = []
    for txt in sent_data:
        score = 0
        for i in range(0, len(tfidf_scores)):
            t_dict = tfidf_scores[i]
            if txt['id'] == t_dict['id']:
                score += t_dict['tfidf_score']
        temp = {
            'id': txt['id'],
            'score': score,
            'sentence': sentences[txt['id']-1]
        }
        sent_data.append(temp)
    return sent_data

def summary(sent_data):
    count = 0
    summary = []
    for t_dict in sent_data:
        count += t_dict['score']
    avg = count / len(sent_data)
    for sent in sent_data:
        if sent['score'] >= (avg * .9):
            summary.append(sent['sentence'])
    summary = ". ".join(summary)
    return summary

sentences = clean_text("summary.txt")
text_data = count_in_sentences(sentences)
freq_list = freq_dict(sentences)
tf_scores = calc_TF(text_data, freq_list)
idf_scores = calc_IDF(text_data, freq_list)
tfidf_scores = calc_TFIDF(tf_scores, idf_scores)
sent_data = sent_scores(tfidf_scores, sentences, text_data)
results = summary(sent_data)
print("Final Summary: ")
print(results)


