#-*- coding: utf-8 -*-
'''
Created on Jul 30, 2017

@author: polina
'''

import nltk
import re
import string
import collections
import json
import os
import codecs
from collections import Counter
import pymorphy2


def jsonify_folder(input_folder_path, output_folder_path):
    if os.listdir(input_folder_path):
        author_name = os.path.basename(input_folder_path)
        tagged_text = dict()
        for filename in os.listdir(input_folder_path):
            filepath = os.path.join(input_folder_path, filename)
            text = codecs.open(filepath, encoding="utf-8").read()
            tagged_text['author'] = author_name
            print(author_name)
            tagged_text['book_name'] = filename
            tagged_text['number_words'] = get_number_words(text)
            tagged_text['text'] = get_tagged_text(text)
            tagged_text['count_sentences'] = len(tagged_text['text'])
            with open(os.path.join(output_folder_path,filename), 'w') as fp:
                json.dump(tagged_text, fp)
    else:
        print("No such folder")
    print("Jsonify is finished!")

def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens

def remove_characters_before_tokenization(sentence,keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
        PATTERN = r'[?|&|$|*|%|@|(|)|~]' # add other characters here to
    else:
        PATTERN = r'[^a-zA-Z0-9|\n ]' # only extract alpha-numeric characters
    filtered_sentence = re.sub(PATTERN, r'', sentence)
    return filtered_sentence

def remove_none(tokens):
    return filter(lambda x: x is not None, tokens)

def get_tagged_text(text):
    sentences = nltk.sent_tokenize(text)
    clean_sentences = [remove_characters_before_tokenization(sentence, keep_apostrophes=False) for sentence in sentences]
    word_tokens = [nltk.word_tokenize(sentence) for sentence in clean_sentences]
    tagged_text = [get_pos_tags(nltk.pos_tag(words)) for words in word_tokens]
    return  tagged_text 

def get_tagged_text_ru(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    morph = pymorphy2.MorphAnalyzer()
    tagged_text = [[morph.parse(w)[0].tag.POS for w in words] for words in word_tokens]
    tagged_text = [remove_none(tokens) for tokens in tagged_text]
    return  tagged_text 

def get_pos_tags(tagged_text):
    return [item for (_, item) in tagged_text]

def get_number_words(text):
    words =  re.findall('[a-z]+', text.lower())
    return sum(collections.Counter(words).values())

def get_ngrams_set(tags_sequence, n = 3):
    ngrams = list()
    for i in range(len(tags_sequence)- n + 1):
        ngrama = ','.join(tuple(tags_sequence[i: i + n]))
        ngrams.append(ngrama)
    return set(ngrams)

def get_frequency_dictionary(tagged_text, n=[3], most_common = None):
    tags = []
    for sentence in tagged_text:
        for k in n:
            tags += get_ngrams_set(sentence, k)
    d = Counter(tags)
    if most_common != None:
        return d.most_common(most_common)
    return d

# def split_on_tags(text):
    

# jsonify_folder(os.path.join(Constants.test_book,"Charles"), os.path.join(Constants.pos_json_en,"Charles"))
text = u'''Время волшебников прошло. По всей вероятности, их никогда и  не  было
на самом деле. Все это выдумки и  сказки  для  совсем  маленьких  детей.
Просто некоторые фокусники умели так ловко обманывать всяких зевак,  что
этих фокусников принимали за колдунов и волшебников.'''

tagged_text = get_tagged_text_ru(text)
print tagged_text
#print get_frequency_dictionary(tagged_text, [2]).most_common(5)
print get_frequency_dictionary(tagged_text).most_common(5)
