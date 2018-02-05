#-*- coding: utf-8 -*-
'''
Created on Apr 4, 2017

@author: polina
'''

import nltk
import os
import Constants
from copy import deepcopy
#nltk.download('averaged_perceptron_tagger')

def tag_text_dict(author_text_dict):
    for texts in author_text_dict.values():
        for key, value in texts.items():
            texts[key] = get_pos_tags(value)
    print "Dictionary of texts is TAGGED!"
    return author_text_dict 

def save_tagged_dict(tagged_text_dict, path):
    for author, texts in tagged_text_dict.items() :
        path_s = os.path.join(path, author)
        if not os.path.isdir(path_s):
            os.makedirs(path_s)
        ind = 1
        for text in texts.values():
            save_tags_in_file(text, str(ind), path_s)
            ind += 1
    print "Tagged texts are SAVED!" 
    
def save_tags_in_file(tag_list, file_name, path):
    new_file = open(os.path.join(path,file_name +".txt"), "w")
    new_file.write('\n'.join(tag_list))
    new_file.close()
            
def get_pos_tags(text, ngram = 1):
    tokenized = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokenized)
    return [item for (_, item) in pos_tags]

def get_most_frequency_tag(tagged_text, n):
    tag_fd = nltk.FreqDist(tag for (_, tag) in tagged_text)
    return tag_fd.most_common(n)

#return list of ngram 
def get_ngrams(tag_list, n = 3):
    ngrams = list()
    for i in range(len(tag_list)- n + 1):
        ngrama = ','.join(tag_list[i: i + n])
        ngrams.append(ngrama)
    return ngrams

def create_ngram_freq_dict(ngram_list):
    freq_ngram_dict = dict()
    for ngrama in ngram_list:
        if freq_ngram_dict.has_key(ngrama):
            freq_ngram_dict[ngrama] += 1
        else:
            freq_ngram_dict[ngrama] = 1
    return freq_ngram_dict


def get_k_most_freq_ngram(freq_ngram_dict, k = 100):
    dict_len = len(freq_ngram_dict)
    if k <= dict_len:
        most_freq =  sorted(freq_ngram_dict.iteritems(), reverse = True, key = lambda (k,v):(v,k))[:k]
    elif k > dict_len:
        print "k is more than length of dictionary"
        most_freq = sorted(freq_ngram_dict.iteritems(), reverse = True, key = lambda (k,v):(v,k))
    return dict(most_freq)

def create_common_profile(tagged_text_dict, N, k_most_freq=None):
    common_ngram_list = list()
    for texts in tagged_text_dict.values() :
        for text in texts.values():
            ngrams = get_ngrams(text, N)
            common_ngram_list += ngrams
    common_profile = create_ngram_freq_dict(common_ngram_list)
    if k_most_freq != None:
        common_profile = get_k_most_freq_ngram(common_profile, k_most_freq)
    print "Common profile is CREATED!"
    return common_profile

def create_test_profiles(tagged_text_dict, N = 3, k_most_freq = None, normalized = True):
    authors_freq_ngrams = dict()
    for author, tagged_texts in tagged_text_dict.items() :
        ngram_frequency_dict = dict()
        for name, text in tagged_texts.items():
            ngrams = get_ngrams(text, N)
            profile = create_ngram_freq_dict(ngrams)
            if k_most_freq != None:
                profile = get_k_most_freq_ngram(profile, k_most_freq)
            if normalized:
                profile = normalize(profile)
            ngram_frequency_dict[name] = profile
        authors_freq_ngrams[author] = ngram_frequency_dict
    print "Test Profiles is CREATED!"
    return authors_freq_ngrams

def create_train_profiles(tagged_text_dict, N = 3, k_most_freq = None, normalized = True):
    authors_freq_ngrams = dict()
    for author, tagged_texts in tagged_text_dict.items() :
        ngrams= list()
        for text in tagged_texts.values():
            ngrams += get_ngrams(text, N)      
        profile = create_ngram_freq_dict(ngrams)
        if k_most_freq != None:
            profile = get_k_most_freq_ngram(profile, k_most_freq)
        if normalized:
            profile = normalize(profile)
        authors_freq_ngrams[author] = profile
    print "Train Profiles is CREATED!"
    return authors_freq_ngrams


def normalize(freq_ngram_dict):
    sum_ngram = sum(freq_ngram_dict.values()) 
    frequency_dict = map(lambda (_, v): (_,v * 1.0 / sum_ngram), freq_ngram_dict.iteritems())
    return dict(frequency_dict)

def extract_ngrams(freq_ngram_dict, common_profile):
    filtered_ngram = filter(lambda (k,v): k in common_profile.keys(), freq_ngram_dict.iteritems())
    return filtered_ngram

def get_needed_ngrams_test(ngrams_dict, needed_ngrams):
    authors_dict = dict()
    for author, ngrams in ngrams_dict.items():
        ngram_frequency_dict = dict()
        for name, ngram_text in ngrams.items():
            filtered_ngrams_list = filter(lambda (k,v): k in needed_ngrams, ngram_text.items())
            ngram_frequency_dict[name] = dict(filtered_ngrams_list)
        authors_dict[author] = ngram_frequency_dict
    print "Gets needed ngrams"
    return authors_dict

def get_needed_ngrams_train(ngrams_dict, needed_ngrams):
    authors_dict = dict()
    for author, ngram_text in ngrams_dict.items():
        filtered_ngrams = filter(lambda (k,v): k in needed_ngrams, ngram_text.items())
        authors_dict[author] = dict(filtered_ngrams)
    print "Gets needed ngrams"
    return authors_dict

def split_on_sentences(text):
 
    text = text.split('.')
    return text

######## by article
def extract_pos_pattern_from_text(text,min_len_pattern=1, max_len_pattern=3, normalized=True):
    text_sentences = split_on_sentences(text)
    #delete last element which is gap
    text_sentences.pop()
    text_ngrams = list()
    fng = dict()
    for n in range(min_len_pattern,max_len_pattern):
        for sentence in text_sentences:
            sentence_to_list = sentence.strip().split('\n')
            ngrams = get_ngrams(sentence_to_list,n)
            text_ngrams += set(ngrams)
        freq_ngram_dict = create_ngram_freq_dict(text_ngrams)
        if normalized:
            count_sentence = len(text_sentences)
            normalized = map(lambda (_, v): (_,v * 1.0 / count_sentence), freq_ngram_dict.iteritems())
            fng.update(dict(normalized))
        else:
            fng.update(freq_ngram_dict)
    return fng

def sum_by_unique_intersection(ds):
    r = dict()
    c = dict()
    for i in range(len(ds)):
        d = ds[i]
        for (k,v) in d.iteritems():
            r[k] = r.get(k, 0) + v
            c[k] = c.get(k, 0) + 1
    return dict([(k, r[k]) for k in r.keys() if c[k] == len(ds)])

def create_authors_patterns(train_dictionary, min_len_pattern, max_len_pattern, k_most_freq = 100):
    patterns_dict = dict()
    for train_author, train_texts in train_dictionary.items():
        pattern_current_author = list()
        count_sentences = 0
        for text in train_texts.values():
            extracted_pattern = extract_pos_pattern_from_text(text, min_len_pattern, max_len_pattern, normalized=False)
            most_freq_patterns = get_k_most_freq_ngram(extracted_pattern, k_most_freq)
            pattern_current_author.append(most_freq_patterns)
            text_sentences = split_on_sentences(text)
            count_sentences += len(text_sentences)
        #find intersection of text pattern of author
        common_patterns = sum_by_unique_intersection(pattern_current_author)
        normalized = map(lambda (_, v): (_,v * 1.0 / count_sentences), common_patterns.iteritems())
        patterns_dict[train_author] = dict(normalized)
    return patterns_dict

def create_common_patterns(data, min_len_pattern, max_len_pattern,k_most_freq):
    patterns = list()
    count_sentences = 0
    for texts in data.values():
        for t in texts.values():
            extracted_pattern = extract_pos_pattern_from_text(t, min_len_pattern, max_len_pattern, normalized=False)
            most_freq_patterns = get_k_most_freq_ngram(extracted_pattern, k_most_freq)
            patterns.append(most_freq_patterns)
            text_sentences = split_on_sentences(text)
            count_sentences += len(text_sentences)
    common_patterns = sum_by_unique_intersection(patterns)
    normalized = map(lambda (_, v): (_,v * 1.0 / count_sentences), common_patterns.iteritems())
    return dict(normalized)
        

        
#examples

# print extract_ngrams(f, d)
# text = "It is that we should fasten a bell round the neck of our enemy the cat, which will by its tinkling warn us of her approach."
# tagged =  get_pos_tags(text)
# print tagged
# print get_k_most_frequency_ngram(tagged, 3, 5)

#text = nltk.word_tsorted(freq_ngram_dict, reverse=True)[:k]
#tokenize("The quick brown fox jumps over the lazy dog")
#text = nltk.sent_tokenize("The quick brown fox jumps over the lazy dog.")
#print(text)