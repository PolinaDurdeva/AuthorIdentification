# -*- coding: utf-8 -*-
'''
Created on Apr 4, 2017

@author: polina
'''
import os
import codecs
import string
import Constants
from copy import deepcopy

def load_data(datapath, author_names=None, books_names=None, load_log_file=None, isClean = False):
    author_list = list()
    auth_texts_dict = dict()
    if os.listdir(datapath):
        for author_folder_name in os.listdir(datapath):
            if (author_names == None or (author_names != None and author_folder_name in author_names)):
                author_list.append(author_folder_name)
                author_folder_path = os.path.join(datapath, author_folder_name)
                texts_list = dict()
                for filename in os.listdir(author_folder_path):
                    if (books_names == None or (books_names != None and filename in books_names)):
                        # #Problem with encoding sometimes
                        filepath = os.path.join(author_folder_path, filename)
                        texts_list[filename] = open_text(filepath, isClean=isClean)
                auth_texts_dict.update([(author_folder_name, texts_list)])
    # print in log file
    if (load_log_file != None and os.listdir(Constants.logger_path)):
        logpath = os.path.join(Constants.logger_path, load_log_file)
        log_file = open(logpath, "a")
        log_file.write("Authors: \n")
        log_file.write('\n'.join(author_list))
        log_file.write("\n")
        log_file.close()
    print ("Loading is DONE")
    return auth_texts_dict

def open_text(filepath, isClean=False):
    text_open = codecs.open(filepath, encoding="utf-8")
    text = text_open.read()
    if isClean:
        text = clean_eng_text(text)
    text_open.close()
    return text

def save_result_in_file(evaluation, path):
    res = open(Constants.result_file, "a")
    res.write(to_string_dict(evaluation))
    res.close()
    print "POS files are SAVED!"
    
def to_string_dict(dictionary):
    string = ""
    for key, value in dictionary.items():
        string += key + ", " + str(value) + " "
    return string + "\n"

def load_train_data():
    return load_data(Constants.train_path, load_log_file="log.txt")
    
def load_test_data():
    return load_data(Constants.test_path, load_log_file="log.txt")

def clean_eng_text(text):
    text = text.lower()
    punctuation = "!?. "
    allowed_symbols = string.ascii_lowercase +  punctuation
    text = filter(lambda x: x in allowed_symbols, text)
    return text

def clean_ru_text(text):
    # TODO: revise
    text = text.lower()
    text = filter(lambda x: x in string.ascii_lowercase, text)
    return text

def extract_taglist(authors_tags_dict, separator='\n'):
    for tags_dict in authors_tags_dict.values():
        for book_name, tag_list in tags_dict.items():
            tags_dict[book_name] = split_text(tag_list, separator)
    return authors_tags_dict

def split_text(text, separator='\n'):
    return text.split(separator)

#return dictionarys{"author"[(name_book, text),(name_book, text, ), ...],....}
def divide_on_train_and_test(author_tag_dict, ntrain = 0.8, ntest = None):
    train_dict = dict()
    test_dict = dict() 
    count_train = 0
    count_test = 0
    if ntrain >= 1.0:
            count_train = ntrain
            count_test = ntest + ntrain  
    for author, atexts in author_tag_dict.items():
        if ntrain < 1.0:
            len_atexts = len(atexts)
            count_train = int(round(ntrain * len_atexts))
            count_test = len_atexts
        train_dict[author] = dict(atexts.items()[0:count_train])
        test_dict[author] = dict(atexts.items()[count_train : count_test-1])
    return train_dict, test_dict

def save_ngrams_dictionary(ngrams_dict, path, file_name):
    new_file = open(os.path.join(path,file_name +".txt"), "w")
    for author, text_dict in ngrams_dict.items() :
        new_file.write(author + '\n')
        for name, ngrams in text_dict.items():
            new_file.write(name + ': ' + str(ngrams.keys()))
    new_file.close()            
    print "NGRAMS are SAVED!"        
#####
            
        
