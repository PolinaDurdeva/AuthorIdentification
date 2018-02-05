#-*- coding: utf-8 -*-
'''
Created on May 3, 2017

@author: polina
'''
### PATH ###
import os

main_path = os.path.join(os.getcwd(), "..", "Data")
train_path = os.path.join(main_path, "train")
test_path = os.path.join(main_path, "test")
test = os.path.join(test_path, "test")
test_article = os.path.join(test_path,"ARTICLE")
test_book = os.path.join(test_path,"BOOK")
logger_path = os.path.join(os.getcwd(), "..", "Log")
pos_path = os.path.join(os.getcwd(), "..", "POS")
en_article_pos = os.path.join(os.getcwd(), "..", "POS_article")
pos_book_pos_without_p = os.path.join(pos_path, "POS_WP_BOOK")
pos_book = os.path.join(pos_path, "POS_BOOK")
result_file = os.path.join(logger_path, "result.txt")
pos_without_punctuation = os.path.join(pos_path, "POS_WP")
pos_json_en = os.path.join(pos_path, "POS_JSON_EN") 

