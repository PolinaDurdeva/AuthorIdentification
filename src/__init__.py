#-*- coding: utf-8 -*-
import TrainTestLoader as loader
import POS as pos
from nltk.data import load
import Constants
import Metrics as metr
import itertools
import numpy as np
import os

#test_data = loader.load_data(Constants.test_path, load_log_file = "log.txt")
#tagged_test_data = pos.tag_text_dict(test_data)



def evaluate_accuracy(test_profiles_dict, train_profiles_dict, common_profile = None, result_file = Constants.result_file):
    assessment = {'stomat_cng': 0, 'l1_norm': 0, 'kl': 0, 'cng':0, 'all':0}
    for test_author, text_profiles in  test_profiles_dict.items():
        for profile in text_profiles.values():
            distances_st = dict()
            distances_l1 = dict()
            distances_kl = dict()
            distances_cng = dict()
            for train_author, author_profile in train_profiles_dict.items():
                distances_l1[train_author] = metr.l1_norm(profile, author_profile, common_profile)
                distances_cng[train_author] = metr.cng(profile, author_profile)
                distances_kl[train_author] = metr.kullback_leibler_divergence(profile, author_profile,common_profile)
                if common_profile != None:
                    distances_st[train_author] = metr.stomat_cng(profile, author_profile, common_profile)
            suspicious_author_l1 = sorted(distances_l1, key = lambda k: distances_l1[k])
           # print suspicious_author_l1[0:2], test_author
            suspicious_author_kl = sorted(distances_kl, key = lambda k: distances_kl[k])
            suspicious_author_cng = sorted(distances_cng, key = lambda k: distances_cng[k])
            assessment['all'] += 1
            if common_profile != None:
                suspicious_author_st = sorted(distances_st, key = lambda k: distances_st[k])
                if (test_author in suspicious_author_st[0:2]):
                    assessment['stomat_cng'] +=1
            if (test_author in suspicious_author_l1[0:2]):
                assessment['l1_norm'] +=1
            if (test_author in suspicious_author_kl[0:2]):
                assessment['kl'] +=1
            if (test_author in suspicious_author_cng[0:2]):
                assessment['cng'] +=1
    print assessment
    loader.save_result_in_file(assessment, result_file)
            
#test base Orlov algorithm with pos-tags        
def test1(train, test, tagged_data, N, K = None ):
    common_profile = pos.create_common_profile(tagged_data, N, int(0.3* N))
    normalized_common_profile = pos.normalize(common_profile)
    train_profiles = pos.create_train_profiles(train, N, K)
    test_profiles = pos.create_test_profiles(test, N, K)
    evaluate_accuracy(test_profiles, train_profiles, normalized_common_profile)                

def test2(train, test, data, n = 1, x = 5, k = 50, s=10):
    train_patterns = pos.create_authors_patterns(train, n, x, k)
    test_pattern = dict()  
    for test_author, test_texts in test.items():
        t = dict()
        for name, text in test_texts.items():
            extracted_patterns = pos.extract_pos_pattern_from_text(text, n, x)
            t[name] = pos.get_k_most_freq_ngram(extracted_patterns, k)
        test_pattern[test_author] = t
    common_paterns = pos.create_common_patterns(data, n, x, s) 
    print "Pattern is created for texts"
    evaluate_accuracy(test_profiles_dict=test_pattern, train_profiles_dict=train_patterns, common_profile=common_paterns)
   
def create_pos_corpus(input_path, output_path):            
    test_data = loader.load_data(input_path, load_log_file = "log.txt", isClean=False)
    tagged_test_data = pos.tag_text_dict(test_data)
    pos.save_tagged_dict(tagged_test_data, output_path)
    

# if __name__ == "__main__":
    # logpath = os.path.join(Constants.logger_path, Constants.result_file)
    # #create_pos_corpus(Constants.test_book, Constants.pos_book_pos_without_p)
    # #authors=["DarrenSchuettler", "JanLopatka", "JohnMastrini", "JonathanBirt","AaronPressman", "GrahamEarnshaw","FumikoFujisaki","EdnaFernandes","JimGilchrist","BernardHickey"]
    # #train, test = loader.divide_on_train_and_test(test_data, ntrain=10, ntest=5)
    # #test4(train,test,test_data, 2,4,50)
    # #tagged_test_data = loader.extract_taglist(test_data)
    #
    #
    # log_file = open(logpath,"a")
    # test_data = loader.load_data(Constants.pos_book_pos_without_p,  isClean=False)
    # for (n,x) in [(1,3),(2,3),(2,4)]:
    #     for (ct,cn) in [(1,4),(3,2),(2,3) ]:
    #         log_file.write("n, x = "+str(n)+", " +str(x)+ "; train, test = " + str(ct) +", " + str(cn) + "\n")
    #         train, test = loader.divide_on_train_and_test(test_data, ntrain=ct, ntest=cn)
    #         print "start test differ k______________"
    #         for k in [50,100,150,200]:
    #             log_file.write("k = " + str(k))
    #             test2(train,test,test_data, n,x,k)
    #         log_file.flush()
    # log_file.write("_______________________\n")
    # log_file.close()
    #
    #
    #