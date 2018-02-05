'''
Created on May 4, 2017

@author: polina
'''
import numpy as np
from math import sqrt

def stomat_cng(text_profile, author_profile, common_profile):
    set_of_ngram = np.union1d(text_profile.keys(), author_profile.keys())
    print set_of_ngram
    sum_v = dict(map(lambda k: (k, text_profile.get(k,0) + author_profile.get(k,0)), set_of_ngram))
    diff_v = dict(map(lambda k: (k, 2.0 * (text_profile.get(k,0) - author_profile.get(k,0))), set_of_ngram))
    div_v = dict(map(lambda k: (k, (diff_v.get(k,0) / sum_v.get(k)) ** 2), set_of_ngram))

    sum_df = dict(map(lambda k: (k, text_profile.get(k,0) + common_profile.get(k,0)), common_profile.keys()))
    diff_df = dict(map(lambda k: (k, 2.0 * (text_profile.get(k,0) - common_profile.get(k,0))), common_profile.keys()))
    div_df = dict(map(lambda k: (k, (diff_df.get(k,0) / sum_df.get(k)) ** 2), common_profile.keys()))
    proud = map(lambda k: div_v.get(k,0) * div_df.get(k,0), common_profile.keys())
    distance = np.sum(proud)
    return distance

def cng(text_profile, author_profile):
    set_of_ngram = np.union1d(text_profile.keys(), author_profile.keys())
    sum_v = map(lambda k: (text_profile.get(k,0) + author_profile.get(k,0))**2, set_of_ngram)
    diff_v = map(lambda k: 2.0 * (text_profile.get(k,0) - author_profile.get(k,0))**2, set_of_ngram)
    div_v = np.divide(diff_v, sum_v)
    distance = np.sum(div_v)
    return distance

def l1_norm(text_profile, author_profile, common_profile):
    set_of_ngram = np.union1d(text_profile.keys(), author_profile.keys())
#    int_set = np.intersect1d(text_profile.keys(), author_profile.keys())
#    int_set = np.setdiff1d(set_of_ngram, common_profile.keys())
    subtract = map(lambda k: np.abs(text_profile.get(k,0) - author_profile.get(k,0)), set_of_ngram)
    distance = np.sum(subtract)
    return distance

def kullback_leibler_divergence(text_profile,author_profile,common_profile):
    set_of_ngram = np.union1d(text_profile.keys(), author_profile.keys())
    int_set = np.intersect1d(text_profile.keys(), author_profile.keys())
    j = len(int_set) * 1.0 / len(set_of_ngram)
    sum_v = map(lambda k:  (text_profile.get(k,0) + author_profile.get(k,0)), int_set)
    diff_v = map(lambda k: 2.0 * (text_profile.get(k,0) - author_profile.get(k,0)), int_set)
    div_v = np.divide(diff_v, sum_v)
    pow_v = div_v ** 2
    distance = np.sum(pow_v)/j
    return distance

def countCommonNgrams(text_profile,author_profile):
    set_of_ngram = np.intersect1d(text_profile.keys(), author_profile.keys())
    union_set = np.union1d(text_profile.keys(), author_profile.keys())
    #jaccard_distance(s1, s2)
    return len(set_of_ngram)/len(union_set)    

def pearson_correlation(text_profile, author_profile):
    set_of_ngram = np.intersect1d(text_profile.keys(), author_profile.keys())
    aavg = 0.0
    tavg = 0.0
    count = len(set_of_ngram)
    for ngram in set_of_ngram:
        aavg += author_profile.get(ngram)
        tavg += text_profile.get(ngram) 
    aavg = aavg / count
    tavg = tavg / count
    s1 = 0
    s2 = 0
    d1 = 0
    d2 = 0
    for ngram in set_of_ngram:
        s1 += text_profile.get(ngram) - tavg
        s2 += author_profile.get(ngram) - aavg
        d1 += s1**2
        d2 += s2**2
    pearson = (s1 * s2) / (sqrt(d1) * sqrt(d2) + 1.0)
    return pearson
    
    