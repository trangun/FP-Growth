#!/usr/bin/env python
# coding: utf-8

# Reference:
# http://csc.lsu.edu/~jianhua/FPGrowth.pdf
# https://adataanalyst.com/machine-learning/fp-growth-algorithm-python-3/
# 

# In[1]:


import datetime as dt
import pandas as pd
import numpy as np
import random as rand
import operator
import csv
import time
import copy
import collections
from itertools import combinations


# In[2]:


file = r'E:\School\@Grad School\Data Mining\Project 1\adult.data'
#file.read()

df = pd.read_csv(file, sep=',',
                 names=["Age", "Workclass", "fnlwgt","education", "education_num","marital status","occupation","relationship","Race","Sex","capital_gain","capital_loss","hours_per_week","native country","income"],
                 skipinitialspace=True)
#df.replace(to_replace='?', value=np.nan)
df.head()


# In[3]:


def clean_data(df):
    df= df.applymap(lambda x: np.nan if x == '?' else x)
    df=df.dropna(axis=0)
    df.drop('fnlwgt', axis=1, inplace=True)
    df.drop('education_num', axis=1, inplace=True)
    df['Age'] = pd.cut(df['Age'], [0, 26, 46, 66,100], 
                        labels = ['Young', 'Middle-aged', 'Senior', 'Old'], 
                        right = True, include_lowest = True)
    df['hours_per_week'] = pd.cut(df['hours_per_week'], [0, 25, 40, 60, 100], 
                              labels = ['Part-time', 'Full-time', 'Over-time', 'Too-much'], 
                              right = True, include_lowest = True)
    df['capital_gain'] = pd.cut(df['capital_gain'], [0, 1, 10000000], 
                           labels = ['No-Gain', 'Positive-Gain'], 
                           right = True, include_lowest = True)
    df['capital_loss'] = pd.cut(df['capital_loss'], [0, 1, 10000000],
                            labels = ['No-Loss', 'Positive-Loss'], 
                            right = True, include_lowest = True)


    return df


# In[4]:


dataset = [['M', 'O', 'N', 'K', 'E', 'Y'], 
            ['D', 'O', 'N', 'K', 'E', 'Y'], 
            ['M', 'A', 'K', 'E'],
            ['M', 'U', 'C', 'K', 'Y'],
            ['C', 'O', 'O', 'K', 'I', 'E']]

def open_data():
    
    
    data = clean_data(df)
    dataset= data.values.tolist()
    return dataset
dataset= open_data()
dataset


# In[5]:


#scan data and find support for each item
def Scan_D(dataset):
    C1 = {}
    for item in dataset:
        for itemset in item:
            if itemset in C1:
                C1[itemset] += 1
            elif itemset not in C1:
                C1[itemset] = 1
    return C1
C1=Scan_D(open_data())
C1


# In[6]:


#discard infrequent item
def frequent (min_sup):
    frequent = {k:v for k, v in C1.items() if v >= min_sup*len(dataset)}
    return frequent
frequent(0.6)


# In[7]:


def initset (dataset):
    initial ={}
    for trans in dataset:
        initial[tuple(trans)] =1
    #print (initial)
    return initial
initset(dataset)


# In[8]:


#sort the frequent items in descending order based on support count
def sort_freq(dataset, min_sup=0.6):
    freq = frequent(min_sup)
    freq_set = set(freq.keys())
    sorted_freq = [] 
    for trans in dataset:
        freq_trans = {}
        for item in trans:
            if item in freq_set:
                freq_trans[item] = freq[item]
        if len(freq_trans) > 0:
            sort_freq = [val[0] for val in sorted(freq_trans.items(), key=lambda x: x[1], reverse=True)]
            sorted_freq.append(sort_freq)
    
    return sorted_freq
#sort_freq(dataset)


# In[9]:


class tree:
    def __init__ (self, value, freq, parents):
        self.name = value
        self.count = freq
        self.node_link = None
        self.parent = parents
        self.children = {}
        
        
    
    #increment the count
    def inc_count (self, count):
        self.count += count
        


# In[10]:


#FP tree
def make_tree(dataset, min_sup=.6):
    
    #first scan  
    freq = frequent(min_sup)    
    freq_set = set(freq.keys())
    if len(freq_set) == 0:
        return None, None
    
    for f in freq:
        freq[f]= [freq[f], None]
    
    root = tree('Null', 1, None)
    #i=0
    for transaction, count in dataset.items():       
        freq_trans = {}  
       
        for itemset in transaction:
            if itemset in freq_set:
                freq_trans[itemset] = freq[itemset][0]
            
        if len(freq_trans) > 0:
            sort_freq = [val[0] for val in sorted(freq_trans.items(), key=lambda x:x[1], reverse=True)]
            get_path(sort_freq, root, freq, count)
            #i+=1
    return root, freq


# In[11]:


def get_path(itemset, root, frequent, count):

    if itemset[0] not in root.children:
        root.children[itemset[0]] = tree(itemset[0], count, root)
        
        if frequent[itemset[0]][1] == None:
            frequent[itemset[0]][1] = root.children[itemset[0]]
        else:
            update_nodelink(frequent[itemset[0]][1], root.children[itemset[0]])

    else:
        root.children[itemset[0]].inc_count(count)
    if len(itemset) >1:
        get_path(itemset[1::], root.children[itemset[0]], frequent, count)


# In[12]:


def update_nodelink(test_node, target_node):
    while (test_node.node_link != None):
        test_node = test_node.node_link
    test_node.node_link = target_node


# In[13]:


#FPtree, count = make_tree(init, 0.6)

#myHeaderTab
#FPtree.display()
#count


# In[14]:


def go_up(leaf, prefix): #ascends from leaf node to root
    if leaf.parent != None:
        prefix.append(leaf.name)
        go_up(leaf.parent, prefix)


# In[15]:


def find_prefix(pattern_base, tree): 
    #node = tree[pattern_base][1][1]
    cond_pattern = {}
    while tree != None:
        prefix = []
        go_up(tree, prefix)
        if len(prefix) > 1: 
            cond_pattern[tuple(prefix[1:])] = tree.count
        tree = tree.node_link
    return cond_pattern


# In[16]:


#find_prefix('Y',count['Y'][1])


# In[17]:


def condition_FPtree(FPtree, freq, min_sup,prefix, freq_set):
    L=[v[0] for v in sorted(freq.items(),key=lambda x: x[1])]
    #=print( L)
    
    for pattern_base in L:
        new_freqset = prefix.copy()
        new_freqset.add(pattern_base)
        #add frequent itemset to final list of frequent itemsets
        freq_set.append(new_freqset)
        conditional_pattern_base = find_prefix(pattern_base, freq[pattern_base][1])
        #print(conditional_pattern_base)
        
        if(conditional_pattern_base!={}):
            conditional_FPtree, conditional_freq = make_tree(conditional_pattern_base, min_sup)
        #conditional_FPtree.display()
        
            if conditional_freq != None:
                condition_FPtree(conditional_FPtree, conditional_freq, min_sup, new_freqset, freq_set)


# In[18]:


init= initset(dataset)
s = time.time()
FPtree, count = make_tree(init)
freq_set=[]
condition_FPtree(FPtree, count, 0.1,set([]), freq_set)
e = time.time()
r=e-s
r


# In[19]:


freq_set 

