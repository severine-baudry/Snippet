#!/usr/bin/env python
# coding: utf-8

# In[38]:


import json
import os
import sys
import argparse
from git import Repo
from collections import OrderedDict
import time
import pandas as pd
import matplotlib.pyplot as plt

# In[3]:


def find_output_marking(notebook, output_mark):
    for cell_idx,cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            for output in cell["outputs"]:
                if "text" in output :
                    found = False
                    for line in output["text"]:
                        if line.find(output_mark)!= -1:
                            #print("MARK FOUND !", cell_idx)
                            found = True
                    if found :
                        return output["text"]


# In[5]:


import re
def tokenize_output(output_text):
    '''
    split notebook into tokens 
    separators : whitespace  ','  ':'
    '''
    res =[]
    for line in output_text :
    #    a = re.split(' |\n|:', line)
        #a = re.split('\s|:|,', line)
        # split outputs on whitespace, : and ,
        split_line = re.split('[\s|:|,]+', line)
        # remove empty tokens
        split_line = [ a for a in split_line if len(a)]
        #print(split_line)
        res.append(split_line)
    return res


# In[33]:
from collections import OrderedDict
def parse_nn_performances(split_lines):
    res = OrderedDict( )
    for line in split_lines:
        if 'VALID' in line :
            res.setdefault('VALID', OrderedDict())
            perf_type = 'VALID'
        elif 'TEST' in line :
            res.setdefault('TEST', OrderedDict())
            perf_type = 'TEST'            
        else :
            perf_type = 'TRAIN'
            res.setdefault('TRAIN', OrderedDict())
        if not 'Epoch' in line :
            print("not a result line")
            continue
        d_index = {}
        l_names = [ 'loss', 'accuracy', 'Epoch']
        d_type = {'loss':float, 'accuracy':float, 'Epoch':int }
        d_val = {}
        for name in l_names :
            d_index[name] = line.index(name)
        if d_index['loss'] == -1 and d_index['accuracy'] == -1 :
            print("error : not a performance line", line)
        else :
            try :
                for name, index in d_index.items():
                    if index > -1 :
                        d_val[name] = d_type[name](line[index+1])
            except ValueError :
                print("error conversion", name, index, line)
            else :
                epoch = d_val["Epoch"]
                del(d_val['Epoch'])
                for name, perf in d_val.items() :
                    res[perf_type].setdefault(name, OrderedDict() )
                    res[perf_type][name][ epoch] = perf

    return res
                      

import matplotlib.pyplot as plt
def plot_performance(results, metric):
    plt.figure()
    plt.plot( list( res["TRAIN"][metric].keys()), list(res["TRAIN"][metric].values() ) , label ="TRAIN")
    plt.plot( list( res["VALID"][metric].keys()), list(res["VALID"][metric].values() ) , label ="VALID")
    plt.legend()
    plt.title(metric)    


# In[34]:


def extract_perf_(notebook_f, l_markers = ["Begin Training", "TEST" ] ):
        notebook = json.load(notebook_f)
        res = OrderedDict()
        for output_mark in  l_markers :
            # find output cell beginning with output_mark
            output_text = find_output_marking(notebook, output_mark)
            if output_text :
                # tokenize output cell
                split_lines = tokenize_output(output_text)
                # extract results from cell
                dict_result = parse_nn_performances(split_lines)
                res.update(dict_result)
        return res

def extract_notebook_train_valid(notebook_name, l_markers = ["Begin Training", "TEST" ]  ):
    if type(notebook_name) is str :
        with open(notebook_name) as f :
            return extract_perf_(f, l_markers)
    else :
        return extract_perf_(notebook_name)
# In[43]:


def test_notebook_parsing():
    results = extract_notebook_train_valid("/home/severine/MOOCS/UDACITY/DEEP_LEARNING/TP/P2_dog_classification/Transfer_Learning_Solution_copy.ipynb")
    plot_performance(results, "loss")                 
    plot_performance(results, "accuracy")

def extract_metric_phase(l_results, phase, metric):
    df_result = pd.DataFrame()
    for result in result :
        sha = result["sha"][:5]        
        dres = result[phase][metric]
        df_current = pd.DataFrame.from_dict(dres, orient = 'index')
        df_result = pd.concat([df_current], axis = 1)
    return df_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest = "directory", required = True)
    parser.add_argument("-nb", dest = "notebook", required = True)
    parser.add_argument("-o", dest = "output", required = True)
    l_args = parser.parse_args()
    
    print(l_args.notebook)
    print(l_args.output)
    
    repo = Repo(l_args.directory)
    head = repo.head.reference

    l_results = []
    # iterate on the previous commits
    for commit in list( repo.iter_commits( ) ) :
        sha = commit.hexsha
        msg = commit.message
        dat = commit.authored_date
        strdate = time.strftime("%d/%m/%Y %H:%M", time.gmtime(dat))
        # dat = commit.commited_date
        # files in the commit
        for tr in commit.tree:
            # load the notebook
            if tr.name == l_args.notebook:
                print(sha[:7], strdate, msg )
                results = extract_notebook_train_valid(tr.data_stream)
                res_dict = OrderedDict( [("sha", sha), ("date", strdate), ("msg", msg), ("res", results) ] )
                l_results.append(res_dict)
    with open(l_args.output, "w") as fs :
        json.dump(l_results, fs, indent = 2)
    

#python notebook_parsing.py  -d ../P2_dog_classification/ -nb Transfer_Learning_Solution_copy.ipynb  -o dogs_transfer_learning.json

