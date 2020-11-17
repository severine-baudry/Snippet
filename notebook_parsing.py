#!/usr/bin/env python
# coding: utf-8

# In[38]:


import json
import os
import sys
import argparse
from git import Repo


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


import matplotlib.pyplot as plt
def plot_performance(results, metric):
    plt.figure()
    plt.plot( list( res["TRAIN"][metric].keys()), list(res["TRAIN"][metric].values() ) , label ="TRAIN")
    plt.plot( list( res["VALID"][metric].keys()), list(res["VALID"][metric].values() ) , label ="VALID")
    plt.legend()
    plt.title(metric)    


# In[34]:


def extract_notebook_train_valid(notebook_name, l_markers = ["Begin Training", "TEST" ] ):
    with open(notebook_name) as f :
        notebook = json.load(f)
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


# In[43]:


def test_notebook_parsing():
    results = extract_notebook_train_valid("/home/severine/MOOCS/UDACITY/DEEP_LEARNING/TP/P2_dog_classification/Transfer_Learning_Solution_copy.ipynb")
    plot_performance(results, "loss")                 
    plot_performance(results, "accuracy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest = "directory")
    parser.add_argument("-nb", dest = "notebook")
    parser.add_argument("-o", dest = "output")
    l_args = parser.parse_args()
    
    print(l_args.notebook)
    print(l_args.output)
    
    repo = Repo(l_args.directory)
    head = repo.head.reference
    git_log = head.log()
    print(len(git_log))
    for l in reversed(git_log):
        print(l)
    commit = repo.commit()
    print("commit:", commit.hexsha, commit.message)
    for commit in list( repo.iter_commits( max_count = 10) ) :
        print(commit.hexsha[:5], commit.message)
        for tr in commit.tree:
            print("\t", tr, type(tr), tr.name)
            if tr.name == "Transfer_Learning_Solution_copy.ipynb":
                toto = json.load(tr.data_stream)
                print(type(toto))
            
    

