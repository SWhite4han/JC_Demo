# Script for ElasticSearch score script
import json
import math
import numpy as np


def match_cos():
    return np.sum(d1*c1)/math.sqrt(np.sum(d1*d1))*math.sqrt(np.sum(c1*c1))

def match_eucdiean():
    pass


"""
Main process
"""
# 1. Convert source/target to variable 'List of float'
#    doc prepare by ElasticSearch
#    method/imgVec send by parameter of query
d1 = np.asarray(doc['imgVec'])
c1 = np.asarray(imgVec)

# 2. Choose algorithm and score
if method == 'cos':
    match_cos()
elif method == 'eucdiean':
    match_eucdiean()
else:
    0.0

