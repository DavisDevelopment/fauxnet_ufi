# -*- coding: utf-8 -*-
from math import log, sqrt
import torch
import random
from torch import Tensor
from torch.jit import script

def unique_counts(labels):
    """
    Unique count function used to count labels.
    """
    assert isinstance(labels, Tensor)
    labels = torch.asarray(labels)
    
    results = {}
    for label in labels:
        value = label.item()
        if value not in results.keys():
            results[value] = 0
        
        results[value] += 1
    
    # print(results.keys)
    return results

from typing import *

# @script
def split_function(vector:Tensor, column:int, value:Tensor):
    """
    Split function
    """
    return vector[column] >= value

# @script
def divide_set(vectors, labels, column:int, value):
    # print(column, value)
    """
    Divide the sets into two different sets along a specific dimension and value.
    """
    # set_1 = [(vector, label) for vector, label in zip(vectors, labels) if split_function(vector, column, value)]
    # set_2 = [(vector, label) for vector, label in zip(vectors, labels) if not split_function(vector, column, value)]
    n:List[float] = [0.0]
    set_1:Tuple[List[Tensor], List[float]] = ([], n[0:0])
    set_2:Tuple[List[Tensor], List[float]] = ([], n[0:0])
    for (vector, label) in zip(vectors, labels):
        a = (set_1 if split_function(vector, column, value) else set_2)
        vecs, lbls = a
        vecs.append(vector)
        lbls.append(label.item())

    empty:List[float] = []
    vectors_set_1 = set_1[0]
    vectors_set_1 = torch.vstack(vectors_set_1) if len(vectors_set_1) > 0 else torch.tensor(empty)
    vectors_set_2 = set_2[0]
    vectors_set_2 = torch.vstack(vectors_set_2) if len(vectors_set_2) > 0 else torch.tensor(empty)
    label_set_1   = torch.tensor(set_1[1])
    label_set_2   = torch.tensor(set_2[1])
    # print(vectors_set_1[0], label_set_1[0])
    return vectors_set_1, label_set_1, vectors_set_2, label_set_2


def log2(x):
    """
    Log2 function
    """
    return log(x) / log(2)


def sample_vectors(vectors, labels, nb_samples):
    """
    Sample vectors and labels uniformly.
    """
    sampled_indices = torch.LongTensor(random.sample(range(len(vectors)), nb_samples))
    sampled_vectors = torch.index_select(vectors,0, sampled_indices)
    sampled_labels = torch.index_select(labels,0, sampled_indices)

    return sampled_vectors, sampled_labels


def sample_dimensions(vectors):
    """
    Sample vectors along dimension uniformly.
    """
    sample_dimension = torch.LongTensor(random.sample(range(len(vectors[0])), int(sqrt(len(vectors[0])))))

    return sample_dimension


def entropy(labels):
    """
    Entropy function.
    """
    results = unique_counts(labels)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(labels)
        ent = ent - p * log2(p)
    return ent


def variance(values):
    """
    Variance function.
    """
    mean_value = mean(values)
    var = 0.0
    for value in values:
        var = var + torch.sum(torch.sqrt(torch.pow(value-mean_value,2))).item()/len(values)
    return var


def mean(values):
    """
    Mean function.
    """
    m = 0.0
    for value in values:
        m = m + value/len(values)
    return m
