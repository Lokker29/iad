#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn import datasets


# In[2]:


pd.options.mode.chained_assignment = None

np.random.seed(1)


# In[3]:


CLUSTERS_COUNT = 3

iris = datasets.load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names

preprocessed_data = data.copy()

DIMENSIONAL = data.shape[1]
SIZE = data.shape[0]

EPSILON = 0.0005


# In[4]:


def encode_on_hypercube(data):
    """
    Метод для кодирования на гиперкуб
    
    :param data: DataFrame instance
    return DataFrame instance
    """
    def encode_by_column(column):
        minimum, maximum = np.min(column), np.max(column)
        return np.apply_along_axis(lambda x: 2 * (x - minimum) / (maximum - minimum) - 1, 0, column)
    
    return np.apply_along_axis(encode_by_column, 0, data)


# In[5]:


preprocessed_data = encode_on_hypercube(preprocessed_data)
np.random.shuffle(preprocessed_data)

cluster_column = np.zeros((data.shape[0], 1))
preprocessed_data = np.append(preprocessed_data, cluster_column, axis=1)
preprocessed_data


# In[6]:


centroids = np.random.uniform(low=-1, high=1, size=(CLUSTERS_COUNT, DIMENSIONAL))
centroids


# In[7]:


def get_distance(row, center, sqrt=True):
    diff = row - center
    result = np.dot(diff.reshape((1, DIMENSIONAL)), diff.reshape((DIMENSIONAL, 1)) )

    return np.sqrt(result) if sqrt else result


# In[8]:


def identify_members_of_clusters(data, centroids):
    new_data = data.copy()
    for index, row in enumerate(data):
        centers = np.array([get_distance(row[:DIMENSIONAL], center) for center in centroids])
        new_data[index][-1] = centers.argmin()
    
    new_centroids = centroids.copy()
    for index in range(centroids.shape[0]):
        cluster_filter = new_data.T[DIMENSIONAL] == index
        new_centroids[index] = np.mean(new_data[cluster_filter], axis=0)[:DIMENSIONAL]
        
    assert new_data.shape == (SIZE, DIMENSIONAL + 1)
    assert new_centroids.shape == (CLUSTERS_COUNT, DIMENSIONAL)
    
    return new_data, new_centroids


# In[9]:


def get_answer(data, centroids):
    centroids = centroids.copy()
    data, new_centroids = identify_members_of_clusters(data, centroids)
    while any([get_distance(new_centroids[index], centroids[index], sqrt=False) > EPSILON
               for index in range(centroids.shape[0])
              ]):
        
        centroids = new_centroids
        data, new_centroids = identify_members_of_clusters(data, centroids)

    return data, new_centroids


# In[10]:


clustered_data, centroids = get_answer(preprocessed_data, centroids)


# In[11]:


def print_results(data, centroids):
    for index in range(centroids.shape[0]):
        print(f'Count objects in the {index + 1} cluster:', data[data.T[DIMENSIONAL] == index].shape[0])
    
    print()
    print('Centroids:', centroids, sep='\n')
    
    sum_distances = [
        np.array([get_distance(row[:DIMENSIONAL], centroid) for row in data[data.T[DIMENSIONAL] == index]]).sum()
        for index, centroid in enumerate(centroids)
    ]
    print()
    print('Sum of distances:', sum_distances)
    
    distances = [
        [round(get_distance(row[:DIMENSIONAL], centroid).item(),  3)
        for row in data[data.T[DIMENSIONAL] == index]]
        for index, centroid in enumerate(centroids)
    ]
    print()
    print('Distances:')
    for index, row in enumerate(distances, start=1):
        print(f'\tCluster №{index}:', row)


# In[12]:


print_results(clustered_data, centroids)

