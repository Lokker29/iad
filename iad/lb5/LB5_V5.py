#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from sklearn import datasets


# In[ ]:


pd.options.mode.chained_assignment = None

np.random.seed(1)


# In[ ]:


CLUSTERS_COUNT = 3

iris = datasets.load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names

preprocessed_data = data.copy()

DIMENSIONAL = data.shape[1]
SIZE = data.shape[0]

As = np.array([np.identity(DIMENSIONAL), np.identity(DIMENSIONAL), np.identity(DIMENSIONAL)])

EPSILON = 0.0005


# In[ ]:


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


# In[ ]:


preprocessed_data = encode_on_hypercube(preprocessed_data)
np.random.shuffle(preprocessed_data)

cluster_column = np.zeros((data.shape[0], 1))
preprocessed_data = np.append(preprocessed_data, cluster_column, axis=1)
preprocessed_data


# In[ ]:


centroids = np.random.uniform(low=-1, high=1, size=(CLUSTERS_COUNT, DIMENSIONAL))
centroids


# In[ ]:


def get_distance(row, center, A):
    diff = row - center
    result = np.dot(np.dot(diff.reshape((1, DIMENSIONAL)), A), diff.reshape((DIMENSIONAL, 1)) ).item()

    return result


# In[ ]:


def get_owners(data, centers, As):
    owners = np.zeros((SIZE, CLUSTERS_COUNT))
    for index, row in enumerate(data):
        distances = np.array([1 / get_distance(row[:DIMENSIONAL], center, np.linalg.inv(As[n_index])) 
                              for n_index, center in enumerate(centers)])
        sum_distances = np.sum(distances)

        owners[index] = np.array([
            dist / sum_distances for dist in distances
        ])
    
    assert owners.shape == (SIZE, CLUSTERS_COUNT)

    return owners


# In[ ]:


def get_centroids(data, owners):
    sqrt_owners = np.power(owners, 2)
    
    sum_center = np.apply_along_axis(np.sum, 0, sqrt_owners)
    
    sum_data = np.zeros((CLUSTERS_COUNT, DIMENSIONAL))
    for index, row in enumerate(data):
        sum_data += np.dot(
            sqrt_owners[index].reshape((CLUSTERS_COUNT, 1)),
            row[:DIMENSIONAL].reshape((1, DIMENSIONAL))
        )
    
    result = sum_data / sum_center.reshape((CLUSTERS_COUNT, 1))

    assert result.shape == (CLUSTERS_COUNT, DIMENSIONAL)
    return result


# In[ ]:


def get_new_Fs(data, centroids, owners):
    Fs = np.zeros((CLUSTERS_COUNT, DIMENSIONAL, DIMENSIONAL))
    
    sqrt_owners = np.power(owners, 2)
    
    for index, center in enumerate(centroids):

        for n_index, row in enumerate(data):
            diff = row[:DIMENSIONAL] - center   

            Fs[index] += np.dot(
                (sqrt_owners[n_index][index] * diff).reshape((DIMENSIONAL, 1)),
                diff.reshape((1, DIMENSIONAL))
            )
        Fs[index] /= np.sum(sqrt_owners.T[index])

    return Fs


# In[ ]:


def get_new_As(data, centroids, owners):
    Fs = get_new_Fs(data, centroids, owners)
    
    As = np.zeros((CLUSTERS_COUNT, DIMENSIONAL, DIMENSIONAL))
    for index, F in enumerate(Fs):
        As[index] = np.sqrt(np.sqrt(1 / np.linalg.det(F))) * F
    
    return As


# In[ ]:


def get_answer(data, centroids, As):
    owners = get_owners(data, centroids, As)
    new_centroids = get_centroids(data, owners)
    
    As = get_new_As(data, new_centroids, owners)
        
    while any([get_distance(new_centroids[index], centroids[index], As[index]) > EPSILON
               for index in range(centroids.shape[0])
              ]):
        owners = get_owners(data, new_centroids, As)

        centroids = new_centroids
        new_centroids = get_centroids(data, owners)
        As = get_new_As(data, new_centroids, owners)

    return data, new_centroids, As, owners


# In[ ]:


result_data, new_centroids, new_As, new_owners = get_answer(preprocessed_data, centroids, As)


# In[ ]:


new_centroids


# In[ ]:


new_owners


# In[ ]:




