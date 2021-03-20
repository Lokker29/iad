#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA


# In[2]:


pd.options.mode.chained_assignment = None


# In[3]:


# irdata = pd.read_csv('./data/iris.data', header=None)
iris = datasets.load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names

preprocessed_data = data.copy()


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


def get_means(data):
    """
    Метод для получения среднего по столбцам
    """
    return np.mean(data, axis=0)


# In[6]:


def center(data):
    means = get_means(data)
    assert means.shape == (4,)
    
    for index, column in enumerate(data.T):
        column -= means[index]
    return data


# In[7]:


def generate_weights(height, width, start, end):
    return np.random.uniform(start, end, (height, width))


# In[8]:


def normilize(weights):
    return np.sum(weights**2) ** (1./2)


# In[9]:


def MyPCA(copied_data):
    data = copied_data.copy()
    records_count, attributes_count = data.shape
    
    weights = generate_weights(attributes_count, attributes_count, -1, 1)

    assert weights.shape == (attributes_count, attributes_count)
    
    results = np.zeros((attributes_count, records_count))
    
    for number, row_weights in enumerate(weights.T):
        row_weights /= normilize(row_weights)
        
        for epoch in range(10 ** (number + 1)):
            
            for record_number, row_result in enumerate(results.T):
                row_result[number] = row_weights.dot(data[record_number])

                row_weights += (row_result[number] 
                                * (data[record_number] - row_result[number] * row_weights) / records_count)
                row_weights /= normilize(row_weights)
        
        data -= results[number].reshape((records_count, 1)).dot(weights[:, number].reshape((1, attributes_count)))
    
    return results.T, weights            


# In[10]:


preprocessed_data = encode_on_hypercube(preprocessed_data)
preprocessed_data


# In[11]:


preprocessed_data = center(preprocessed_data)
preprocessed_data


# In[12]:


start_time = time.time()

components, weights = MyPCA(preprocessed_data)

end_time = time.time()
diff = end_time - start_time
print(diff)


# In[13]:


components.dot(weights.T)


# In[14]:


pca = PCA(n_components=4)
X_r = pca.fit(data).transform(data)

X_r


# In[15]:


plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(components[target == i, 0], components[target == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset (Our PCA method)')
plt.show()


# In[16]:


plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[target == i, 0], X_r[target == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset (Sklearn PCA method)')
plt.show()

