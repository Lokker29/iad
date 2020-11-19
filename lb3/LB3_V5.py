#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('./data/data.csv')
rows_count = data.shape[0]
data['A'] = pd.Series([1.0] * rows_count)
data.head()


# In[12]:


X0, X1, X2, X3, Y = 'x0', 'x1', 'x2', 'x3', 'Y'
X_DATA_LENGTH = 4

configs = {
    1: {
        X0: {'column_name': 'A', 'start_index': 0},
        X1: {'column_name': 'F', 'start_index': 0},
        X2: {'column_name': 'M', 'start_index': 1},
        X3: {'column_name': 'V', 'start_index': 0},
        Y:  {'column_name': 'C', 'start_index': 1},
    },
    2: {
        X0: {'column_name': 'A', 'start_index': 0},
        X1: {'column_name': 'M', 'start_index': 0},
        X2: {'column_name': 'F', 'start_index': 0},
        X3: {'column_name': 'C', 'start_index': 1},
        Y:  {'column_name': 'V', 'start_index': 1},
    },
    3: {
        X0: {'column_name': 'A', 'start_index': 0},
        X1: {'column_name': 'C', 'start_index': 0},
        X2: {'column_name': 'V', 'start_index': 1},
        X3: {'column_name': 'F', 'start_index': 1},
        Y:  {'column_name': 'M', 'start_index': 1},
    },
    4: {
        X0: {'column_name': 'A', 'start_index': 0},
        X1: {'column_name': 'M', 'start_index': 0},
        X2: {'column_name': 'V', 'start_index': 1},
        X3: {'column_name': 'C', 'start_index': 0},
        Y:  {'column_name': 'F', 'start_index': 1},
    },
    5: {
        X0: {'column_name': 'A', 'start_index': 0},
        X1: {'column_name': 'M', 'start_index': 0},
        X2: {'column_name': 'F', 'start_index': 1},
        X3: {'column_name': 'V', 'start_index': 1},
        Y:  {'column_name': 'C', 'start_index': 1},
    },
    6: {
        X0: {'column_name': 'A', 'start_index': 0},
        X1: {'column_name': 'V', 'start_index': 0},
        X2: {'column_name': 'C', 'start_index': 1},
        X3: {'column_name': 'M', 'start_index': 1},
        Y:  {'column_name': 'F', 'start_index': 1},
    },
}
SELECTED_VARIANT = 1


# In[10]:


def calculate_coefficients(data, config):
    loops_count = data.shape[0] - 1
    
    first_component = np.zeros((loops_count, X_DATA_LENGTH))
    second_component = np.zeros(loops_count)
    
    for index in range(loops_count):
        first_component[index] = np.array([data[config[X0]['column_name']][config[X0]['start_index'] + index],
                                           data[config[X1]['column_name']][config[X1]['start_index'] + index],
                                           data[config[X2]['column_name']][config[X2]['start_index'] + index],
                                           data[config[X3]['column_name']][config[X3]['start_index'] + index]])

        second_component[index] = data[config[Y]['column_name']][config[Y]['start_index'] + index]
    
    second_component = first_component.T.dot(second_component)
    first_component = first_component.T.dot(first_component)
    
    assert first_component.shape == (X_DATA_LENGTH, X_DATA_LENGTH)
    assert second_component.shape == (X_DATA_LENGTH,)
    
    reversed_first_component = np.linalg.inv(first_component)
    
    result = reversed_first_component.dot(second_component)
    assert result.shape == (X_DATA_LENGTH,)
    
    return result


# In[11]:


calculate_coefficients(data, configs[SELECTED_VARIANT])


# In[ ]:




