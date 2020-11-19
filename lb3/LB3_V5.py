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
SELECTED_VARIANT = 4


# In[15]:


def calculate_coefficients(data, config):
    first_component = np.zeros((X_DATA_LENGTH, X_DATA_LENGTH))
    second_component = np.zeros(X_DATA_LENGTH)
    
    for index in range(data.shape[0] - 1):
        x_row = np.array([data[config[X0]['column_name']][config[X0]['start_index'] + index],
                          data[config[X1]['column_name']][config[X1]['start_index'] + index],
                          data[config[X2]['column_name']][config[X2]['start_index'] + index],
                          data[config[X3]['column_name']][config[X3]['start_index'] + index]])
        x_row_as_2d = x_row.reshape((X_DATA_LENGTH, 1))
        first_component += x_row_as_2d.dot(x_row_as_2d.T)
        second_component += x_row * data[config[Y]['column_name']][config[Y]['start_index'] + index]
    
    assert first_component.shape == (X_DATA_LENGTH, X_DATA_LENGTH)
    assert second_component.shape == (X_DATA_LENGTH,)
    
    reversed_first_component = np.linalg.inv(first_component)
    second_component_as_2d = second_component.reshape((X_DATA_LENGTH, 1))
    
    result = reversed_first_component.dot(second_component_as_2d)
    assert result.shape == (X_DATA_LENGTH, 1)
    
    return result.reshape(X_DATA_LENGTH)


# In[14]:


calculate_coefficients(data, configs[SELECTED_VARIANT])


# In[ ]:




