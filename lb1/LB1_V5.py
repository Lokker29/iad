#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from functools import reduce


# In[ ]:


data = pd.read_csv('data/diabetes.csv')


# In[ ]:


OUTCOME = 'Outcome'

target = data[OUTCOME]
data_without_target = data.loc[:, data.columns != OUTCOME]


# In[ ]:


data


# In[ ]:


data.describe()


# In[ ]:


def avg_utmost_values(column):
    """
    Метод для определения полусуммы крайних наблюдений
    
    :param column: Series
    return float
    """
    return (column.min() + column.max()) / 2


# In[ ]:


def avg_absolute_deviations(column):
    """
    Метод для определения среднего модуля отклонений
    
    :param column: Series instance
    return float
    """
    median = column.median()
    return np.sum(np.absolute(column.apply(lambda value: value - median) / column.size))


# In[ ]:


def scope(column):
    """
    Метод для определения размаха
    
    :param column: Series instance
    return float
    """
    return column.max() - column.min()


# In[ ]:


def normilize(data):
    """
    Метод для нормализации и центрирования данных
    
    :param data: DataFrame instance
    return DataFrame instance
    """
    def encode_by_column(column):
        mean, std = column.mean(), column.std()
        return column.apply(lambda x: (x - mean) / std)
    
    return data.apply(encode_by_column)


# In[ ]:


def encode_on_hypercube(data):
    """
    Метод для кодирования на гиперкуб
    
    :param data: DataFrame instance
    return DataFrame instance
    """
    def encode_by_column(column):
        minimum, maximum = column.min(), column.max()
        return column.apply(lambda x: 2 * (x - minimum) / (maximum - minimum) - 1)
    
    return data.apply(encode_by_column)


# In[ ]:


def encode_on_hyperball(data):
    """
    Метод для кодирования на гипершар
    
    :param data: DataFrame instance
    return DataFrame instance
    """
    def encode_by_column(column):
        minimum, maximum = column.min(), column.max()
        return column.apply(lambda x: (x - minimum) / (maximum - minimum))
    
    return data.apply(encode_by_column)


# In[ ]:


def recurrent_relationship_of_mean(data):
    """
    Метод для рекуррентного соотношения среднего значения
    
    :param data: DataFrame instance
    return Series instance
    """
    rows = data.iterrows()
    _, mean = next(rows)
    number = 1
    
    for _, row in rows:
        number += 1
        mean += (row - mean) / number
    
    return mean


# In[ ]:


def recurrent_relationship_of_median(data):
    """
    Метод для рекуррентного соотношения медианы
    
    :param data: DataFrame instance
    return Series instance
    """
    rows = data.iterrows()
    _, median = next(rows)
    number = 1
    
    for _, row in rows:
        number += 1
        median += np.sign(row - median) / number
    
    return median


# In[ ]:


def print_info_about_column(label, column):
    """
    Метод для отображение базовой информации о конкретном параметре
    
    :param label: str, name of the column
    :param column: Series instance
    """
    print(f'Первичный анализ параметра {label}:')
    print(f'\tМат. ожидание: {column.mean()}')
    print(f'\tМедиана: {column.median()}')
    print(f'\tПолусумма крайних наблюдений: {avg_utmost_values(column)}')
    print(f'\tСреднеквадратическое отклонение: {column.std()}')
    print(f'\tСредний модуль отклонений: {avg_absolute_deviations(column)}')
    print(f'\tРазмах: {scope(column)}')
    print(f'\tДисперсия: {column.var()}')
    
    print(f'\tМинимальное значение: {column.min()}')
    print(f'\tМаксимальное значение: {column.max()}')


# In[ ]:


def print_info_about_data(data):
    """
    Метод для отображения базовой информации по каждой колонке датасета
    
    :param data: DataFrame instance
    """
    for label, column in data.iteritems():
        print_info_about_column(label, column)
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        
    print('Нормализированный и отцентрированный датасет:')
    print(normilize(data))

    print('------------------------------------')
    print('Закодированный на гипершар датасет:')
    print(encode_on_hyperball(data))

    print('------------------------------------')
    print('Закодированный на гиперкуб датасет:')
    print(encode_on_hypercube(data))
    
    print('------------------------------------')
    print('Рекуррентное соотношение медианы:')
    print(recurrent_relationship_of_median(data))
    
    print('------------------------------------')
    print('Рекуррентное соотношение среднего значения:')
    print(recurrent_relationship_of_mean(data))

    print('####################################')


# In[ ]:


# Отображение информации по всему датасету
print(f'Count of rows is {data_without_target.shape[0]}')
print_info_about_data(data_without_target)


# In[ ]:


# Отображение информации по датасету, включающему информацию только о людях с подтвержденным диабетом
outcome_true_dataset = data[data[OUTCOME] == 1].loc[:, data.columns != OUTCOME]
print(f'Count of rows is {outcome_true_dataset.shape[0]}')
print_info_about_data(outcome_true_dataset)


# In[ ]:


# Отображение информации по датасету, включающему информацию только о людях с неподтвержденным диабетом
outcome_false_dataset = data[data[OUTCOME] == 0].loc[:, data.columns != OUTCOME]
print(f'Count of rows is {outcome_false_dataset.shape[0]}')
print_info_about_data(outcome_false_dataset)

