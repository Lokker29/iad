#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linprog


# In[2]:


costs = [30, 70, 30, 3, 10, 30]

data = {
    'Молоко (л.)': [720, 344, 18, 0.2, costs[0], 10, 24, 6],
    'Мясо (кг.)': [107, 1460, 151, 10.1, costs[1], 20, 27, 1],
    'Яйца (дес.)': [7080, 1040, 78, 13.2, costs[2], 120, 0, 0.25],
    'Хлеб (100 гр.)': [0, 75, 2.5, 0.75, costs[3], 0, 15, 10],
    'Овощи (100 гр.)': [134, 17.4, 0.2, 0.15, costs[4], 0, 1.1, 10],
    'Апельс. сок (литр)': [1000, 240, 4, 1.2, costs[5], 0, 52, 4],
}

min_values = [5000, 2500, 63, 12.5]

index = [
    'Витамин А',
    'Калории',
    'Протеин',
    'Железо',
    'Стоимость',
    'Холестерин',
    'Углеводы',
    'Макс. потребление'
]

df = pd.DataFrame(data, index=index)
df


# In[3]:


# целевые функции
price = df.loc[index[4]].values
cholesterol = df.loc[index[5]].values
carbonates = df.loc[index[6]].values

# общие ограничения
general_bounds = np.array([
    df.loc[index[0]].values,
    df.loc[index[1]].values,
    df.loc[index[2]].values,
    df.loc[index[3]].values,
]) * -1
# правая часть общих ограничений
free_bounds = np.array(min_values) * -1

# ограничения переменных
vars_bounds = list(zip([0] * df.loc[index[-1]].size, df.loc[index[-1]].values))


# In[4]:


method = "simplex"


# In[5]:


def calculate_simplex(c, A_ub, b_ub, bounds, method):
    f = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)

    print(f)
    coefs = np.array(f['x'])

    print()
    print('цена:', np.dot(coefs, price))
    print('холестерин:', np.dot(coefs, cholesterol))
    print('углеводы:', np.dot(coefs, carbonates))
    return f


# In[6]:


f1 = calculate_simplex(c=price, A_ub=general_bounds, b_ub=free_bounds, bounds=vars_bounds, method=method)
coefs1 = np.array(f1['x'])


# In[7]:


f2 = calculate_simplex(c=cholesterol, A_ub=general_bounds, b_ub=free_bounds, bounds=vars_bounds, method=method)
coefs2 = np.array(f2['x'])


# In[8]:


f3 = calculate_simplex(c=carbonates, A_ub=general_bounds, b_ub=free_bounds, bounds=vars_bounds, method=method)
coefs3 = np.array(f3['x'])


# In[9]:


results = np.array([
    np.array([np.dot(coefs1, price), np.dot(coefs1, cholesterol), np.dot(coefs1, carbonates)]),
    np.array([np.dot(coefs2, price), np.dot(coefs2, cholesterol), np.dot(coefs2, carbonates)]),
    np.array([np.dot(coefs3, price), np.dot(coefs3, cholesterol), np.dot(coefs3, carbonates)]),
])

objectives = [price, cholesterol, carbonates]
alphas = [0.3, 0.2, 0.5]

print(results)


# In[10]:


final_criteria = np.array([
    np.array(alphas[index] * objectives[index] / arr.max()) for index, arr in enumerate(results.T)
])

final = final_criteria.sum(axis=0)


# In[11]:


f4 = calculate_simplex(c=final, A_ub=general_bounds, b_ub=free_bounds, bounds=vars_bounds, method=method)
coefs4 = np.array(f4['x'])


# In[12]:


x2 = coefs4[1]
x4 = coefs4[3]
x5 = coefs4[4]
x6 = coefs4[5]

F1 = price
F2 = carbonates

scatter_x = []
scatter_y = []
diff = 0.05

x1 = 0

# while x1 <= 1:
while x1 <= df.loc[index[-1]].values[0]:
    x3 = 0
    while x3 <= df.loc[index[-1]].values[2]:
        x = [x1, x2, x3, x4, x5, x6]
        
        scatter_x.append(np.dot(F1, x))
        scatter_y.append(np.dot(F2, x))
        
        x3 += diff
    x1 += diff

plt.scatter(scatter_x, scatter_y)

# plt.scatter(
#     [np.dot(F1, coefs1), np.dot(F1, coefs2), np.dot(F1, coefs3), np.dot(F1, coefs4)],
#     [np.dot(F2, coefs1), np.dot(F2, coefs2), np.dot(F2, coefs3), np.dot(F2, coefs4)], 
#     c=['green', 'yellow', 'purple', 'red'])
plt.scatter(
    [np.dot(F1, coefs4)],
    [np.dot(F2, coefs4)], 
    c=['red'])
plt.show()


# In[ ]:




