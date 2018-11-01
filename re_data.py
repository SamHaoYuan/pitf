# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 16:23:25 2018

@author: Sam
"""

import pandas as pd

data_path = 'data/movielens/all_id_core1'

data = pd.read_csv(data_path, delimiter='\t', names=['u','i','tag','time'])

# 过滤掉data中只出现次数少于3次的u

user_group = data.groupby('u')
for key, id_ in user_group:
    if len(id_) <3:
        data = data[data.u !=key]

# 重新对user,item,tag编号
user = data.u.drop_duplicates()
user_id = dict()
for u_id in range(len(user)):
    user_id[user.values[u_id]] = u_id
users = data.u.values
for count in range(len(users)):
    u_ = users[count]
    n_id = user_id[u_]
    users[count] = n_id
data.u= users

item = data.i.drop_duplicates()
item_id = dict()
for i_id in range(len(item)):
    item_id[item.values[i_id]] = i_id
items = data.i.values
for count in range(len(items)):
    i_ = items[count]
    n_id = item_id[i_]
    items[count] = n_id
data.i= items

tag = data.tag.drop_duplicates()
tag_id = dict()
for t_id in range(len(tag)):
    tag_id[tag.values[t_id]] = t_id
tags = data.tag.values
for count in range(len(tags)):
    t_ = tags[count]
    n_id = tag_id[t_]
    tags[count] = n_id
data.tag= tags

# 输出保存
# data.to_csv('data/movielens/user_id_core3', sep='\t', header=False, index=False)

# 训练集与测试集划分（留一法）