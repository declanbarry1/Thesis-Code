# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:33:25 2021

@author: Declan
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import evall
from scipy import sparse
train_data = np.load("new_fa_train_data.npy")
np.random.shuffle(train_data)
counter_data = np.load("train_counter_examples.npy")
counter_size = counter_data.shape[0]
np.random.shuffle(counter_data)

test_data = np.load("new_fa_test_data.npy")

#test_data = test_data[:, 1:4]

test_data = np.unique(test_data, axis=0)

ia_matrix = np.load("new_fa_ia_matrix.npy")
user_emb_matrix = np.load("new_fa_ua_matrix.npy")
item_emb_matrix = np.load("item_attributes.npy")

ui_matrix = np.load("fa_ui_matrix.npy")
def get_popular_item(top_k):
     sale_volume = np.sum(ia_matrix, axis = 0)
     popular_item_list = np.argsort(-sale_volume)[0: top_k]
     sub_matrix = ia_matrix[:, popular_item_list]
     return sub_matrix
 
def get_batchdata(start_index, end_index): 
    '''get train samples'''
    batch_data = train_data[start_index: end_index]
    user = [x[0] for x in batch_data]
    country = [x[1] for x in batch_data]
    postcode = [x[2] for x in batch_data]
    item = [x[3] for x in batch_data]
    #pricetype = [x[4] for x in batch_data]
    loyal = [x[4] for x in batch_data]
    gender = [x[5] for x in batch_data]
    brand_id = [x[6] for x in batch_data]
    category = [x[7] for x in batch_data]
    colour = [x[8] for x in batch_data]
    divisioncode = [x[9] for x in batch_data]
    itemcategorycode = [x[10] for x in batch_data]
    itemfamilycode = [x[11] for x in batch_data]
    itemseason = [x[12] for x in batch_data]
    productgroup = [x[13] for x in batch_data]
    user_emb_batch = user_emb_matrix[user]
    item_emb_batch = item_emb_matrix[item]
    return country, postcode, loyal, gender,  brand_id, category, colour, divisioncode, itemcategorycode, itemfamilycode, \
        itemseason, productgroup, user_emb_batch, item_emb_batch

def get_counter_batch(start_index, end_index):
    '''get counter examples'''
    start_index = start_index % counter_size
    end_index = end_index % counter_size
    batch_data = counter_data[start_index: end_index]
    user = [x[0] for x in batch_data]
    brand_id = [x[1] for x in batch_data]
    category = [x[2] for x in batch_data]
    colour = [x[3] for x in batch_data]
    divisioncode = [x[4] for x in batch_data]
    itemcategorycode = [x[5] for x in batch_data]
    itemfamilycode = [x[6] for x in batch_data]
    itemseason = [x[7] for x in batch_data]
    productgroup = [x[8] for x in batch_data]
    user_emb_batch = user_emb_matrix[user]
    return  brand_id, category, colour, divisioncode, itemcategorycode, itemfamilycode, \
        itemseason, productgroup, user_emb_batch

    
def get_testdata(start, end):
    '''get test samples'''
    x = test_data[start:end]
    user = x[:,0]
    country = x[:,1]
    postcode = x[:,2]
    item = x[:,3] 
    #pricetype = x[4] 
    loyal = x[:,4] 
    gender = x[:,5]
    brand_id = x[:,6] 
    category = x[:,7]
    colour = x[:,8] 
    divisioncode = x[:,9] 
    itemcategorycode = x[:,10] 
    itemfamilycode = x[:,11] 
    itemseason = x[:,12] 
    productgroup = x[:,13] 
    return country, postcode, loyal, gender,  brand_id, category, colour, divisioncode, itemcategorycode, itemfamilycode, \
        itemseason, productgroup, item , user


#user_sqrt = np.sqrt(np.sum(np.multiply(user_emb_matrix, user_emb_matrix), axis=1))
def get_intersection_similar_user(G_user, k):
    user_emb_matrixT = np.transpose(user_emb_matrix)
    intersection_rank_matrix = np.argsort( -np.matmul(G_user, user_emb_matrixT) )     
    return intersection_rank_matrix[:, 0:k]

def get_intersection_similar_item(G_item, k):
    item_emb_matrixT = np.transpose(item_emb_matrix)
    intersection_rank_matrix = np.argsort( -np.matmul(G_item, item_emb_matrixT) )     
    return intersection_rank_matrix[:, 0:k]


def test(item_batch, test_G_user):
    k = 20
    test_BATCH_SIZE = np.size(item_batch)
    intersection_similar_user = get_intersection_similar_user(G_user, k)
    count = 0
    for i, user_list in zip(item_batch, intersection_similar_user):       
        for u in user_list:
            if ui_matrix[u,i] == 1:
                count = count + 1            
    p_at_20 = round(count/(test_BATCH_SIZE * k_value), 4)

    RS = []
    ans = 0.0
    for i, user_list in zip(item_batch, intersection_similar_user):           
        r=[]
        for user in user_list:
             r.append(ui_matrix[user][test_i])
        ans = ans + evall.ndcg_at_k(r, k, method=1)
        RS.append(r)
    G_at_20 = ans/test_BATCH_SIZE
    M_at_20 = evall.mean_average_precision(RS)
    
    k = 10
    intersection_similar_user = get_fa_intersection_similar_user(G_user, k)
    count = 0
    for i, user_list in zip(item_batch, intersection_similar_user):       
        for u in user_list:
            if np.sum(fa_ia_matrix[i] + fa_user_emb_matrix[u] == 2) ==8:
                count = count + 1            
    p_at_10 = round(count/(test_BATCH_SIZE * k), 4)

    RS = []
    ans = 0.0
    for i, user_list in zip(item_batch, intersection_similar_user):           
        r=[]
        for user in user_list:
            if np.sum(fa_ia_matrix[i] + fa_user_emb_matrix[user] == 2) ==8:
                r.append(1)
            else:
                r.append(0)
        ans = ans + evall.ndcg_at_k(r, k, method=1)
        RS.append(r)
    G_at_10 = ans/test_BATCH_SIZE
    M_at_10 = evall.mean_average_precision(RS)
    return p_at_10,p_at_20,M_at_10,M_at_20,G_at_10,G_at_20



def test_big(item_batch, test_G_user):
    
    k = 20
    test_BATCH_SIZE = np.size(item_batch)
    intersection_similar_user = get_intersection_similar_user(G_user, k)
    count = 0
    for i, user_list in zip(item_batch, intersection_similar_user):       
        for u in user_list:
            if np.sum(fa_ia_matrix[i] + fa_user_emb_matrix[u] == 2) >=9:
                count = count + 1            
    p_at_20 = round(count/(test_BATCH_SIZE * k), 4)

    RS = []
    ans = 0.0
    for i, user_list in zip(item_batch, intersection_similar_user):           
        r=[]
        for user in user_list:
            if np.sum(fa_ia_matrix[i] + fa_user_emb_matrix[user] == 2) >=9:
                r.append(1)
            else:
                r.append(0)
        ans = ans + evall.ndcg_at_k(r, k, method=1)
        RS.append(r)
    G_at_20 = ans/test_BATCH_SIZE
    M_at_20 = evall.mean_average_precision(RS)
    
    k = 10
    intersection_similar_user = get_fa_intersection_similar_user(G_user, k)
    count = 0
    for i, user_list in zip(item_batch, intersection_similar_user):       
        for u in user_list:
            if np.sum(fa_ia_matrix[i] + fa_user_emb_matrix[u] == 2) ==8:
                count = count + 1            
    p_at_10 = round(count/(test_BATCH_SIZE * k), 4)

    RS = []
    ans = 0.0
    for i, user_list in zip(item_batch, intersection_similar_user):           
        r=[]
        for user in user_list:
            if np.sum(fa_ia_matrix[i] + fa_user_emb_matrix[user] == 2) ==8:
                r.append(1)
            else:
                r.append(0)
        ans = ans + evall.ndcg_at_k(r, k, method=1)
        RS.append(r)
    G_at_10 = ans/test_BATCH_SIZE
    M_at_10 = evall.mean_average_precision(RS)
    return p_at_10,p_at_20,M_at_10,M_at_20,G_at_10,G_at_20
