# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import evall
#train_data = np.array(pd.read_csv('data/train_data.csv', usecols = ['userid','itemid','re_classid','re_brand']))
## load the test data
#test_data_csv = pd.read_csv('data/test_data.csv', usecols = ['itemid', 're_classid', 're_brand'])
## drop duplicates
#test_data = np.array(test_data_csv.drop_duplicates(subset='itemid', keep='first', inplace=False))

train_data = np.load("np_train_data.npy")
np.random.shuffle(train_data)
#print(train_data.shape)
counter_data = np.load("np_train_counter_examples.npy")
counter_size = counter_data.shape[0]
np.random.shuffle(counter_data)

test_data = np.load("np_test_data.npy")

test_data = test_data[:, 1:4]

test_data = np.unique(test_data, axis=0)

ui_matrix = np.load("ui_matrix.npy")
ucb_matrix = np.load("ucb_matrix.npy")
#user_emb_matrix = np.load("cf_user_emb.npy")



def get_popular_item(top_k):
     sale_volume = np.sum(ui_matrix, axis = 0)
     popular_item_list = np.argsort(-sale_volume)[0: top_k]
     sub_matrix = ui_matrix[:, popular_item_list]
     return sub_matrix
 
#user_emb_matrix = get_popular_item(1047)
user_emb_matrix = ucb_matrix

def get_batchdata(start_index, end_index): 
    '''get train samples'''
    batch_data = train_data[start_index: end_index]
    user_batch = [x[0] for x in batch_data]
    item_batch = [x[1] for x in batch_data]
    class_batch = [x[2]for x in batch_data]
    brand_batch = [x[3] for x in batch_data]
    user_emb_batch = user_emb_matrix[user_batch]
    return item_batch, brand_batch, class_batch, user_emb_batch

def get_counter_batch(start_index, end_index):
    '''get counter examples'''
    start_index = start_index % counter_size
    end_index = end_index % counter_size
    counter_batch_data = counter_data[start_index: end_index]
    counter_user_batch = counter_batch_data[:, 0]
    counter_class_batch = counter_batch_data[:, 1]
    counter_brand_batch = counter_batch_data[:, 2]
    counter_user_emb_batch = user_emb_matrix[counter_user_batch]
    return counter_brand_batch, counter_class_batch, counter_user_emb_batch
    
def get_testdata():
    '''get test samples'''
    test_item_batch = test_data[:, 0]
    test_classid_batch = test_data[:, 1]
    test_brand_batch = test_data[:, 2]
    return test_item_batch, test_brand_batch, test_classid_batch


#user_sqrt = np.sqrt(np.sum(np.multiply(user_emb_matrix, user_emb_matrix), axis=1))
def get_intersection_similar_user(G_user, k):
    user_emb_matrixT = np.transpose(user_emb_matrix)
    A = np.matmul(G_user, user_emb_matrixT)      
    intersection_rank_matrix = np.argsort(-A)  
    return intersection_rank_matrix[:, 0:k]


def test(test_item_batch, test_G_user):
    
    k_value = 20
    test_BATCH_SIZE = np.size(test_item_batch)
    
    test_intersection_similar_user = get_intersection_similar_user(test_G_user, k_value)
    count = 0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):       
        for test_u in test_userlist:
            
            if ui_matrix[test_u, test_i] == 1:
                count = count + 1            
    p_at_20 = round(count/(test_BATCH_SIZE * k_value), 4)
           
    ans = 0.0
    RS = []
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):  
        r=[]
        for user in test_userlist:
            r.append(ui_matrix[user][test_i])
        RS.append( r)
#    print('MAP @ ',k_value,' is ',  evall.mean_average_precision(RS) )  
    M_at_20 = evall.mean_average_precision(RS)
  
    ans = 0.0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):  
        r=[]
        for user in test_userlist:
            r.append(ui_matrix[user][test_i])
        ans = ans + evall.ndcg_at_k(r, k_value, method=1)
#    print('ndcg @ ',k_value,' is ', ans/test_BATCH_SIZE) 
    G_at_20 = ans/test_BATCH_SIZE
    k_value = 10 
    
    count = 0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):       
        for test_u in test_userlist[:k_value]:
            
            if ui_matrix[test_u, test_i] == 1:
                count = count + 1            
    p_at_10 = round(count/(test_BATCH_SIZE * k_value), 4)
         
    ans = 0.0
    RS = []
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):  
        r=[]
        for user in test_userlist[:k_value]:
            r.append(ui_matrix[user][test_i])
        RS.append( r)
#    print('MAP @ ',k_value,' is ',  evall.mean_average_precision(RS) ) 
    M_at_10 = evall.mean_average_precision(RS)
    

    ans = 0.0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):  
        r=[]
        for user in test_userlist[:k_value]:
            r.append(ui_matrix[user][test_i])
        ans = ans + evall.ndcg_at_k(r, k_value, method=1)
#    print('ndcg @ ',k_value,' is ', ans/test_BATCH_SIZE) 
    G_at_10 = ans/test_BATCH_SIZE
  

    return p_at_10,p_at_20,M_at_10,M_at_20,G_at_10,G_at_20
