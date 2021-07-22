# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 17:00:31 2021

@author: Declan
"""

from orion.sources import S3Source
from orion.sources.io import read_csv, write_csv
import pandas as pd
import numpy as np
from scipy import sparse

#source = RedshiftSource(query='SELECT * FROM publish.inventory_lookup WHERE productid IS NULL')
#df_stock = source.read_csv()

with S3Source(key="masters/uploads/customers/1560425511130_Peak_customers.csv", bucket="kilimanjaro-prod-datalake") as s3:
    df_cust = read_csv(s3)

df_trans = read_csv(S3Source(key="masters/uploads/transactions/1560426066385_Peak_transactions.csv", bucket="kilimanjaro-prod-datalake"))
df_prod = read_csv(S3Source(key="masters/uploads/product/1560425499995_Peak_product.csv", bucket="kilimanjaro-prod-datalake"))


# 30pc of transactions have keys that are not in df_cust
pd.Series(df_trans.customerkey.unique()).isin(df_cust.customerkey).sum()/len(df_trans.customerkey.unique())
# 45pc of item numbers not in df_prod
pd.Series(df_trans.itemnumber.unique()).isin(df_prod.itemnumber).sum()/len(df_trans.itemnumber.unique())

df_cust.drop(df_cust[df_cust.gender=='unknown'].index, axis=0, inplace=True)

# Fix item number
df_prod.itemnumber = pd.to_numeric(df_prod.itemnumber, errors='coerce') 

# Remove keys not in customer and product tables
df_trans = df_trans.iloc[df_trans.customerkey[df_trans.customerkey.isin(df_cust.customerkey)].index,:]
df_trans.reset_index(inplace=True, drop=True)
df_trans = df_trans.iloc[df_trans.itemnumber[df_trans.itemnumber.isin(df_prod.itemnumber)].index,:]

# Reset index before changing keys to integers
df_trans.reset_index(inplace=True, drop=True)
df_prod.reset_index(inplace=True, drop=True)


# Drop nas
df_prod.dropna(inplace=True)
df_prod.itemnumber.unique().shape

print(df_cust.customerkey.unique().shape)
df_cust.dropna(inplace=True)
print(df_cust.customerkey.unique().shape)

print(df_trans.customerkey.unique().shape)
df_trans.dropna(inplace=True)
print(df_trans.customerkey.unique().shape)

# Drop redundant cols. Fabric all unknown, item colour less info than colourvalue
df_prod.drop(["itemstylecode", "itemquarter", "itemcolour", "fabric"], axis=1, inplace=True)

df_prod.reset_index(inplace=True, drop=True)

df_prod.iloc[:,1:].duplicated().sum()

dups = df_prod.loc[df_prod.iloc[:,1:].duplicated(),:]

df_prod.drop(dups.index, inplace=True)

# Drop columns
df_trans.drop(["salestransactionkey", "salesordernumber","discountpercent", "grosssales",
              'orderdate', 'ordertime', 'shippingdate','grossprofit'], axis=1, inplace=True)

df = df_trans.merge(df_cust, on='customerkey', how = "outer")
df = df.merge(df_prod, on = "itemnumber")

# Rename and drop columns, nas
df.rename(columns={"customerkey":"user_id","itemnumber":"item_id","shipcountry":"country", "brandcode":"brand_id",
                  "colourvalue":"colour", }, inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True, drop=True)

df_trans.rename(columns={"customerkey":"user_id","itemnumber":"item_id"}, inplace=True)
df_cust.rename(columns={"customerkey":"user_id"}, inplace=True)
df_prod.rename(columns={"itemnumber":"item_id"}, inplace=True)

df.shape

df.drop_duplicates(inplace=True)

df.shape,df.user_id.unique().shape, df.item_id.unique().shape

### Take subset of data


df_sub = df.sample(frac = 0.10)

cats = df_sub.select_dtypes('object').columns
keys = df_sub[cats].apply(lambda x: x.factorize()[1])
df_sub[cats] = df_sub[cats].apply(lambda x: x.factorize()[0])

df = df_sub

df.drop(['unitssold', 'pricetype'], axis=1, inplace=True)

len(df.item_id.unique())

user_num = df.user_id.unique().shape[0]
item_num = df.item_id.unique().shape[0]

user_num, item_num

#Renumber item ids
ints = [i for i in range(item_num)]
item_keys = [i for i in df.item_id.unique()]
item_key_dict = {i:j for i,j in zip( item_keys, ints)}

# Change keys to integers in dataframes
df.item_id = [item_key_dict[df.item_id.iloc[i]] for i in range(len(df))]

user_cols = ['user_id', 'country','postcode','loyaltyaccount','gender']
users = df[user_cols]
items = df[['item_id',"brand_id","category","colour", "divisioncode","itemcategorycode","itemfamilycode","itemseason","productgroup"]]

interactions = df[['user_id', 'item_id']]

users.drop_duplicates(inplace=True)
items.drop_duplicates(inplace=True)

users.to_csv("users")
items.to_csv("items")
interactions.to_csv("interactions")

# Train-test split
train = df.sample(frac = 0.8)
ind = df.index.isin(train.index)
test = df[~ind]

train.head()

np.save("new_fa_train_data",train.to_numpy())
np.save("new_fa_test_data",test.to_numpy())

### Make dfs fr Orion

## Matrices

USER_NUM = len(df.user_id.unique())
country_num = len(df.country.unique())
postcode_num = len(df.postcode.unique())	
item_id_num = len(df.item_id.unique())
pricetype_num = len(df.pricetype.unique())
loyaltyaccount_num = len(df.loyaltyaccount.unique())
gender_num = len(df.gender.unique())
brand_id_num = len(df.brand_id.unique())
category_num = len(df.category.unique())
colour_num = len(df.colour.unique())
divisioncode_num = len(df.divisioncode.unique())
itemcategorycode_num = len(df.itemcategorycode.unique())
itemfamilycode_num = len(df.itemfamilycode.unique())
itemseason_num = len(df.itemseason.unique())
productgroup_num = len(df.productgroup.unique())

print(USER_NUM,
country_num,
postcode_num,
item_id_num,
pricetype_num,
loyaltyaccount_num,
gender_num,
brand_id_num,
category_num,
colour_num,
divisioncode_num,
itemcategorycode_num,
itemfamilycode_num,
itemseason_num,
productgroup_num)

### Item Attribute matrix

features = [   brand_id_num, category_num, colour_num, divisioncode_num, 
            itemcategorycode_num, itemfamilycode_num, itemseason_num, productgroup_num]
names = [  'brand_id',
         'category', 'colour', 'divisioncode', 'itemcategorycode',
         'itemfamilycode', 'itemseason', 'productgroup']
dic = dict(zip(names,features))

matrices = []
for i in dic.keys():
    mat = np.zeros((item_num, dic[i]), dtype = np.int)    
    pair = df.loc[:,["item_id", i]]
    M = np.unique(pair, axis=0)
    for m in M:
        mat[m[0], m[1]] = 1
    print(mat[1])
    matrices.append(mat)

[(matrices[i]).shape for i in range(len(matrices))]

ia_matrix = np.concatenate((matrices), axis=1)

np.save('new_fa_ia_matrix.npy', ia_matrix)

matrices[0].shape

### User- attribute matrix

matrices = []
for i in dic.keys():
    mat = np.zeros((USER_NUM, dic[i]), dtype = np.int)
    pair = df.loc[:,["user_id", i]]
    M = np.unique(pair, axis=0)
    for m in M:
        mat[m[0], m[1]] = 1
    print(mat[1])
    matrices.append(mat)

fa_matrix = np.concatenate((matrices), axis=1)

fa_matrix.shape[1]

[(matrices[i]).shape for i in range(len(matrices))]

print(np.unique(ia_matrix, axis=0).shape, np.unique(fa_matrix, axis=0).shape)

fa_matrix.shape

np.save('new_fa_ua_matrix', fa_matrix)

### User - attribute Matrices

features = [   country_num, postcode_num, gender_num, loyaltyaccount_num]
names = [ "country", "postcode", "gender", "loyaltyaccount" ]
dic = dict(zip(names,features))

matrices = []
for i in dic.keys():
    mat = np.zeros((USER_NUM, dic[i]), dtype = np.int)
    pair = df.loc[:,["user_id", i]]
    M = np.unique(pair, axis=0)
    for m in M:
        mat[m[0], m[1]] = 1
    print(mat[1])
    matrices.append(mat)

user_atts_matrix = np.concatenate((matrices), axis=1)

np.save("user_attributes", user_atts_matrix)

### User - Item Matrix

mat = np.zeros((user_num, item_num), dtype = np.int32)
pair = df.loc[:,["user_id", "item_id"]]
M = np.unique(pair, axis=0)
for m in M:
    mat[m[0], m[1]] = 1
print(mat[1])

np.save("fa_ui_matrix", mat)