# from google.colab import drive
# drive.mount('/content/gdrive') 
# %cd '/content/gdrive/My Drive/'

# !pip install xgboost==1.7.2
# !pip install category_encoders

# import library
import os
import pandas as pd
import numpy as np
import time
import collections
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

"""# Preprocess public and train data"""

def preprocess(data_dir):
    # declare csv path
    train_alert_date_csv = os.path.join(data_dir, 'train_x_alert_date.csv')
    cus_info_csv = os.path.join(data_dir, 'public_train_x_custinfo_full_hashed.csv')
    y_csv = os.path.join(data_dir, 'train_y_answer.csv')

    ccba_csv = os.path.join(data_dir, 'public_train_x_ccba_full_hashed.csv')
    cdtx_csv = os.path.join(data_dir, 'public_train_x_cdtx0001_full_hashed.csv')
    dp_csv = os.path.join(data_dir, 'public_train_x_dp_full_hashed.csv')
    remit_csv = os.path.join(data_dir, 'public_train_x_remit1_full_hashed.csv')

    public_x_csv = os.path.join(data_dir, 'public_x_alert_date.csv')

    cus_csv = [ccba_csv, cdtx_csv, dp_csv, remit_csv]
    date_col = ['byymm', 'date', 'tx_date', 'trans_date']
    data_use_col = [[1,3,4,5,6,7,8,9],[2,3,4],[1,4,5,6,7,8,9,10,11],[2,3]]
    
    print('Reading csv...')
    # read csv
    df_y = pd.read_csv(y_csv)
    df_cus_info = pd.read_csv(cus_info_csv)
    df_date = pd.read_csv(train_alert_date_csv)
    cus_data = [pd.read_csv(_x) for _x in cus_csv]
    df_public_x = pd.read_csv(public_x_csv)

    # do label encoding
    le = LabelEncoder()
    cus_data[2].debit_credit = le.fit_transform(cus_data[2].debit_credit)
    
    cnts = [0] * 4
    labels = []
    training_data = []

    print('Start processing training data...')
    start = time.time()
    for i in range(df_y.shape[0]):
        # from alert key to get customer information
        cur_data = df_y.iloc[i]
        alert_key, label = cur_data['alert_key'], cur_data['sar_flag']

        cus_info = df_cus_info[df_cus_info['alert_key']==alert_key].iloc[0]
        cus_id = cus_info['cust_id']
        cus_features = cus_info.values[2:]

        date = df_date[df_date['alert_key']==alert_key].iloc[0]['date']


        cnt = 0
        for item, df in enumerate(cus_data):
            cus_additional_info = df[df['cust_id']==cus_id]
            cus_additional_info = cus_additional_info[cus_additional_info[date_col[item]]<=date]

            if cus_additional_info.empty:
                cnts[item] += 1
                len_item = len(data_use_col[item])
                if item == 2:
                    len_item -= 1
                cus_features = np.concatenate((cus_features, [np.nan] * len_item), axis=0)
            else:
                cur_cus_feature = cus_additional_info.loc[cus_additional_info[date_col[item]].idxmax()]
                
                cur_cus_feature = cur_cus_feature.values[data_use_col[item]]
                # 處理 實際金額 = 匯率*金額
                if item == 2:
                    cur_cus_feature = np.concatenate((cur_cus_feature[:2], [cur_cus_feature[2]*cur_cus_feature[3]], cur_cus_feature[4:]), axis=0)
                cus_features = np.concatenate((cus_features, cur_cus_feature), axis=0)
        labels.append(label)
        training_data.append(cus_features)
        print('\r processing data {}/{}'.format(i+1, df_y.shape[0]), end = '')
    print('Processing time: {:.3f} secs'.format(time.time()-start))
    print('Missing value of 4 csvs:', cnts)


    print('Start processing testing data')
    testing_data, testing_alert_key = [], []
    for i in range(df_public_x.shape[0]):
        # from alert key to get customer information
        cur_data = df_public_x.iloc[i]
        alert_key, date = cur_data['alert_key'], cur_data['date']

        cus_info = df_cus_info[df_cus_info['alert_key']==alert_key].iloc[0]
        cus_id = cus_info['cust_id']
        cus_features = cus_info.values[2:]

        for item, df in enumerate(cus_data):
            cus_additional_info = df[df['cust_id']==cus_id]
            cus_additional_info = cus_additional_info[cus_additional_info[date_col[item]]<=date]

            if cus_additional_info.empty:
                len_item = len(data_use_col[item])
                if item == 2:
                    len_item -= 1
                cus_features = np.concatenate((cus_features, [np.nan] * len_item), axis=0)
            else:
                cur_cus_feature = cus_additional_info.loc[cus_additional_info[date_col[item]].idxmax()]
                cur_cus_feature = cur_cus_feature.values[data_use_col[item]]
                # 處理 實際金額 = 匯率*金額
                if item == 2:
                    cur_cus_feature = np.concatenate((cur_cus_feature[:2], [cur_cus_feature[2]*cur_cus_feature[3]], cur_cus_feature[4:]), axis=0)
                cus_features = np.concatenate((cus_features, cur_cus_feature), axis=0)

        testing_data.append(cus_features)
        testing_alert_key.append(alert_key)
        # print(cus_features)
        print('\r processing data {}/{}'.format(i+1, df_public_x.shape[0]), end = '')
    return np.array(training_data), labels, np.array(testing_data), testing_alert_key

data_dir = './ITF_data'
training_data, labels, testing_data, testing_alert_key = preprocess(data_dir)

"""# Preprocess private data"""

def preprocess(data_dir):
    # declare csv path
    df_private_x = os.path.join(data_dir, 'private_x_alert_date.csv')
    cus_info_csv = os.path.join(data_dir, 'private_x_custinfo_full_hashed.csv')

    ccba_csv = os.path.join(data_dir, 'private_x_ccba_full_hashed.csv')
    cdtx_csv = os.path.join(data_dir, 'private_x_cdtx0001_full_hashed.csv')
    dp_csv = os.path.join(data_dir, 'private_x_dp_full_hashed.csv')
    remit_csv = os.path.join(data_dir, 'private_x_remit1_full_hashed.csv')

    cus_csv = [ccba_csv, cdtx_csv, dp_csv, remit_csv]
    date_col = ['byymm', 'date', 'tx_date', 'trans_date']
    data_use_col = [[1,3,4,5,6,7,8,9],[2,3,4],[1,4,5,6,7,8,9,10,11],[2,3]]
    
    print('Reading csv...')
    # read csv
    df_cus_info = pd.read_csv(cus_info_csv)
    df_date = pd.read_csv(df_private_x)
    cus_data = [pd.read_csv(_x) for _x in cus_csv]

    # do label encoding
    le = LabelEncoder()
    cus_data[2].debit_credit = le.fit_transform(cus_data[2].debit_credit)

    cnts = [0] * 4

    print('Start processing testing data')
    testing_data, testing_alert_key = [], []
    for i in range(df_date.shape[0]):
        # from alert key to get customer information
        cur_data = df_date.iloc[i]
        alert_key, date = cur_data['alert_key'], cur_data['date']

        cus_info = df_cus_info[df_cus_info['alert_key']==alert_key].iloc[0]
        cus_id = cus_info['cust_id']
        cus_features = cus_info.values[2:]

        for item, df in enumerate(cus_data):
            cus_additional_info = df[df['cust_id']==cus_id]
            cus_additional_info = cus_additional_info[cus_additional_info[date_col[item]]<=date]

            if cus_additional_info.empty:
                len_item = len(data_use_col[item])
                if item == 2:
                    len_item -= 1
                cus_features = np.concatenate((cus_features, [np.nan] * len_item), axis=0)
            else:
                cur_cus_feature = cus_additional_info.loc[cus_additional_info[date_col[item]].idxmax()]
                cur_cus_feature = cur_cus_feature.values[data_use_col[item]]
                # 處理 實際金額 = 匯率*金額
                if item == 2:
                    cur_cus_feature = np.concatenate((cur_cus_feature[:2], [cur_cus_feature[2]*cur_cus_feature[3]], cur_cus_feature[4:]), axis=0)
                cus_features = np.concatenate((cus_features, cur_cus_feature), axis=0)

        testing_data.append(cus_features)
        testing_alert_key.append(alert_key)
        # print(cus_features)
        print('\r processing data {}/{}'.format(i+1, df_date.shape[0]), end = '')
    return np.array(testing_data), testing_alert_key

path = './ITF_data/private-data'
private_test, private_testing_alert_key = preprocess(path)
print(private_test.shape)


test_labels = pd.read_csv('./ITF_data/' + "24_ESun_public_y_answer.csv", sep=',', header=None)
test_labels = np.int_(np.array(test_labels)[1:, 1])


count_list, bad_index_list = [], []
threshold = 13
bad_data_count = 0
for i, row in enumerate(training_data):
  count = 0
  for ele in row:
    if pd.isna(ele):
      count += 1
  if count > threshold:
    bad_data_count += 1
    bad_index_list.append(i)
  count_list.append(count)

print(np.array(count_list).mean())
print(np.median(np.array(count_list)))
print(np.max(np.array(count_list)))
print("bad data count:", bad_data_count)

training_data = np.delete(training_data, bad_index_list, axis = 0)
labels = np.delete(labels, bad_index_list, axis = 0)

training_data = np.concatenate((training_data , testing_data), axis = 0)
labels = np.concatenate((labels, test_labels), axis = 0)

"""#Split training data"""

# split train / validation set
from sklearn.model_selection import train_test_split
train_set, val_set, train_labels, val_labels = train_test_split(training_data, labels, test_size=0.1, random_state=7777)

"""# count loss"""

neg, pos = 0,0
for l in train_labels:
  if l == 0:
    neg += 1
  else:
    pos += 1
print(neg, pos)

neg, pos = 0,0
for l in val_labels:
  if l == 0:
    neg += 1
  else:
    pos += 1
print(neg, pos)
val_pos = pos

neg, pos = 0,0
for l in test_labels:
  if l == 0:
    neg += 1
  else:
    pos += 1
print(neg, pos)
test_pos = pos

"""## 缺失值補漏"""

''' Missing Value Imputation '''

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor


def imputation_fit(data):
  numerical_data = data[:, numerical_index]
  non_numerical_data = data[:, non_numerical_index]

  Iter_median.fit(numerical_data)
  Iter_frequent.fit(non_numerical_data)

def imputation_transform(data):
  numerical_data = data[:, numerical_index]
  non_numerical_data = data[:, non_numerical_index]

  numerical_data = Iter_median.transform(numerical_data)
  non_numerical_data = Iter_frequent.transform(non_numerical_data)
  data = np.concatenate((non_numerical_data, numerical_data), axis=1)
  return data

def col_imputation_fit(data):
  numerical_data = data[:, numerical_index]
  non_numerical_data = data[:, non_numerical_index]
  for i in range(len(numerical_index)):
    Iter_median.fit(numerical_data[:, i].reshape(-1,1))
  for i in range(len(non_numerical_index)):
    Iter_frequent.fit(non_numerical_data[:, i].reshape(-1,1))
  data = np.concatenate((non_numerical_data, numerical_data), axis=1)
  return data

def col_imputation_transform(data):
  numerical_data = data[:, numerical_index]
  non_numerical_data = data[:, non_numerical_index]
  for i in range(len(numerical_index)):
    numerical_data[:, i] = Iter_median.transform(numerical_data[:, i].reshape(-1,1)).reshape(-1)
  for i in range(len(non_numerical_index)):
    non_numerical_data[:, i] = Iter_frequent.transform(non_numerical_data[:, i].reshape(-1,1)).reshape(-1)
  data = np.concatenate((non_numerical_data, numerical_data), axis=1)
  return data

"""## Encoding"""

from category_encoders.target_encoder import TargetEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.cat_boost import CatBoostEncoder


def encode_fit(data, data_labels):
  data = pd.DataFrame(data)
  for index in one_hot_index:
    data[index] = data[index].astype('category')
  enc.fit(data, data_labels)

def encode_transform(data):
  data = pd.DataFrame(data)
  for index in one_hot_index:
    data[index] = data[index].astype('category')
  data = enc.transform(data)
  data = data.to_numpy()
  # print(data.shape)
  return data

"""# Testing data preprocess"""

data_dir = './ITF_data/'
public_private_test_csv = os.path.join(data_dir, '預測的案件名單及提交檔案範例.csv')
df_public_private_test = pd.read_csv(public_private_test_csv)

testing_data = np.concatenate((testing_data, private_test), axis = 0)

non_numerical_index = [1,3,15,16,18,21,22]
numerical_index = [2,4,5,6,7,8,9,10,11,14]

Iter_median = IterativeImputer(max_iter=30, random_state=1997, initial_strategy='median')
Iter_frequent = IterativeImputer(max_iter=30, random_state=1997, initial_strategy='most_frequent')
one_hot_index = [0,1,2,3,4]
enc = TargetEncoder(cols=one_hot_index, handle_unknown='value', smoothing=0.9)

imputation_fit(train_set)
train_set = imputation_transform(train_set)
val_set = imputation_transform(val_set)
testing_data = imputation_transform(testing_data)

encode_fit(train_set, train_labels)
train_set = encode_transform(train_set)
val_set = encode_transform(val_set)
testing_data = encode_transform(testing_data)

new_index = [0, 9, 10, 11, 12, 13, 16, 1, 3, 4, 6, 14]
new_index = [0,1,3,4,6,9,10,11,12,13,14,16]

train_set = train_set[:, new_index]
val_set = val_set[:, new_index]
testing_data = testing_data[:, new_index]