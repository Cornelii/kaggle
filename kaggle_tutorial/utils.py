import numpy as np
import pandas as pd

def apply_log(x):
        if x < 1:
            x = abs(x)+1
        return np.log(x)

def preprocessing(df_train):
    # date
    df_train['date'] = df_train['date'].apply(lambda x: x[:6])
    df_train['year'] = df_train['date'].apply(lambda x: int(x[:4]))
    df_train['month'] = df_train['date'].apply(lambda x: int(x[4:6]))
    df_train.drop(['date'],axis=1, inplace=True)
    
    # arbitrarily addition
    df_train['rooms'] = df_train['bedrooms']+df_train['bathrooms']
    df_train['avg_room'] = df_train.sqft_above / (df_train.rooms+1)
    df_train['sqft_inner_total'] = df_train.sqft_above+df_train.sqft_basement
    df_train['sqft_total'] = df_train.sqft_inner_total + df_train.sqft_lot
    df_train['sqft_total_origin'] = df_train.sqft_living + df_train.sqft_lot
    df_train['sqft_total_15'] = df_train.sqft_living15 + df_train.sqft_lot15
    
    
    lambda_fn = lambda row: row['year'] - row['yr_built'] + 2
    df_train['age'] = df_train.apply(lambda_fn, axis=1)
    
    df_train['lat'] = df_train['lat'].values - 47
    df_train['long'] = df_train['long'].values + 123
    
#     zip_index = df_train.loc[df_train['price']>=2000000].groupby('zipcode').sum().index
#     df_train['mask_zipcode'] = df_train['zipcode'].apply(lambda x: 1 if x in zip_index else 0)
    
    skew_columns = ['rooms','sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement',
                    'lat','long','sqft_living15','sqft_lot15','sqft_inner_total',
                    'sqft_total','sqft_total_origin','sqft_total_15','age']

    for col in skew_columns:
        df_train[col] = df_train[col].apply(apply_log)

        
    df_train.drop(['id'], axis=1, inplace=True)
    
    return df_train

def get_X_y(df, is_train_data=True):
    if is_train_data:
        X = df.drop(['price'], axis=1).values
        y = df.price.values
        return X, y
    else:
        X = df.values
        return X
    
def get_data(df, is_train_data = True):
    df = preprocessing(df)
    
    if is_train_data:
        X, y = get_X_y(df, is_train_data=is_train_data)
        return X, y
    else:
        X = get_X_y(df, is_train_data=is_train_data)
        return X


    

