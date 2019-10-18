import numpy as np
import pandas as pd
from datetime import datetime

def data_reader(path, norm, normval=None):
    p=0 #change to something else to not create plots of input data
    df = pd.read_csv(path)
    df = df.drop('id',axis=1) #remove id column
#    
    df['year']=pd.DatetimeIndex(df['date']).year #create day mo yr columns
    df['month']=pd.DatetimeIndex(df['date']).month
    df['day']=pd.DatetimeIndex(df['date']).day
        
    df = df.drop('date',axis=1) #remove date column

    features=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',\
              'view','condition','grade','sqft_above','sqft_basement','yr_built',\
              'yr_renovated','zipcode', 'lat','long','sqft_living15','sqft_lot15','price']
    num_features=['waterfront','condition','grade']
    
    #normalize data to range of 0 to 
    
    if norm==True:
        if normval is None:
            normval=df/df.max()
        for feat in features:
            df[feat]=df[feat]/normval[feat]
                
    data=df.to_numpy()

    return data, normval
