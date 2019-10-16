
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def data_reader(path, norm):
    p=0 #change to something else to not create plots of input data
    df = pd.read_csv(path)
    df = df.drop('id',axis=1) #remove id column
#    
#    df['year']=pd.DateTimeIndex(df['date']).year #create day mo yr columns
#    df['month']=pd.DateTimeIndex(df['date']).month
#    df['day']=pd.DateTimeIndex(df['date']).day
##        
    ############## wasn't sure if zipcode/lat/long counted as a feature what do you guys think?#######
    features=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',
              'view','condition','grade','sqft_above','sqft_basement','yr_built',\
              'yr_renovated','sqft_living15','sqft_lot15']
    num_features=['waterfront','condition','grade']
      
    ####for statistics of features###
    #stats={}
    #for feat in features:
    #    stats[(feat,'mean')]=df[feat].mean()
    #    stats[(feat,'std_dev')]=df[feat].std()
    #    stats[(feat,'range')]=[df[feat].min(), df[feat].max()]
    #    if p==1:
    #        plt.figure()
    #        df[feat].plot.kde() #need to play w bandwidth on some of these
    #        plt.ylabel(feat)
    #for feat in num_features:
    #    pctwith=df[feat]    
    
    #normalize data to range of 0 to 1
    if norm==True:
        for feat in features:
            df[feat]=df[feat]/df[feat].max()

    data=df.to_numpy()
    return data
