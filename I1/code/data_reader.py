import numpy as np
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt

def data_reader(path, norm, normval=None):
    p=0 #change to something else to not create plots of input data
    df = pd.read_csv(path)
    df = df.drop('id',axis=1) #remove id column
#    
    df['year']=pd.DatetimeIndex(df['date']).year #create day mo yr columns
    df['month']=pd.DatetimeIndex(df['date']).month
    df['day']=pd.DatetimeIndex(df['date']).day
        
    df = df.drop('date',axis=1) #remove date column

    ############## wasn't sure if zipcode/lat/long counted as a feature what do you guys think?#######
    features=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',\
              'view','condition','grade','sqft_above','sqft_basement','yr_built',\
              'yr_renovated','zipcode', 'lat','long','sqft_living15','sqft_lot15','price']
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
    
    #normalize data to range of 0 to 
    
    if norm==True:
        if normval is None:
            # normval=df/df.max()
            normval = df.max()
        df = df/df.max()
        # for feat in features:
        #     df[feat]=df[feat]/normval[feat]
                
    data=df.to_numpy()
    print(data)
    # data=normval.to_numpy()


    return data, normval

def save_results(filename, w, data_norm, norm_scale, iterations):
    #format results of w
    w_flat = []
    for ssr_key, ssr_val in w.items():
        for lvr_key, lvr_val in ssr_val.items():
            new_row = [ssr_key,lvr_key ] #add keys
            new_row.append(iterations[ssr_key][lvr_key]) #add iterations
            for item in lvr_val: #add the data
                new_row.append(item) 
            w_flat.append(new_row)

    #save results of w
    with open("../results/{0}_w.csv".format(filename), mode='w') as w_file:
        w_writer = csv.writer(w_file, delimiter=',')
        for row in w_flat:
            w_writer.writerow(row)

