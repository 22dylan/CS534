import numpy as np
import pandas as pd
import csv
from datetime import datetime
pd.set_option('display.max_columns', None)

def data_reader(path, norm, normval=None):
    p=0 #change to something else to not create plots of input data
    df = pd.read_csv(path)
    df = df.drop('id',axis=1) #remove id column

    df['year']=pd.DatetimeIndex(df['date']).year #create day mo yr columns
    df['month']=pd.DatetimeIndex(df['date']).month
    df['day']=pd.DatetimeIndex(df['date']).day
        
    df = df.drop('date',axis=1) #remove date column

    features=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront',\
              'view','condition','grade','sqft_above','sqft_basement','yr_built',\
              'yr_renovated','zipcode', 'lat','long','sqft_living15','sqft_lot15','price',\
              'year', 'month', 'day']
    num_features=['waterfront','condition','grade']
    
    #normalize data to range of 0 to 
    if norm==True:
        if normval is None:
            # normval=df/df.max()
            normval = df.max()
        for feat in features:
            if feat in df.keys():
                df[feat]=df[feat]/float(normval[feat])
    data=df.to_numpy()
    return data, normval


def save_results(filename, results):
    #format results of w
    results_flat = []

    for trial in results:
        new_row = [trial['step_size'], trial['lambda_reg'], trial['convergence_count'], trial['SSE']]
        for vals in trial['w_values']:
            new_row.append(vals)

        results_flat.append(new_row)

    #save results of w
    with open("../output/{0}_w.csv".format(filename), mode='w') as w_file:
        w_writer = csv.writer(w_file, delimiter=',')
        for row in results_flat:
            w_writer.writerow(row)
