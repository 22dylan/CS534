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


def results_to_csv(csv_filename, results):
    #format results of w
    w_results_flat = []
    sse_results_flat = []

    for key, trial in results.items():
        # print(trial)
        w_new_row = [key, trial['step_size'], trial['lambda_reg'], trial['convergence_count']]
        SSE_new_row = [key]
        
        #collect w
        for vals in trial['w_values']:
            w_new_row.append(vals)

        #collect SSE
        for sse in trial['SSE']:
            for val in sse[1]:
                SSE_new_row.append(val)

        w_results_flat.append(w_new_row)
        sse_results_flat.append(SSE_new_row)



    #save results of w
    with open("../output/csv/{0}_w.csv".format(csv_filename), mode='w', newline='') as w_file:
        w_writer = csv.writer(w_file, delimiter=',')
        for row in w_results_flat:
            w_writer.writerow(row)
    
    #save results of sse
    with open("../output/csv/{0}_sse.csv".format(csv_filename), mode='w', newline='') as w_file:
        w_writer = csv.writer(w_file, delimiter=',')
        for row in sse_results_flat:
            w_writer.writerow(row)


def validation_to_csv(csv_filename, validation):
    results_flat = []

    for key, trial in validation.items():
        # print(trial)
        new_row = [key]
        
        #collect w
        for vals in trial:
            new_row.append(vals)
        results_flat.append(new_row)

    #save results of sse
    with open("../output/csv/{0}.csv".format(csv_filename), mode='w', newline='') as w_file:
        w_writer = csv.writer(w_file, delimiter=',')
        for row in results_flat:
            w_writer.writerow(row)


def price_pickle_to_csv(part_num, predicted_y):
    #save results of sse
    with open("../output/csv/predicted_y_{0}.csv".format(part_num), mode='w', newline='') as w_file:
        w_writer = csv.writer(w_file, delimiter=',')
        for y in predicted_y:
            w_writer.writerow([y])
