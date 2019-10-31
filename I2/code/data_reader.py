import numpy as np
import pandas as pd
import csv

pd.set_option('display.max_columns', None)


def data_reader(path):
    data = np.genfromtxt(path, delimiter=',')
    data[data[:, 0] == 3, 0] = 1
    data[data[:, 0] == 5, 0] = -1
    data = np.column_stack((data, np.ones(len(data))))
    return data


def results_to_csv(csv_filename, results):
    # format results of w
    w_results_flat = []
    sse_results_flat = []

    for key, trial in results.items():
        # print(trial)
        w_new_row = [key, trial['step_size'], trial['lambda_reg'], trial['convergence_count']]
        SSE_new_row = [key]

        # collect w
        for vals in trial['w_values']:
            w_new_row.append(vals)

        # collect SSE
        for sse in trial['SSE']:
            for val in sse[1]:
                SSE_new_row.append(val)

        w_results_flat.append(w_new_row)
        sse_results_flat.append(SSE_new_row)

    # save results of w
    with open("../output/csv/{0}_w.csv".format(csv_filename), mode='w', newline='') as w_file:
        w_writer = csv.writer(w_file, delimiter=',')
        for row in w_results_flat:
            w_writer.writerow(row)

    # save results of sse
    with open("../output/csv/{0}_sse.csv".format(csv_filename), mode='w', newline='') as w_file:
        w_writer = csv.writer(w_file, delimiter=',')
        for row in sse_results_flat:
            w_writer.writerow(row)


def validation_to_csv(csv_filename, validation):
    results_flat = []

    for key, trial in validation.items():
        # print(trial)
        new_row = [key]

        # collect w
        for vals in trial:
            new_row.append(vals)
        results_flat.append(new_row)

    # save results of sse
    with open("../output/csv/{0}.csv".format(csv_filename), mode='w', newline='') as w_file:
        w_writer = csv.writer(w_file, delimiter=',')
        for row in results_flat:
            w_writer.writerow(row)


def price_pickle_to_csv(part_num, predicted_y):
    # save results of sse
    with open("../output/csv/predicted_y_{0}.csv".format(part_num), mode='w', newline='') as w_file:
        w_writer = csv.writer(w_file, delimiter=',')
        for y in predicted_y:
            w_writer.writerow([y])