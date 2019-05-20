import sys
import os
import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

import model_io
import pls_analysis

fontSize = 8
totalItems = 0
totalSpectra = 0
snapPositions = 4
#feasibleIndex = range(400, 2400, 1)

def main(argv):
    parser = argparse.ArgumentParser(description='Generic PLS deploy')
    parser.add_argument('--data_file', action='store', nargs='?', default='data.csv', dest='data_file', type=str, required=False, help='Specify the name of csv file.')
    parser.add_argument('--target_file', action='store', nargs='?', default='target.csv', dest='target_file', type=str, required=False, help='Specify the filename containing targets corresponding to input_file.')
    parser.add_argument('--model_file', action='store', nargs='?', default='pls_model.json', dest='model_file', type=str, required=False, help='Specify the JSON file that will contain meta-data of trained PLS model.')
    parser.add_argument('--config_file', action='store', nargs='?', default='pls_config.json', dest='config_file', type=str, required=False, help='Specify the JSON file that will contain configuration of PLS model.')
    parser.add_argument('--result_file', action='store', nargs='?', default='result.csv', dest='result_file', type=str, required=False, help='Specify the result file containing pls deploying.')
    args = parser.parse_args()

    print('data_file:', args.data_file)
    print('target_file:', args.target_file)
    print('model_file:', args.model_file)
    print('config_file:', args.config_file)
    print('result_file:', args.result_file)

    if not os.path.isfile(args.data_file):
        raise Exception('The data_file does not exist.'%(args.data_file))
        
    #if not os.path.isfile(args.target_file):
    #    raise Exception('The target_file does not exist.'%(args.target_file))

    if not os.path.isfile(args.model_file):
        raise Exception('The model_file does not exist.'%(args.model_file))
        
    if not os.path.isfile(args.config_file):
        raise Exception('The config_file does not exist.'%(args.config_file))

    ### Loading data
    data_list = []
    data_reader = csv.DictReader(open(args.data_file, 'r', newline=''))
    for row in data_reader:
        data_list.append(list(row.values()))
        
    Xs = np.array(data_list, dtype=np.float64)
    
    print('Total samples:', Xs.shape[0])
    totalSamples = Xs.shape[0]

    ### Loading PLS model
    plsModel = model_io.loadModelFromJSON(args.model_file)
    predictYs = plsModel.predict(Xs)

    ### Record
    result_writer = csv.DictWriter(open(args.result_file, 'w', newline=''), fieldnames=['result'])
    result_writer.writeheader()
    for result in predictYs:
        result_writer.writerow({'result': result[0]})

    ### Scoring
    if os.path.isfile(args.target_file):
        target_list = []
        target_reader = csv.DictReader(open(args.target_file, newline=''))
        for row in target_reader:
            target_list.append(list(row.values()))

        Ys = np.array(target_list, dtype=np.float64)
        assert(Xs.shape[0] == Ys.shape[0])
        
        score = r2_score(Ys, predictYs)
        mse = mean_squared_error(Ys, predictYs)
        sep = np.std(predictYs[:,0] - Ys)
        rpd = np.std(Ys)/sep
        bias = np.mean(predictYs[:,0] - Ys)
        print('R2: %5.3f'%(score,))
        print('MSE: %5.3f'%(mse,))
        print('SEP: %5.3f'%(sep,))
        print('RPD: %5.3f'%(rpd,))
        print('Bias: %5.3f'%(bias,))
    
    return 0

if __name__ == '__main__':
    main(sys.argv)
