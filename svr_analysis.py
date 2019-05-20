#!/usr/bin/env python3 -w
import sys
import os
import math
import argparse
import datetime
import json
import csv
import matplotlib

import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

import model_io

fontSize = 9
crossValidations = 5

totalItems = -1
totalSpectra = -1
trainIndex = None
testIndex = None

svrParameters = {'degree':[3, 4, 5, 6],
                 'C': np.logspace(-5, 15, num=21, base=2.0, dtype=np.float64),
                 'gamma': np.logspace(-15, 3, num=19, base=2.0, dtype=np.float64)}

def svr_validate(svr, validate_x, validate_y):
    predict_y = svr.predict(validate_x)
    #print(predict_y)
    score = r2_score(validate_y, predict_y)
    mse = mean_squared_error(validate_y, predict_y)
    sep = np.std(predict_y - validate_y)
    rpd = np.std(validate_y)/sep
    bias = np.mean(predict_y - validate_y)
    return predict_y, score, mse, sep, rpd, bias

def main(argv):
    def restricted_float(x):
        x = float(x)
        if x < 0.0 or x >= 1.0:
            raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
        return x

    parser = argparse.ArgumentParser(description='Generic PLS regression')
    
    parser.add_argument('--data_file', action='store', nargs='?', default='data.csv', dest='data_file', type=str, required=False, help='Specify the name of csv file.')
    parser.add_argument('--target_file', action='store', nargs='?', default='target.csv', dest='target_file', type=str, required=False, help='Specify the filename containing targets corresponding to input_file.')
    parser.add_argument('--model_file', action='store', nargs='?', default='pls_model.json', dest='model_file', type=str, required=False, help='Specify the JSON file that will contain meta-data of trained PLS model.')
    parser.add_argument('--config_file', action='store', nargs='?', default='pls_config.json', dest='config_file', type=str, required=False, help='Specify the JSON file that will contain configuration of PLS model.')
    parser.add_argument('--testset_ratio', action='store', nargs='?', default=0.33, dest='testset_ratio', type=restricted_float, required=False, help='Specify the item ratio of test set for cross-validation.')
    args = parser.parse_args()

    print('data_file:', args.data_file)
    print('target_file:', args.target_file)
    print('model_file:', args.model_file)
    print('config_file:', args.config_file)
    print('testset_ratio:', args.testset_ratio)
    
    if not os.path.isfile(args.data_file):
        raise Exception('The data_file does not exist.'%(args.data_file))
        
    if not os.path.isfile(args.target_file):
        raise Exception('The target_file does not exist.'%(args.target_file))

    data_list = []
    data_reader = csv.DictReader(open(args.data_file, newline=''))
    for row in data_reader:
        data_list.append(list(row.values()))

    Xs = np.array(data_list, dtype=np.float64)
    
    target_list = []
    target_reader = csv.DictReader(open(args.target_file, newline=''))
    for row in target_reader:
        target_list.append(list(row.values()))

    Ys = np.array(target_list, dtype=np.float64)

    assert(Xs.shape[0] == Ys.shape[0])

    print('Total samples:', Xs.shape[0])
    totalSamples = Xs.shape[0]

    # Shuffle Xs, and Ys
    min_Y,max_Y = math.floor(np.min(Ys)), math.ceil(np.max(Ys))
    print(min_Y, max_Y)
    
    bins_Y = np.arange(min_Y, max_Y+1.0, 0.5)
    print(bins_Y)
    class_list = ['None'] * totalSamples
    for i in range(bins_Y.shape[0]-1):
        indexing = np.logical_and(Ys >= bins_Y[i], Ys < bins_Y[i+1])
        filter_idx = np.argwhere(indexing == True)[:,0]
        filter_y = Ys[indexing]
        if len(filter_y) > 0:
            if len(filter_y) == 1:
                train_idx = filter_idx
                train_y = filter_y
                test_idx = np.zeros((0,), dtype=np.float64) # Tricky !!!
                test_y = np.zeros((0,), dtype=np.float64) # Tricky !!!
            else:
                train_idx, test_idx, train_y, test_y = model_selection.train_test_split(filter_idx, filter_y, test_size=args.testset_ratio, random_state=42)

            for idx in train_idx: class_list[idx] = 'train'
            for idx in test_idx: class_list[idx] = 'test'
            #print(filter_x, '->', train_x, test_x)

    train_idx = [i for i,x in enumerate(class_list) if x == 'train']
    test_idx = [i for i,x in enumerate(class_list) if x == 'test']
    test_y = Ys[test_idx]
    #print(test_y)

    fig = plt.figure(figsize=(16,9))
    #fig.canvas.set_window_title('calibration %s/*.txt with %s into %s, %d max outlier, %s spectra'%(dir_name, excel_name, model_name, max_outliers, preprocess))
    fig.tight_layout()

    plt.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.95, wspace=0.4, hspace=0.6)

    with plt.style.context(('ggplot')):
        ax = plt.subplot2grid((4,6), (0,0), rowspan=2, colspan=2)
        n, bins, patches = ax.hist(Ys, bins=bins_Y, facecolor='b',rwidth=0.7, density=False)
        ax.hist(test_y, bins=bins_Y, facecolor='r', rwidth=0.7, density=False)
        ax.set_xlabel('Target')
        ax.set_ylabel('Count')
        ax.grid(True)
        y_range = max(n) - min(n)
        x_range = max(bins) - min(bins)
        ax.text(min(bins)+0.05*x_range, max(n)-0.1*y_range, 'Total: %d'%(sum(n),), fontsize=fontSize)
        ax.text(min(bins)+0.05*x_range, max(n)-0.17*y_range, ' Calibrate: %d'%(sum(n)-test_y.shape[0],), color='g', fontsize=fontSize)
        ax.text(min(bins)+0.05*x_range, max(n)-0.24*y_range, ' Validate: %d'%(test_y.shape[0],), color='r', fontsize=fontSize)

        plt.draw()
        plt.pause(0.001)

    Xcalibrate = Xs[train_idx]
    Ycalibrate = Ys[train_idx]

    print('Perform grid search for finding hyper-parameters of SVR.')
    #svr = SVR(kernel='rbf', C=0.1, gamma=0.01)
    #svr.fit(calibrateX, calibrateY)

    svr = SVR(kernel='rbf')
    #svr = SVR(kernel='poly')
    clf = model_selection.GridSearchCV(svr, svrParameters, cv=crossValidations)
    clf.fit(Xcalibrate, Ycalibrate.ravel())
    print(clf.best_params_)

    bestSVR = SVR(kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
    bestSVR.fit(Xcalibrate, Ycalibrate.ravel())

    Xtest = Xs[test_idx]
    Ytest = Ys[test_idx]

    Ypredict,predict_score,predict_mse,sep,rpd,bias = svr_validate(bestSVR, Xtest, Ytest)
    sys.stdout.write('\n')
    print('R2: %5.3f'%(predict_score,))
    print('MSE: %5.3f'%(predict_mse,))
    print('SEP: %5.3f'%(sep,))
    print('RPD: %5.3f'%(rpd,))
    print('Bias: %5.3f'%(bias,))

    with plt.style.context(('ggplot')):
        y_range = max(Ytest) - min(Ytest)
        x_range = max(Ypredict) - min(Ypredict)

        print(Ytest)
        print(Ypredict)
        z = np.polyfit(Ytest.reshape(-1), Ypredict.reshape(-1), 1)
        
        #fig,ax = plt.subplots(figsize=(9, 5))
        ax = plt.subplot2grid((4,6), (2,0), colspan=2, rowspan=2)
        ax.scatter(Ypredict, Ytest, c='red', edgecolor='k')
        ax.plot(z[1]+z[0]*Ytest, Ytest, c='blue', linewidth=1)
        ax.plot(Ytest, Ytest, color='green', linewidth=1)
        ax.set_xlabel('Predicted', fontsize=fontSize)
        ax.set_ylabel('Measured', fontsize=fontSize)
        #ax.set_title('Prediction', fontsize=fontSize)

        # Print the scores on the plot
        ax.text(min(Ytest)+0.01*x_range, max(Ytest)-0.1*y_range, 'R$^{2}$: %5.3f'%(predict_score,), fontsize=fontSize)
        ax.text(min(Ytest)+0.01*x_range, max(Ytest)-0.17*y_range, 'MSE: %5.3f'%(predict_mse,), fontsize=fontSize)
        ax.text(min(Ytest)+0.01*x_range, max(Ytest)-0.24*y_range, 'SEP: %5.3f'%(sep,), fontsize=fontSize)
        ax.text(min(Ytest)+0.01*x_range, max(Ytest)-0.31*y_range, 'RPD: %5.3f'%(rpd,), fontsize=fontSize)
        ax.text(min(Ytest)+0.01*x_range, max(Ytest)-0.38*y_range, 'Bias: %5.3f'%(bias,), fontsize=fontSize)
        plt.draw()
        plt.pause(0.001)
        
    ### Save pls model as JSON format
    model_io.saveModelAsJSON(bestSVR, args.model_file)
    
    config_dict = { 'date':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'data_file':args.data_file,
                    'target_file':args.target_file,
                    'model_file':args.model_file,
                    'config_file':args.config_file,
                    'testset_ratio':args.testset_ratio }

    with open(args.config_file, 'w') as fp:
        json.dump(config_dict, fp, indent=4, sort_keys=False)
        
    plt.show()

    return 0

if __name__ == '__main__':
    main(sys.argv)
