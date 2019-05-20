import sys
import os
import datetime
import argparse
import csv
import math
import numpy as np
import json
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

import model_io

fontSize = 9
# Confidence ratio for Outlier removal
confidenceRatio = 0.95
# Maximum number of outliers for removal
#crossValidations = 10
crossValidations = 5


def pls_find_best_components_and_outliers(calib_x, calib_y, max_comp=40, max_outliers=70, valid_x=None, valid_y=None, cross_valid=crossValidations):
    mse = np.zeros((max_comp, max_outliers), dtype=np.float64)
    for i in range(max_comp):
        pls = PLSRegression(n_components=i+1, max_iter=5000)
        
        if valid_x is not None and valid_x.shape[0] > 0 and valid_y is not None and valid_y.shape[0] > 0:
            pls.fit(calib_x, calib_y)
            y_cv = pls.predict(valid_x)
            mse[i,0] = mean_squared_error(valid_y, y_cv)
        else:
            y_cv = model_selection.cross_val_predict(pls, calib_x, calib_y, cv=cross_valid)
            mse[i,0] = mean_squared_error(calib_y, y_cv)
            pls.fit(calib_x, calib_y)
            
        # X scores
        T = pls.x_scores_ # Low scores means that the signals are "good fit" to model
        # X loadings
        P = pls.x_loadings_
        # Error
        Err = calib_x - np.dot(T, P.T)
        # Q-residuals (sum over the row of the error matrix
        Q = np.sum(Err**2, axis=1)
        # Hotelling's T-squared (X scores are nomalized by standard diviation)
        Tsq = np.sum((T/np.std(T, axis=0))**2, axis=1)

        # RMS distance 
        rms_dist = np.flip(np.argsort(np.sqrt(Q**2+Tsq**2)), axis=0)
        
        # Sort calibration spectra according to descending RMS distance
        sorted_x = calib_x[rms_dist, :]
        sorted_y = calib_y[rms_dist]
            
        for j in range(1, max_outliers):
            if valid_x is not None and valid_x.shape[0] > 0 and valid_y is not None and valid_y.shape[0] > 0:
                pls.fit(sorted_x[j:,:], sorted_y[j:])
                y_cv = pls.predict(valid_x)
                mse[i,j] = mean_squared_error(valid_y, y_cv)
            else:
                y_cv = model_selection.cross_val_predict(pls, sorted_x[j:,:], sorted_y[j:], cv=cross_valid)
                mse[i,j] = mean_squared_error(sorted_y[j:], y_cv)
                
        comp = 100*(i+1)/max_comp
        sys.stdout.write('\r%d%% completed'%(comp,))
        sys.stdout.flush()
    sys.stdout.write('\n')

    msemin = np.where(mse == np.min(mse[np.nonzero(mse)]))
    mini, minj = msemin[0][0], msemin[1][0]
    print('Suggested number of components: ', mini+1)
    print('Suggested number of removal outliers: ', minj)

    return mse, (mini,minj)

def pls_calibrate(calib_x, calib_y, components, outliers):
    #mseminx, mseminy = np.where(mse == np.min(mse[np.nonzero(mse)]))
    #print(mseminx, mseminy)
    
    pls = PLSRegression(n_components=components, max_iter=5000)
    pls.fit(calib_x, calib_y)

    rms_dist = None
    if outliers > 0:
        # X scores
        T = pls.x_scores_ # Low scores means that the signals are "good fit" to model
        # X loadings
        P = pls.x_loadings_
        # Error
        Err = calib_x - np.dot(T, P.T)
        # Q-residuals (sum over the row of the error matrix
        Q = np.sum(Err**2, axis=1)
        # Hotelling's T-squared (X scores are nomalized by standard diviation)
        Tsq = np.sum((T/np.std(T, axis=0))**2, axis=1)

        # RMS distance 
        rms_dist = np.flip(np.argsort(np.sqrt(Q**2+Tsq**2)), axis=0)
        
        # Sort calibration spectra according to descending RMS distance
        sorted_x = calib_x[rms_dist,:]
        sorted_y = calib_y[rms_dist]

        print('Removed Data:', sorted_x[0:outliers,:])
        print('Removed Target:', sorted_y[0:outliers])
    
        pls.fit(sorted_x[outliers:,:], sorted_y[outliers:])

    return pls, rms_dist

#score, mse, sep, rpd, bias = pls_validate(pls, validateX, validateY)
def pls_validate(pls, validate_x, validate_y):
    predict_y = pls.predict(validate_x)
    score = r2_score(validate_y, predict_y)
    mse = mean_squared_error(validate_y, predict_y)
    sep = np.std(predict_y[:,0] - validate_y)
    rpd = np.std(validate_y)/sep
    bias = np.mean(predict_y[:,0] - validate_y)
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
    parser.add_argument('--max_components', action='store', nargs='?', default=20, dest='max_components', type=int, required=False, help='Specify the maximum number of components that will be option in calibration of PLS model.')
    parser.add_argument('--max_outliers', action='store', nargs='?', default=20, dest='max_outliers', type=int, required=False, help='Specify the maximum number of removal outliers.')
    args = parser.parse_args()

    print('data_file:', args.data_file)
    print('target_file:', args.target_file)
    print('model_file:', args.model_file)
    print('config_file:', args.config_file)
    print('testset_ratio:', args.testset_ratio)
    print('max_components:', args.max_components)
    print('max_outliers:', args.max_outliers)
    
    if not os.path.isfile(args.data_file):
        raise Exception('The data_file does not exist.'%(args.data_file))
        
    if not os.path.isfile(args.target_file):
        raise Exception('The target_file does not exist.'%(args.target_file))

    data_list = []
    data_reader = csv.DictReader(open(args.data_file, newline=''))
    for row in data_reader:
        data_list.append(list(row.values()))

    target_list = []
    target_reader = csv.DictReader(open(args.target_file, newline=''))
    for row in target_reader:
        target_list.append(list(row.values()))

    Xs = np.array(data_list, dtype=np.float64)
    Ys = np.array(target_list, dtype=np.float64)

    assert(Xs.shape[0] == Ys.shape[0])

    print('Total samples:', Xs.shape[0])
    totalSamples = Xs.shape[0]
    if Xs.shape[1] < args.max_components:
        args.max_components = Xs.shape[1]
    
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
    #fig.canvas.set_window_title('calibration %s/*.txt with %s into %s, %3.2f testset ratio, %d max components, %d max outlier, %s spectra'%(dir_name, excel_name, model_name, testset_ratio, max_components, max_outliers, preprocess))
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
    
    mse, minmse = pls_find_best_components_and_outliers(Xcalibrate, Ycalibrate, args.max_components, args.max_outliers)#, Xvalidate, Yvalidate)
    optimumComponents = minmse[0]+1
    optimumOutliers = minmse[1]

    y,x = np.mgrid[1:mse.shape[0]+1, 0:mse.shape[1]]
    with plt.style.context(('ggplot')):
        #ax1 = plt.subplot(211)
        ax1 = plt.subplot2grid((4,6), (1,2), rowspan=3, colspan=3)
        #ax1.margins(x=0, y=0, tight=True)
        c1 = ax1.contourf(x, y, mse, 256, cmap=plt.cm.nipy_spectral_r, origin='lower')
        ax1.plot([minmse[1], minmse[1]], [plt.axis()[2], plt.axis()[3]], '--', color='blue')
        ax1.plot([plt.axis()[0], plt.axis()[1]], [minmse[0]+1, minmse[0]+1], '--', color='blue')
        #ax1.set_xticks(np.arange(0, mse.shape[1], 1.0))
        #ax1.set_yticks(np.arange(1, mse.shape[0]+1, 1.0))
        ax1.set_xlim(0, mse.shape[1])
        ax1.set_ylim(1, mse.shape[0]+1)
        ax1.set_ylabel('Number of components', fontsize=fontSize)
        ax1.set_xlabel('Number of outliers', fontsize=fontSize)
        
        ax2 = plt.subplot2grid((4,6), (1,5), rowspan=3, sharey=ax1)
        #ax2.margins(y=0)
        components = range(1, args.max_components+1)
        ax2.plot(mse[:,minmse[1]], components, '-v', color='blue', mfc='blue')#, transform=rot+base)
        ax2.plot(mse[minmse[0], minmse[1]], components[minmse[0]], 'P', ms=10, mfc='red')#, transform=rot+base)
        #ax2.set_ylabel('Number of PLS components', fontsize=fontSize)
        ax2.set_xlabel('MSE', fontsize=fontSize)
        ax2.yaxis.tick_right()
        #ax2.set_anchor('SE')
        
        '''
        ax2.plot(components, mse[:,minmse[1]], '-v', color='blue', mfc='blue')#, transform=rot+base)
        ax2.plot(components[minmse[0]], mse[minmse[0], minmse[1]], 'P', ms=10, mfc='red')#, transform=rot+base)
        ax2.set_xlabel('Number of PLS components', fontsize=fontSize)
        ax2.set_ylabel('MSE', fontsize=fontSize)
        '''
        ax3 = plt.subplot2grid((4,6), (0,2), colspan=3, sharex=ax1)
        #ax3.margins(x=0)
        components = range(0, args.max_outliers)
        ax3.plot(components, mse[minmse[0],:], '-v', color='blue', mfc='blue')
        ax3.plot(components[minmse[1]], mse[minmse[0],minmse[1]], 'P', ms=10, mfc='red')
        #ax3.set_xlabel('Number of outliers', fontsize=fontSize)
        ax3.set_ylabel('MSE', fontsize=fontSize)
        #ax1.title('PLS')
        ax3.set_xlim(left=-1)
        ax3.xaxis.tick_top()

        #plt.colorbar(c1, ax=ax1)
        plt.draw()
        plt.pause(0.001)

    pls,outlier_index = pls_calibrate(Xcalibrate, Ycalibrate, optimumComponents, optimumOutliers)
    #_ys_ = pls.predict(preprocessX)
    #np.savetxt('test_calib.txt', _ys_, delimiter=',')
    
    Xtest = Xs[test_idx]
    Ytest = Ys[test_idx]

    Ypredict, predict_score, predict_mse, sep, rpd, bias = pls_validate(pls, Xtest, Ytest)
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
    model_io.saveModelAsJSON(pls, args.model_file)
    
    config_dict = { 'date':datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'data_file':args.data_file,
                    'target_file':args.target_file,
                    'model_file':args.model_file,
                    'config_file':args.config_file,
                    'testset_ratio':args.testset_ratio,
                    'max_components':args.max_components,
                    'max_outliers':args.max_outliers }

    with open(args.config_file, 'w') as fp:
        json.dump(config_dict, fp, indent=4, sort_keys=False)
        
    plt.show()
    
    return 0

if __name__ == '__main__':
    main(sys.argv)

    
