# -*- coding: utf-8 -*-
"""
This script shows how to apply 80-20 holdout train and validate regression model to predict
MOS from the features computed with compute_features_example.m

Author: Zhengzhong Tu
"""
# Load libraries
import warnings
import time
import pandas
import math
import random as rnd
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.stats
import scipy.io
from scipy.optimize import curve_fit
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
# ignore all warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# Here starts the main part of the script
#
data_name = 'YOUTUBE_UGC'
algo_name = 'PAQ2PIQ'
csv_file = '/media/ztu/Data/ClassifyQA/mos_feat_files/'+data_name+'_metadata.csv'
# mat_file = './mos_feat_files/'+data_name+'_'+algo_name+'_feats.mat'
feats_file = 'result/'+data_name+'_'+algo_name+'_feats.mat'
try:
    df = pandas.read_csv(csv_file, skiprows=[], header=None)       
except:
    raise Exception('Read csv file error!')
array = df.values
if data_name == "LIVE_VQC" or data_name == "KONVID_1K":
    y = array[1:,1]
elif data_name == "YOUTUBE_UGC":
    y = array[1:,4]
y = np.array(list(y), dtype=np.float)

# read scores
X = scipy.io.loadmat(feats_file)['feats_mat'].squeeze()
X_score = []
for i in range(len(X)):  # for each video
    frame_scores = []
    for j in range(X[i].shape[0]):  # for each frame
        frame_scores.append(X[i][j][0].item())  # get paq2piq frame features
    X_score.append(sum(frame_scores) / len(frame_scores))

X_score = np.array(X_score, dtype=np.float)
print(X_score.shape)
'''======================== parameters end ===========================''' 
print("Evaluating algorithm {} on dataset {} ...".format(algo_name, data_name))

MOS_predicted_all_repeats = []
y_all_repeats = []
model_params_all_repeats = []
PLCC_all_repeats = []
SRCC_all_repeats = []
KRCC_all_repeats = []
RMSE_all_repeats = []
popt_all_repeats = []

for i in range(1,101):
    try:
        print(i,'th repeated 80-20 hold out validation')
        t0 = time.time()

        # Split data to test and validation sets randomly   
        test_size = 0.2
        _, X_test, _, y_test = \
            model_selection.train_test_split(X_score, y, test_size=test_size, random_state=math.ceil(8.8*i))

        def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
            logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
            yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
            return yhat
        y_test = np.array(list(y_test), dtype=np.float)
        try:
            # logistic regression
            beta = [np.max(y_test), np.min(y_test), np.mean(X_test), 0.5]
            popt, _ = curve_fit(logistic_func, X_test, \
                y_test, p0=beta, maxfev=100000000)
            X_test_logistic = logistic_func(X_test, *popt)
        except:
            raise Exception('Fitting logistic function time-out!!')
        plcc_tmp = scipy.stats.pearsonr(y_test, X_test_logistic)[0]
        rmse_tmp = np.sqrt(mean_squared_error(y_test, X_test_logistic))
        srcc_tmp = scipy.stats.spearmanr(y_test, X_test)[0]
        krcc_tmp = scipy.stats.kendalltau(y_test, X_test)[0]
        
        PLCC_all_repeats.append(plcc_tmp)
        RMSE_all_repeats.append(rmse_tmp)
        SRCC_all_repeats.append(srcc_tmp)
        KRCC_all_repeats.append(krcc_tmp)

        # plot figs
        print('======================================================')
        print('======================================================')
        print('SRCC_test: ', srcc_tmp)
        print('KRCC_test: ', krcc_tmp)
        print('PLCC_test: ', plcc_tmp)
        print('RMSE_test: ', rmse_tmp)
        print('======================================================')

        print(' -- ' + str(time.time()-t0) + ' seconds elapsed...\n\n')
    except:
        continue

print('\n\n')
print('======================================================')
print('Average results among all repeated 80-20 holdouts:')
print('SRCC: ',np.median(SRCC_all_repeats),'( std:',np.std(SRCC_all_repeats),')')
print('KRCC: ',np.median(KRCC_all_repeats),'( std:',np.std(KRCC_all_repeats),')')
print('PLCC: ',np.median(PLCC_all_repeats),'( std:',np.std(PLCC_all_repeats),')')
print('RMSE: ',np.median(RMSE_all_repeats),'( std:',np.std(RMSE_all_repeats),')')
print('======================================================')
print('\n\n')
