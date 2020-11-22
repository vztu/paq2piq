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
'''======================== parameters ================================''' 


data_name = 'ALL_COMBINED'
algo_name = 'PAQ2PIQ'
color_only = False
# mat_file = './mos_feat_files/'+data_name+'_'+algo_name+'_feats.mat'

## read KONVID_1K
data_name = 'KONVID_1K'
csv_file = '/media/ztu/Data/ClassifyQA/mos_feat_files/'+data_name+'_metadata.csv'
feats_file = 'result/'+data_name+'_'+algo_name+'_feats.mat'
try:
    df = pandas.read_csv(csv_file, skiprows=[], header=None)       
except:
    raise Exception('Read csv file error!')
array = df.values
y1 = array[1:,1]
y1 = np.array(list(y1), dtype=np.float)
# apply scaling transform
temp = np.divide(5.0 - y1, 4.0) # convert MOS do distortion
temp = -0.0993 + 1.1241 * temp # apply gain and shift produced by INSLA
temp = 5.0 - 4.0 * temp # convert distortion to MOS
y1 = temp
# read scores
X1 = scipy.io.loadmat(feats_file)['feats_mat'].squeeze()
X_score1 = []
for i in range(len(X1)):  # for each video
    frame_scores = []
    for j in range(X1[i].shape[0]):  # for each frame
        frame_scores.append(X1[i][j][0].item())  # get paq2piq frame features
    X_score1.append(sum(frame_scores) / len(frame_scores))
X_score1 = np.array(X_score1, dtype=np.float)

## read LIVE-VQC
data_name = 'LIVE_VQC'
csv_file = '/media/ztu/Data/ClassifyQA/mos_feat_files/'+data_name+'_metadata.csv'
feats_file = 'result/'+data_name+'_'+algo_name+'_feats.mat'
try:
    df = pandas.read_csv(csv_file, skiprows=[], header=None)
except:
    raise Exception('Read csv file error!')
array = df.values  
y2 = array[1:,1]
y2 = np.array(list(y2), dtype=np.float)
# apply scaling transform
temp = np.divide(100.0 - y2, 100.0) # convert MOS do distortion
temp = 0.0253 + 0.7132 * temp # apply gain and shift produced by INSLA
temp = 5.0 - 4.0 * temp # convert distortion to MOS
y2 = temp
# read scores
X2 = scipy.io.loadmat(feats_file)['feats_mat'].squeeze()
X_score2 = []
for i in range(len(X2)):  # for each video
    frame_scores = []
    for j in range(X2[i].shape[0]):  # for each frame
        frame_scores.append(X2[i][j][0].item())  # get paq2piq frame features
    X_score2.append(sum(frame_scores) / len(frame_scores))
X_score2 = np.array(X_score2, dtype=np.float)

## read YOUTUBE_UGC
data_name = 'YOUTUBE_UGC'
csv_file = '/media/ztu/Data/ClassifyQA/mos_feat_files/'+data_name+'_metadata.csv'
feats_file = 'result/'+data_name+'_'+algo_name+'_feats.mat'
try:
    df = pandas.read_csv(csv_file, skiprows=[], header=None)
except:
    raise Exception('Read csv file error!')
array = df.values  
y3 = array[1:,4]
y3 = np.array(list(y3), dtype=np.float)
# read scores
X3 = scipy.io.loadmat(feats_file)['feats_mat'].squeeze()
#### 57 grayscale videos in YOUTUBE_UGC dataset, we do not consider them for fair comparison ####
if color_only:
    gray_indices = [3,6,10,22,23,46,51,52,68,74,77,99,103,122,136,141,158,173,368,426,467,477,506,563,594,\
    639,654,657,666,670,671,681,690,697,702,703,710,726,736,764,768,777,786,796,977,990,1012,\
    1015,1023,1091,1118,1205,1282,1312,1336,1344,1380]
    gray_indices = [idx - 1 for idx in gray_indices]
    X3 = np.delete(X3, gray_indices, axis=0)
    y3 = np.delete(y3, gray_indices, axis=0)
X_score3 = []
for i in range(len(X3)):  # for each video
    frame_scores = []
    for j in range(X3[i].shape[0]):  # for each frame
        frame_scores.append(X3[i][j][0].item())  # get paq2piq frame features
    X_score3.append(sum(frame_scores) / len(frame_scores))
X_score3 = np.array(X_score3, dtype=np.float)

X_score = np.vstack((X_score1.reshape(-1,1), X_score2.reshape(-1,1), X_score3.reshape(-1,1))).squeeze()
y = np.vstack((y1.reshape(-1,1), y2.reshape(-1,1), y3.reshape(-1,1))).squeeze()

# # remove nan & inf values
good_indices = np.isfinite(X_score)
X_score = X_score[good_indices]
y = y[good_indices]


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
    # try:
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
    # except:
    #     continue

print('\n\n')
print('======================================================')
print('Average results among all repeated 80-20 holdouts:')
print('SRCC: ',np.median(SRCC_all_repeats),'( std:',np.std(SRCC_all_repeats),')')
print('KRCC: ',np.median(KRCC_all_repeats),'( std:',np.std(KRCC_all_repeats),')')
print('PLCC: ',np.median(PLCC_all_repeats),'( std:',np.std(PLCC_all_repeats),')')
print('RMSE: ',np.median(RMSE_all_repeats),'( std:',np.std(RMSE_all_repeats),')')
print('======================================================')
print('\n\n')
