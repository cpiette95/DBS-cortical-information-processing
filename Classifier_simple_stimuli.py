#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Classification algorithms for simple stimuli
"""
import os
import numpy as np
import random 

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

import matplotlib.pyplot as plt
from sklearn import metrics
import math
import time 

path3 = '/Sampled_Vectors'
path4 = '/Decoding_Results'

number_seeds=5;

seed=[76,21,32,14,8]; 



for o in range(370):  #done on all 370 network configurations 
    z=o;

    #  Number of different conditions and points
    number_splits = 5;
    Full_data = [];
    score_algo = np.zeros((3,number_seeds*number_splits))
    execution_time = np.zeros((3,number_seeds*number_splits))
    max_i = 100; 
    
    # Second trial: 3 constant, 5 ramp, 6 OU
    stimulus_name  = ['constant','constant','ramp','ramp','ramp','ramp','OU','OU','OU'] 
    stimulus_index = [5,6,7,8,14,15,3,7,10];
    
    stimulus=np.arange(len(stimulus_index))
    

    trial_number = 60;
    target = []
    m=0;
    for k in stimulus:
            target.append(m*np.ones(trial_number))
            m=m+1;
    target = np.hstack(target)
    
    ## Concatenate data
    os.chdir(path3)
    for k in range(len(stimulus_index)):
        M1 = open('Array_classifier_Bin_10_PSTH_Params_'+str(z)+'_classic_'+stimulus_name[k]+'_'+str(stimulus_index[k]))
        Spike_count_DBS = [ map(float,line.split(',')) for line in M1 ]
        Spike_count_DBS_new = np.array(Spike_count_DBS)
        #Spike_count_DBS = [line.split(',') for line in M1 ]
        for l in range(trial_number):
             Full_data.append(np.array(list(Spike_count_DBS[l])))                             
        #New_Spike_count_DBS = Spike_count_DBS[0:trial_number];
        #Full_data.append(New_Spike_count_DBS[:])
    Full_data = np.vstack(Full_data)



    cm_LR = [];
    cm_NN = [];
    
    
    f=0;
                                 
    for n in range(number_seeds):
    
    
        print(seed[n])
        np.random.seed(seed[n])
        kfold = StratifiedKFold(n_splits=number_splits, shuffle=True, random_state=seed[n])
                          
        for train_index, test_index in kfold.split(Full_data,target):

              x_train, x_test = Full_data[train_index,:],Full_data[test_index,:]
              y_train,y_test = target[train_index],target[test_index]
              
              
              # Tested on three different algorithms: 
              
              mul_lr = LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter=max_i)
              start_time = time.time()
              mul_lr.fit(x_train, y_train)
              execution_time[0,f] = time.time()-start_time; 
              score_algo[0,f] = mul_lr.score(x_test, y_test)*100
              print(mul_lr.score(x_test,y_test)*100)
              params_mulr = mul_lr.get_params(deep=True)
              predictions = mul_lr.predict(x_test)
              cm_LR = metrics.confusion_matrix(y_test,predictions)        
              
              
              lda = LinearDiscriminantAnalysis(solver='svd')
              start_time = time.time()
              lda.fit(x_train,y_train)
              execution_time[1,f] = time.time()-start_time; 
              score_algo[1,f]=lda.score(x_test,y_test)*100
              print(lda.score(x_test,y_test)*100)
              
              clf = NearestCentroid(metric='euclidean',shrink_threshold=None)
              start_time = time.time()
              clf.fit(x_train,y_train)
              execution_time[2,f] = time.time()-start_time; 
              score_algo[2,f]=clf.score(x_test,y_test)*100
              print(clf.score(x_test,y_test)*100)
              predictions = clf.predict(x_test)
              cm_NN = metrics.confusion_matrix(y_test,predictions)        
              
              f=f+1; 
              
  
    ## Save data
    os.chdir(path4)
    np.savetxt('score_algo_seeds_'+str(number_seeds)+'_splits_'+str(number_splits)+'_training_DBS_130Hz_map_trial_'+str(z),score_algo,delimiter=',')
    import scipy.io 
    scipy.io.savemat('confusion_matrix_DBS_130Hz_map_trial_'+str(z)+'.mat',{'LR':cm_LR, 'NN' : cm_NN})        
    np.savetxt('computation_time_training_DBS_130Hz_map_trial_'+str(z),execution_time,delimiter=',')

