#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:21:53 2018

@author: charlottepiette
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
#import seaborn as sns
#import keras.utils
import math


path3 = '/Sampled_Vectors'
path4 = '/Decoding_Results'

model_identity=[2,5,7,17,26,40,41,52,58,60,62,67,72,81,85,87,91,104,105,112,134,138,139,151,166,174,180,184,188,200,219,222,223,246,260,268,273,289,317,303,304,321,334,3383,58];


for TU in range(len(model_identity)):

    model_number = model_identity[TU]; 
    
    
    #  Number of different conditions and points
    amplitude_DBS = [200,400,600,800,1000,1200]; #100,300,500,700,900,1100] ;#, #[0,50,150,500,800];
    frequency_DBS = [110,130,150,170,190,210];

    
    for M in range(len(amplitude_DBS)):
        
        amplitude = amplitude_DBS[M]
                
        for P in range(len(frequency_DBS)):
        
            frequency = frequency_DBS[P]
            
            number_seeds = 5;
            number_splits = 5;
            
            Full_data = [];
            score_algo = np.zeros((3,number_seeds*number_splits))
            
            max_i = 300;
            
            ## Choice of stimulus
            stimulus_name=['Bertrand_Avicenne','Bertrand_Habite','Bertrand_voix','Charlie_Brandeis_fr','Charlie_enregistre','Charlie_Paris','Charlotte_Brandeis','Charlotte_College','Elodie_College','Elodie_Habite','Elodie_voix','hannah_fr_Brandeis','hannah_fr_live','hannah_fr_ma_voix','Jonathan_Brandeis','Jonathan_Brookline','Jonathan_Nice','Jonathan_Voix','Laurent_College','Laurent_Habite','Laurent_voix','Paris_VF','Simon_Brandeis','Simon_voix','Sylvie_College','Sylvie_Habite','Sylvie_voix','Voix_charlotte','Waltham_VF'];
          
            trial_number = 20;
            target = []
            m=0; 
          
            
            # voices
            stimulus = [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,17,18,19,20,21,24,25,26,27] ; 
            
            #stimulus = [2,3,4,5,6,7,8,9,10,11,13,14,17,18,20,21,22,23,26,27]  ; #meaning
    
    
            ### Three stimuli for each voice: 
            type = 2;
            for k in stimulus:     
                if type==1 :     # classic possibility : 
                    target.append(m*np.ones(trial_number))
                    m=m+1;
                elif type ==2: # recognize voices    
                    if k in [0,1,2]: # recognize Bertrand
                         target.append(1*np.ones(trial_number)) #Bertrand
                    elif k in [3,4,5]:
                         target.append(2*np.ones(trial_number)) #Charlie
                    elif k in [6,21,27]: # recognize charlotte's
                         target.append(3*np.ones(trial_number))
                    elif k in [8,9,10]: # Elodie
                         target.append(4*np.ones(trial_number))
                    elif k in [11,12,13]: # Hannah
                         target.append(5*np.ones(trial_number))
                    elif k in [14,15,17]: # Jonathan
                         target.append(6*np.ones(trial_number))
                    elif k in [18,19,20]:# recognize Laurent
                        target.append(7*np.ones(trial_number))
                    elif k in [24,25,26]:# recognize Sylvie
                        target.append(8*np.ones(trial_number))
                        
                elif type ==3 :  # recognize identical sentences
                    if k in [3,6,14]:  # Je suis à Brandeis 
                        target.append(1*np.ones(trial_number))
                    elif k in [7,8,18]:  # Je suis au College 
                        target.append(2*np.ones(trial_number))
                    elif k in [4,10,27]:  # J'enregistre ma voix
                        target.append(3*np.ones(trial_number))
                    elif k in [5,9,21]:  # J'habite à Paris 
                        target.append(4*np.ones(trial_number))
                        
                        
            target = np.hstack(target)
            ## Concatenate data
            os.chdir(path0)
            for k in stimulus:
                M1 = open('Modele_'+str(model_number)+'_Array_classifier_Bin_10_PSTH_DBS_'+str(amplitude)+'pA_'+str(frequency)+'Hz_stimulus_intensity_2_New_'+stimulus_name[k])
                Spike_count_DBS = [ map(float,line.split(',')) for line in M1 ]
                Spike_count_DBS_new = np.array(Spike_count_DBS)
                #Spike_count_DBS = [line.split(',') for line in M1 ]
                for l in range(trial_number):
                     Full_data.append(np.array(list(Spike_count_DBS[l])))                             
                #New_Spike_count_DBS = Spike_count_DBS[0:trial_number];
                #Full_data.append(New_Spike_count_DBS[:])
            Full_data = np.vstack(Full_data)
        
            
            
                          
            cm_LR_all = [];
            cm_NN_all = [];
            
            #seed_value=[3,6,5,8,9];
            seed_value=[57,67,45,12,89];
            
            f=0;
                                         
            for n in range(number_seeds):
            
                seed = seed_value[n]
                print(seed)
                np.random.seed(seed)
                kfold = StratifiedKFold(n_splits=number_splits, shuffle=True, random_state=seed)
                                  
                for train_index, test_index in kfold.split(Full_data,target):
            
                      x_train, x_test = Full_data[train_index,:],Full_data[test_index,:]
                      y_train,y_test = target[train_index],target[test_index]
                      
                      mul_lr = LogisticRegression(multi_class='multinomial', solver='newton-cg',max_iter=max_i)
                      mul_lr.fit(x_train, y_train)
                      score_algo[0,f] = mul_lr.score(x_test, y_test)*100
                      print(mul_lr.score(x_test,y_test)*100)
                      predictions = mul_lr.predict(x_test)
                      cm_LR = metrics.confusion_matrix(y_test,predictions)        
                      
                      lda = LinearDiscriminantAnalysis(solver='svd')
                      lda.fit(x_train,y_train)
                      score_algo[1,f]=lda.score(x_test,y_test)*100
                      print(lda.score(x_test,y_test)*100)
                      
                      clf = NearestCentroid(metric='euclidean',shrink_threshold=None)
                      clf.fit(x_train,y_train)
                      score_algo[2,f]=clf.score(x_test,y_test)*100
                      print(clf.score(x_test,y_test)*100)
                      predictions = clf.predict(x_test)
                      cm_NN = metrics.confusion_matrix(y_test,predictions)        
                              
                      
                      cm_NN_all.append(cm_NN); 
                      cm_LR_all.append(cm_LR);
                      
                      f=f+1; 
                      
                          
              
            ## Save data
            os.chdir('/Users/charlotte.piette/Desktop/SIMULATIONS_DBS/NATURAL_STIMULI/Scores_130Hz')
            np.savetxt('Modele_'+str(model_number)+'_score_algo_training_voices_stimuli_2x_stimulus_DBS_'+str(amplitude_DBS[M])+'pA_'+str(frequency_DBS[P])+'Hz',score_algo,delimiter=',')
              
            import scipy.io 
            scipy.io.savemat('Modele_'+str(model_number)+'_confusion_matrix_training_voices_2x_stimulus_DBS_'+str(amplitude_DBS[M])+'pA_'+str(frequency_DBS[P])+'Hz',{'cm_NN':cm_NN_all,'cm_LR':cm_LR_all}) 
