READ ME file 

* Script: Generation_networks
Script for testing and generating different configurations & regimes of activity, along with the quantification of some of their properties. 
The parameter configurations used for the rest of the article are saved in « Maps_parameters.mat » 

* Script: Networks_DBS_effects
Script for testing the effects of DBS on all regimes of activity generated with Generation_networks. 
DBS parameters are indicated in the subsection of the script « DBS_params » 

* Scripts: Simple_stimuli_Classifier_Densification or Naturalistic_stimuli_Classifier_Densification 
These scripts generates for each binned matrix of pyramidal cells responses to a simple or naturalistic stimulus a densified matrix (S = W.M, with W = weight_matrix_classifier, the random weight matrix) that will serve as input to the classifiers. 

* Scripts: Classifiers_simple_stimuli or Classifier_naturalistic_stimuli 
These scripts use the scikit-learn library to run different supervised learning machine-learning algorithms to classify the responses of pyramidal cells according to the stimulus sent as input. 
Confusion matrices as well as accuracy scores are stored.  In the case of the multinomial logistic regression algorithm, the computation time is also saved. 
For simple stimuli, nine different inputs need to be classified according to their nature. 
For naturalistic stimuli, two classifications are tested: either based on the recognition of the voices; or based on the recognition of the meaning. 

* Figure 1: script used for generating graphs of Figure 1 (+ Supplementary), based on Generation_networks and Networks_DBS_effects (using the saved ‘Map_parameters.mat’ and ‘Map_parameters_DBS_effects.mat’)

* Figure 2: two scripts Network_constant_simulations_analysis and Decoding_simple_stimuli 
In Network_constant_simulations_analysis are presented the simulations used for then calculating the SNR and discrimination capacities of the network in response to constant pulses 
Figure_2_Decoding_simple_stimuli is the code used to display figures based on the decoding accuracy (+ confusion matrices) as a function of the network parameters

* Figure 3: script for generating graphs of Figure 3 (+ Supplementary), based on the analysis of the decoding accuracy to naturalistic stimuli

