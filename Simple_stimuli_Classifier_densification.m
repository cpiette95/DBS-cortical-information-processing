%% Reduce the dimensionality of the matrix + densification of the sparse data

path2 = '/Saved_results'; % path where you saved the simulations
path3 = '/Sampled_vectors';

number_bins = 50; % depends on the length of the stimulus 
number_trials = 60; % number of trials
number_cells=800;
number_repetitions=100; % how much you want to "densify": here I started with a 800 x 50 matrix to end up with 100 x 50 matrix

weight_matrix = load('weight_matrix_classifier');

trial_maps=1:370; % full network configurations investigated
for Z=trial_maps
    
    for A=1:3
        
        if A==1
           name_input = 'classic_OU'; % type of inputs
           inputs = [3,7,10]; % name of the inputs
           
        elseif A==2
            name_input ='classic_ramp' ; 
            inputs=[7,8,14,15]; 
            
        elseif A==3
            name_input = 'classic_constant'; 
            inputs=[5,6];
        end
 
    
    for k=inputs
        cd(path2)
        M1 = importdata(strcat('V_PYR_binned_PSTH_DBS_130Hz_Params_',num2str(trial_maps(Z)),'_',name_input,'_',num2str(k)));
        M1_bis = zeros(number_trials*number_cells,number_bins); % (first 200 cells of each 1:800 series correspond to the receptor cells)
        for w=1:number_trials
            M1_small=reshape(M1(w,:),[number_cells,2*number_bins]); % reshape the stored matrix
            r=1; M1_new=[];
            for t=1:number_bins
                M1_new(:,t) = sum(M1_small(:,r:r+1),2);
                r=r+2;
            end
            M1_bis(1+(w-1)*number_cells:number_cells+(w-1)*number_cells,:) = M1_new;
            
        end
        Sampled_new_vector_DBS = zeros(number_trials,number_repetitions,number_bins);
        for w=1:number_trials
            trial_matrix = M1_bis(1+(w-1)*number_cells:number_cells+(w-1)*number_cells,:);
            Sampled_new_vector_DBS(w,:,:)=weight_matrix*trial_matrix;
        end
        
        cd(path3) 
        dlmwrite(strcat('Array_classifier_Bin_10_PSTH_DBS_130Hz_Params_',num2str(trial_maps(Z)),'_',name_input,'_',num2str(k)),Sampled_new_vector_DBS);
        
    end
    end
end
