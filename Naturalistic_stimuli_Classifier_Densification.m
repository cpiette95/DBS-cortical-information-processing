%% Reduce the dimensionality of the matrix + densification of the sparse data

path2 = '/Saved_results'; % path where you saved the simulations
path3 = '/Sampled_vectors';

number_bins = 100; % depends on the length of the stimulus
number_trials = 20; % number of trials
number_cells=800;
number_repetitions=100; 

weight_matrix = load('weight_matrix_classifier');

intensity_range=[0,200,400,600,800,1000,1200]; 
frequency_range=[110,130,150,170,190,210];


name_input = {'New_Bertrand_Avicenne','New_Bertrand_Habite','New_Bertrand_voix','New_Charlie_Brandeis_fr','New_Charlie_Paris','New_Charlie_enregistre','New_Charlotte_Brandeis','New_Charlotte_College','New_Waltham_VF','New_Paris_VF','New_Voix_charlotte','New_Elodie_College','New_Elodie_Habite','New_Elodie_voix','New_hannah_fr_Brandeis','New_hannah_fr_live','New_hannah_fr_ma_voix','New_Jonathan_Brandeis','New_Jonathan_Brookline','New_Jonathan_Nice','New_Jonathan_Voix','New_Laurent_College','New_Laurent_Habite','New_Laurent_voix','New_Simon_Brandeis','New_Simon_voix','New_Sylvie_College','New_Sylvie_Habite','New_Sylvie_voix'};

% Simulations made on a subset of network configurations: 
model_number=[2,5,7,17,26,40,41,52,58,60,62,67,72,81,85,87,91,104,105,112,134,138,139,151,166,174,180,184,188,200,219,222,223,246,260,268,273,289,317,303,304,321,334,3383,58];


for K=1:length(model_number)

    model= model_number(K);  
    
    for Z=1:length(intensity_range)
        disp(Z)
                
        for P=1:length(frequency_range)
            
            for k=1:length(name_input)
                cd(path2)
                M1 = importdata(strcat('Modele_',num2str(model),'_V_PYR_binned_PSTH_DBS_',num2str(intensity_range(Z)),'pA_',num2str(frequency_range(P)),'Hz_stimulus_intensity_2_New_',name_input{k}));
                
                if max(M1(:))==0
                    disp('Warning')
                    disp(Z)
                    disp(P)
                end
                    
                M1_bis = zeros(number_trials*number_cells,number_bins); % (first 200 cells of each 1:800 series correspond to the receptor cells)
                for w=1:number_trials
                    M1_small=reshape(M1(w,:),[number_cells,2*number_bins]); % reshape of the stored matrix
                    r=1; M1_new=[];
                    for t=1:100
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
                dlmwrite(strcat('Modele_',num2str(model),'_Array_classifier_Bin_10_PSTH_DBS_',num2str(intensity_range(Z)),'pA_',num2str(frequency_range(P)),'Hz_stimulus_intensity_2_New_',name_input{k}),Sampled_new_vector_DBS);
                
            end
        end
        
    end
end

