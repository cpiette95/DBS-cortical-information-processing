%% Decoding panels of Figure 2 and Supplementary Figures

clear

path1 = '/Networks';
path3 = '/Saved_Results';
path4 ='/Decoding_Results'; Simple_stimuli_simulations

cd(path1)
load('Map_parameters.mat')
entropy = Entropy_5ms_PYR; 
firing = mean_firing_rate_PYR; 

load('Map_parameters_DBS_effect.mat')
firing_DBS = mean_firing_rate_PYR_DBS; 
entropy_DBS = Entropy_5ms_PYR_DBS; 

[firing_rate_PYR_DBS_ordered,index_order] = sortrows(firing,1);
new_parameters_ordered=parameters(index_order,:);
entropy_ordered=entropy(index_order);
ordered_trials = index_order; 


cd(path4)

% DBS OFF
for Z=1:length(parameters)
    
        data=load(strcat('score_algo_seeds_5_splits_5_training_DBS_OFF_map_trial_',num2str(Z)));
        mean_score_LR(Z,1) = mean(data(1,:));
        mean_score_LDA(Z,1) = mean(data(2,:));
        mean_score_NN(Z,1) = mean(data(3,:));
        std_score_LR(Z) = std(data(1,:));
        std_score_LDA(Z) = std(data(2,:));
        std_score_NN(Z) = std(data(3,:));

        data=load(strcat('computation_time_training_DBS_OFF_map_trial_',num2str(Z)));
        computation_time_LR(Z,1) = mean(data(1,:));
        computation_time_LDA(Z,1) = mean(data(2,:));
        computation_time_NN(Z,1) = mean(data(3,:));
        std_time_LR(Z) = std(data(1,:));
        std_time_LDA(Z) = std(data(2,:));
        std_time_NN(Z) = std(data(3,:));
    
end


% Figure 2C: 2D map with decoding accuracy
figure();
subplot(1,2,1); hold on ;
scatter(entropy,firing,60,mean_score_LR,'filled')
title("decoding accuracy MLR")
ylim([0 9])
subplot(1,2,2); hold on ;
scatter(entropy,firing,60,computation_time_LR,'filled')
title("computation time MLR")
caxis([0 30])
ylim([0 9])



% Example confusion matrices (Figure 2C): 
cd(path4)
figure();
load('confusion_matrix_medium_stim_134.mat')
 subplot(1,3,1) 
 imagesc(LR)
caxis([0 12])
load('confusion_matrix_medium_stim_26.mat')
 subplot(1,3,2) 
 imagesc(LR)
caxis([0 12])
load('confusion_matrix_medium_stim_52.mat')
 subplot(1,3,3) 
 imagesc(LR)
caxis([0 12])
figure();
load('DBS_130Hz_confusion_matrix_medium_stim_134.mat')
subplot(1,3,1) 
imagesc(LR)
caxis([0 12])
load('DBS_130Hz_confusion_matrix_medium_stim_26.mat')
 subplot(1,3,2) 
 imagesc(LR)
caxis([0 12])
load('DBS_130Hz_confusion_matrix_medium_stim_52.mat')
 subplot(1,3,3) 
 imagesc(LR)
caxis([0 12])




new_mean_score_LR = mean_score_LR(index_order);
new_mean_score_NN = mean_score_NN(index_order);
new_mean_score_LDA = mean_score_LDA(index_order);
new_std_score_LR = std_score_LR(index_order);
new_std_score_NN = std_score_NN(index_order);
new_std_score_LDA = std_score_LDA(index_order);


matrix=[10 37]; 
ordered_trials = reshape(ordered_trials,matrix)'; 
new_mean_firing_rate_PYR_DBS=reshape(firing_rate_PYR_DBS_ordered,matrix)';
new_entropy_ordered=reshape(entropy_ordered,matrix)';
new_mean_score_LR=reshape(new_mean_score_LR,matrix)';
new_mean_score_NN=reshape(new_mean_score_NN,matrix)';
new_mean_score_LDA=reshape(new_mean_score_LDA,matrix)';
new_std_score_LR=reshape(new_std_score_LR,matrix)';
new_std_score_NN=reshape(new_std_score_NN,matrix)';
new_std_score_LDA=reshape(new_std_score_LDA,matrix)';

new_ordered_trials=zeros(matrix(2),matrix(1));
new_mean_score_LR_bis = zeros(matrix(2),matrix(1)); new_mean_score_LDA_bis = zeros(matrix(2),matrix(1)); new_mean_score_NN_bis = zeros(matrix(2),matrix(1)); 
new_std_score_LR_bis = zeros(matrix(2),matrix(1)); new_std_score_LDA_bis = zeros(matrix(2),matrix(1)); new_std_score_NN_bis = zeros(matrix(2),matrix(1)); 
new_entropy_ordered_bis=zeros(matrix(2),matrix(1));new_mean_firing_rate_PYR_DBS_bis=zeros(matrix(2),matrix(1));
for k=1:matrix(2)
    [~,idx] = sort(new_entropy_ordered(k,:));
    
    new_entropy_ordered_bis(k,:)=new_entropy_ordered(k,idx);
    new_mean_firing_rate_PYR_DBS_bis(k,:) = new_mean_firing_rate_PYR_DBS(k,idx); 
    
    new_mean_score_LR_bis(k,:) = new_mean_score_LR(k,idx); 
    new_mean_score_LDA_bis(k,:) = new_mean_score_LDA(k,idx); 
    new_mean_score_NN_bis(k,:) = new_mean_score_NN(k,idx); 
    new_std_score_LR_bis(k,:) = new_std_score_LR(k,idx); 
    new_std_score_LDA_bis(k,:) = new_std_score_LDA(k,idx); 
    new_std_score_NN_bis(k,:) = new_std_score_NN(k,idx); 
    
    new_ordered_trials(k,:)=ordered_trials(k,idx); 
    
end

figure(); colormap jet
subplot(1,2,1); 
imagesc(new_mean_firing_rate_PYR_DBS_bis)
colorbar
caxis([0 7])
subplot(1,2,2); 
imagesc(new_entropy_ordered_bis)
colorbar

figure(); 
subplot(1,3,1); 
imagesc(new_mean_score_LR_bis)
title('LR')
caxis([15 80])
subplot(1,3,2); 
imagesc(new_mean_score_NN_bis)
title('NN')
caxis([15 80])
subplot(1,3,3); 
imagesc(new_mean_score_LDA_bis)
title('LDA')
caxis([15 80])

figure(); hold on ;
scatter(entropy,firing,60,mean_score_LR,'filled')
caxis([30 75])
ylim([0 9])


% Addition of DBS 
cd(path4)
for Z=1:length(parameters)
    
        data=load(strcat('score_algo_seeds_5_splits_5_training_DBS_130Hz_map_trial_',num2str(Z)));
        mean_score_LR_DBS(Z,1) = mean(data(1,:));
        mean_score_LDA_DBS(Z,1) = mean(data(2,:));
        mean_score_NN_DBS(Z,1) = mean(data(3,:));
        std_score_LR_DBS(Z) = std(data(1,:));
        std_score_LDA_DBS(Z) = std(data(2,:));
        std_score_NN_DBS(Z) = std(data(3,:));

        data=load(strcat('computation_time_training_DBS_130Hz_map_trial_',num2str(Z)));
        computation_time_LR_DBS(Z,1) = mean(data(1,:));
        computation_time_LDA_DBS(Z,1) = mean(data(2,:));
        computation_time_NN_DBS(Z,1) = mean(data(3,:));
        std_time_LR_DBS(Z) = std(data(1,:));
        std_time_LDA_DBS(Z) = std(data(2,:));
        std_time_NN_DBS(Z) = std(data(3,:));

end


% Figure S3A
figure(); final_index=370; 
colormap jet
subplot(1,3,1); 
scatter(mean_score_LR(1:final_index),mean_score_LR_DBS(1:final_index),60,entropy(1:final_index),'filled')
hold on; plot([20 90],[20 90],'k-')
title('LR')
subplot(1,3,2); 
scatter(mean_score_NN(1:final_index),mean_score_NN_DBS(1:final_index),60,entropy(1:final_index),'filled')
hold on; plot([10 90],[10 90],'k-')
title('NN')
subplot(1,3,3); 
scatter(mean_score_LDA(1:final_index),mean_score_LDA_DBS(1:final_index),60,entropy(1:final_index),'filled')
hold on; plot([20 90],[20 90],'k-')
title('LDA')
colorbar
% Figure S3 C
figure(); 
scatter(entropy,firing,60,computation_time_LR,'filled')
figure(); 
scatter(entropy,firing,60,computation_time_LR-computation_time_LR_DBS,'filled')
ylim([0 9])


figure(); 
colormap jet
subplot(1,3,1); 
scatter(mean_score_LR(1:final_index),mean_score_LR_DBS(1:final_index),60,firing(1:final_index),'filled')
hold on; plot([20 90],[20 90],'k-')
title('LR')
caxis([0 7])
subplot(1,3,2); 
scatter(mean_score_NN(1:final_index),mean_score_NN_DBS(1:final_index),60,firing(1:final_index),'filled')
hold on; plot([10 90],[10 90],'k-')
title('NN')
caxis([0 7])
subplot(1,3,3); 
scatter(mean_score_LDA(1:final_index),mean_score_LDA_DBS(1:final_index),60,firing(1:final_index),'filled')
hold on; plot([20 90],[20 90],'k-')
title('LDA')
colorbar
caxis([0 7])


% Figure 2D
figure(); 
subplot(1,2,1); hold on ;
scatter(firing_DBS-firing,mean_score_LR_DBS-mean_score_LR,60,firing,'filled')
caxis([0 7])
plot([-4 6],[0 0],'k-')
plot([0 0],[-20 40],'k-')
colorbar
xlabel('Delta firing rate')
ylabel('Delta accuracy')
subplot(1,2,2); hold on; 
scatter(entropy_DBS-entropy,mean_score_LR_DBS-mean_score_LR,60,entropy,'filled')
xlabel('Delta entropy')
ylabel('Delta accuracy')
colormap jet 
colorbar
plot([0 0],[-20 40],'k-')
plot([-2 1.5],[0 0],'k-')



figure(); 
subplot(1,2,1); hold on ;
scatter((firing_DBS-firing)./firing,mean_score_LR_DBS-mean_score_LR,60,firing,'filled')
caxis([0 7])
plot([-1 1],[0 0],'k-')
plot([0 0],[-20 40],'k-')
xlabel('Delta firing rate')
ylabel('Delta accuracy')
colorbar
subplot(1,2,2); hold on; 
scatter((entropy_DBS-entropy)./entropy,mean_score_LR_DBS-mean_score_LR,60,entropy,'filled')
xlabel('Delta entropy')
ylabel('Delta accuracy')
colormap jet 
colorbar
plot([0 0],[-20 40],'k-')
plot([-1 1],[0 0],'k-')


%% Supplementary Figure 4

% Construction of a response matrix
for Z=1:length(parameters)
    
   response_matrix = zeros(N_PYR,number_bins,number_trials,number_stimuli); 
   u=1; 
   
    for M=1:3
        
        if M==1
            name_input = 'classic_constant'; % type of inputs
            inputs = [5,6]; % name of the inputs
        elseif M==2
            name_input = 'classic_OU'; % type of inputs
            inputs = [3,7,10]; % name of the inputs
        elseif M==3
            name_input = 'classic_ramp'; % type of inputs
            inputs = [7,8,14,15]; % name of the inputs
        end
        
       
        for k=inputs 
            cd(path3)
            M1 = importdata(strcat('V_PYR_binned_PSTH_DBS_130Hz_Params_',num2str(parameters(Z)),'_',name_input,'_',num2str(k)));
            for t=1:N_PYR
            % for each cell t: 
                a=[0:(number_bins-1)]*number_cells+t;
                response_matrix(t,:,:,u) = M1(:,a)'; % for a given cell on a given stimulus
            end
            u=u+1; 
        end      
    end
    
    save(strcat('Response_matrix_all_',num2str(Z)),'response_matrix')

end


% Pearson correlation coefficient 
dt = 0.05;
stimulus_duration = 500;
Bin_Length=5; 

cd('/Simple_stimuli')
a=load('OU_input_3');
stimulus(1,:) = downsample(a(1:10000),100); 
a=load('OU_input_7');
stimulus(2,:) = downsample(a(1:10000),100); 
a=load('OU_input_10');
stimulus(3,:) = downsample(a(1:10000),100); 
stimulus(4,:) = linspace(0,70,stimulus_duration/Bin_Length);
stimulus(5,:) = linspace(0,60,stimulus_duration/Bin_Length);
stimulus(6,:) = linspace(70,0,stimulus_duration/Bin_Length);
stimulus(7,:) = linspace(60,0,stimulus_duration/Bin_Length);

cd(path3)
for Z=1:length(parameters)
    
    load(strcat('Response_matrix_all_',num2str(Z)))

    for w=1:7
        
        stim = stimulus(w,:);

        for o=1:60
            receiver_response = response_matrix(1:200,:,o,w+2); % attention arrangement différent dans response_matrix des stimuli: d'abord constant puis OU, puis ramp
            all_response = response_matrix(1:800,:,o,w+2); % attention arrangement différent dans response_matrix des stimuli: d'abord constant puis OU, puis ramp
            signal = corrcoef(sum(receiver_response(:,2:end-1),1),stim(2:end-1));
            Corr_1{Z}(w,o) = signal(1,2);
            signal = corrcoef(sum(all_response(:,2:end-1),1),stim(2:end-1));
            Corr_2{Z}(w,o) = signal(1,2);
        end
        
        av_corr_1_mean(Z,w) = nanmean(Corr_1{Z}(w,:));
        av_corr_2_mean(Z,w) = nanmean(Corr_2{Z}(w,:));
        
    end
end

