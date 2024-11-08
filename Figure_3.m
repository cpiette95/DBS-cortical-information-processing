% Figure 3: 
clear

path1='/Networks'; 
path4='/Decoding_Results';

% Simulations made on a subset of network configurations: 
model_name=[2,5,7,17,26,40,41,52,58,60,62,67,72,81,85,87,91,104,105,112,134,138,139,151,166,174,180,184,188,200,219,222,223,246,260,268,273,289,317,303,304,321,334,3383,58];

cd(path1);
load('Map_parameters.mat')
load('Map_parameters_DBS_effect.mat')

norm_firing = (mean_firing_rate_PYR_DBS-mean_firing_rate_PYR)./mean_firing_rate_PYR; 
norm_entropy = (Entropy_5ms_PYR_DBS-Entropy_5ms_PYR)./Entropy_5ms_PYR; 

entropy_subset = Entropy_5ms_PYR(model_name); norm_entropy_subset = norm_entropy(model_name); 
firing_subset = mean_firing_rate_PYR(model_name); norm_firing_subset = norm_firing(model_name); 


% Extract classifier results 
classifier='meaning'; 
cd(path4)
intensity_DBS=[0,200,400,600,800,1000,1200]; 
frequency_DBS=[110,130,150,170,190,210]; 
best_parameters_meaning = zeros(length(model_name),2);
for M=1:length(model_name)    
    for z=1:length(intensity_DBS)        
        if z==1
            frequency_DBS = [130,130,130,130,130,130];
        else
            frequency_DBS=[110,130,150,170,190,210]; 
        end
        
        for k=1:length(frequency_DBS)
            
            algo=load(strcat('Modele_',num2str(model_name(M)),'_score_algo_training_',classifier,'_12_stimuli_2x_stimulus_DBS_',num2str(intensity_DBS(z)),'pA_',num2str(frequency_DBS(k)),'Hz'));
            
            algo_LR{M,z}(k,:) = algo(1,:);
            algo_NN{M,z}(k,:) = algo(3,:);
            algo_LDA{M,z}(k,:) = algo(2,:);
            
            mean_algo_LR{M}(z,k) = mean(algo(1,:));
            mean_algo_NN{M}(z,k) = mean(algo(3,:));
            mean_algo_LDA{M}(z,k) = mean(algo(2,:));
            
            std_algo_LR{M}(z,k) = std(algo(1,:))/sqrt(25);
            std_algo_NN{M}(z,k) = std(algo(3,:))/sqrt(25);
            std_algo_LDA{M}(z,k) = std(algo(2,:))/sqrt(25);
            
        end
    end
    
    frequency_matrix=repmat([110 130 150 170 190 210],[7 1]); frequency_matrix=frequency_matrix(:);
    intensity_matrix=repmat([0 200 400 600 800 1000 1200],[6 1])'; intensity_matrix = intensity_matrix(:);
    
    
    % Extract optimal DBS parameters for maximal decoding accuracy
    [max_val,idx]=max(mean_algo_LR{M}(:)); 
    best_parameters_meaning(M,:) = [intensity_matrix(idx) frequency_matrix(idx)]; 
    energy_meaning(M) = intensity_matrix(idx)*frequency_matrix(idx)*0.002; % duration of pulse 
    best_decoding_LR_meaning(M) = max_val;
    
    [max_val,idx]=max(mean_algo_LDA{M}(:)); 
    best_parameters_LDA_meaning(M,:) = [intensity_matrix(idx) frequency_matrix(idx)]; 
    energy_LDA_meaning(M) = intensity_matrix(idx)*frequency_matrix(idx)*0.002; % duration of pulse 
    best_decoding_LDA_meaning(M) = max_val;

    [max_val,idx]=max(mean_algo_NN{M}(:)); 
    best_parameters_NN_meaning(M,:) = [intensity_matrix(idx) frequency_matrix(idx)]; 
    energy_NN_meaning(M) = intensity_matrix(idx)*frequency_matrix(idx)*0.002; % duration of pulse 
    best_decoding_NN_meaning(M) = max_val;

    
end
 

% Figure : Score without DBS
figure(); 
subplot(1,3,1); 
for k=1:length(model_name)
    score_0DBS(k) = mean_algo_LR{k}(1);
end
scatter(entropy_subset,firing_subset,60,score_0DBS,'filled')
ylabel('Firing rate (Hz)')
xlabel('Entropy (bits)')
caxis([25 40])
subplot(1,3,2); 
for k=1:length(model_name)
    score_0DBS(k) = mean_algo_LDA{k}(1);
end
scatter(entropy_subset,firing_subset,60,score_0DBS,'filled')
ylabel('Firing rate (Hz)')
xlabel('Entropy (bits)')
caxis([25 40])
subplot(1,3,3); 
for k=1:length(model_name)
    score_0DBS(k) = mean_algo_NN{k}(1);
end
scatter(entropy_subset,firing_subset,60,score_0DBS,'filled')
ylabel('Firing rate (Hz)')
xlabel('Entropy (bits)')
caxis([25 35])


figure(); 
subplot(1,2,1); 
for k=1:length(model_name)
    score_0DBS(k) = mean_algo_LR{k}(1);
end
scatter(entropy_subset,firing_subset,60,score_0DBS,'filled')
ylabel('Firing rate (Hz)')
xlabel('Entropy (bits)')
caxis([25 55]) 
subplot(1,2,2); 
scatter(entropy_subset,firing_subset,60,best_decoding_LR,'filled')
ylabel('Firing rate (Hz)')
xlabel('Entropy (bits)')
caxis([25 55])



classifier='voices'; 
line_ordonne = 100/3; 
best_parameters_voices = zeros(length(model_name),2);
for M=1:length(model_name)    
    for z=1:length(intensity_DBS)        
        if z==1
            frequency_DBS = [130,130,130,130,130,130];
        else
            frequency_DBS=[110,130,150,170,190,210]; 
        end
        
        for k=1:length(frequency_DBS)
            
            algo=load(strcat('Modele_',num2str(model_name(M)),'_score_algo_training_',classifier,'_12_stimuli_2x_stimulus_DBS_',num2str(intensity_DBS(z)),'pA_',num2str(frequency_DBS(k)),'Hz'));
            
            algo_LR{M,z}(k,:) = algo(1,:);
            algo_NN{M,z}(k,:) = algo(3,:);
            algo_LDA{M,z}(k,:) = algo(2,:);
            
            mean_algo_LR{M}(z,k) = mean(algo(1,:));
            mean_algo_NN{M}(z,k) = mean(algo(3,:));
            mean_algo_LDA{M}(z,k) = mean(algo(2,:));
            
            std_algo_LR{M}(z,k) = std(algo(1,:))/sqrt(25);
            std_algo_NN{M}(z,k) = std(algo(3,:))/sqrt(25);
            std_algo_LDA{M}(z,k) = std(algo(2,:))/sqrt(25);
            
        end
    end
    
    frequency_matrix=repmat([110 130 150 170 190 210],[7 1]); frequency_matrix=frequency_matrix(:);
    intensity_matrix=repmat([0 200 400 600 800 1000 1200],[6 1])'; intensity_matrix = intensity_matrix(:);
    
    [max_val,idx]=max(mean_algo_LR{M}(:)); 
    best_parameters_voices(M,:) = [intensity_matrix(idx) frequency_matrix(idx)]; 
    energy_voices(M) = intensity_matrix(idx)*frequency_matrix(idx)*0.002; % duration of pulse 
    best_decoding_voices_LR(M) = max_val;

    [max_val,idx]=max(mean_algo_LDA{M}(:)); 
    best_parameters_voices_LDA(M,:) = [intensity_matrix(idx) frequency_matrix(idx)]; 
    energy_voices_LDA(M) = intensity_matrix(idx)*frequency_matrix(idx)*0.002; % duration of pulse 
    best_decoding_voices_LDA(M) = max_val;

    [max_val,idx]=max(mean_algo_NN{M}(:)); 
    best_parameters_voices_NN(M,:) = [intensity_matrix(idx) frequency_matrix(idx)]; 
    energy_voices_NN(M) = intensity_matrix(idx)*frequency_matrix(idx)*0.002; % duration of pulse 
    best_decoding_voices_NN(M) = max_val;

end


% Energy/current
current_map = repmat(intensity_DBS',[1 6]).*repmat(frequency_DBS,[7 1])*0.002;
figure(); 
for k=1:20
    subplot(2,10,k)
    %imagesc(mean_algo_LR{k})
    %title(num2str(model_name(k)))
    %caxis([25 50])
    
    LR_decoding = mean_algo_LR{k}(:);
    current_map_1D = current_map(:); 
    
    scatter(current_map_1D,LR_decoding)

end

figure(); scatter(norm_entropy_subset,energy)
[r,p]=corrcoef(norm_firing_subset,energy)
[r,p]=corrcoef(norm_entropy_subset,energy)



figure(); 
subplot(1,2,1); 
scatter(entropy_subset,firing_subset,90,energy,'filled')
ylabel('Firing rate')
xlabel('Entropy')
colorbar
subplot(1,2,2); hold on; scatter(norm_firing_subset,norm_entropy_subset,90,energy,'filled')
plot([0 0],[-1 1],'k-')
plot([-1 0.5],[0 0],'k-')
ylim([-0.8 0.8])
xlim([-1 0.2])
xlabel('Delta firing rate')
ylabel('Delta entropy')
colorbar


figure(); scatter(best_parameters(:,1),best_parameters(:,2),60,entropy(model_name),'filled')

figure(); 
subplot(1,2,1); 
yyaxis left ; hold on; 
scatter(energy,energy_LDA); 
ylim([50 300])
ylabel('LDA')
yyaxis right
hold on; scatter(energy,energy_NN)
plot([50 300],[50 300],'k-')
ylim([50 300])
ylabel('NN')
xlabel('LR')

[r,p]=corrcoef(energy,energy_LDA) ; %r=0.9226
[r,p]=corrcoef(energy,energy_NN); %r=0.8104

subplot(1,2,2); 
yyaxis left ; hold on; 
scatter(energy,energy_voices); 
ylim([50 300])
xlabel('LR meaning')
ylabel('LR voices')
plot([50 300],[50 300],'k-')
[r,p]=corrcoef(energy,energy_voices) ; %r=0.8091, p<0.001

figure(); subplot(1,3,1)
scatter(parameters(model_name,3),energy)
[r,p]=corrcoef(parameters(model_name,3),energy); %r=0.8098, p<0.001

xlabel('Iext on Pyr')
ylabel('Optimal energy delivered')
subplot(1,3,2); 
scatter(parameters(model_name,4),energy)
[r,p]=corrcoef(parameters(model_name,4),energy) % r=-0.1489, p=0.531
xlabel('We on PV')
ylabel('Optimal energy delivered')
subplot(1,3,3); 
scatter(parameters(model_name,5),energy) ; %r=0.1607 ; p=0.4981,
xlabel('Wi on Pyr')
ylabel('Optimal energy delivered')
[r,p]=corrcoef(parameters(model_name,5),energy)

figure(); subplot(1,3,1)
scatter(parameters(model_name,3),energy_voices)
[r,p]=corrcoef(parameters(model_name,3),energy_voices) % r=0.8355 ; p=<0.001
xlabel('Iext on Pyr')
ylabel('Optimal energy delivered')
subplot(1,3,2); 
scatter(parameters(model_name,4),energy_voices)
[r,p]=corrcoef(parameters(model_name,4),energy_voices); % r=-0.2396, p=0.3089
xlabel('We on PV')
ylabel('Optimal energy delivered')
subplot(1,3,3); 
scatter(parameters(model_name,5),energy_voices)
xlabel('Wi on Pyr')
ylabel('Optimal energy delivered')
[r,p]=corrcoef(parameters(model_name,5),energy_voices) % r=0.1647, p=0.4878












 