%% Analysis of the results with/without DBS (Figure 1)

load('Map_parameters')
remove_trials=find(parameters(:,1)==0 | mean_firing_rate_PYR>12); % removed trials from instable network regimes
Entropy_5ms_PYR(remove_trials)=[]; 
mean_firing_rate_PV(remove_trials)=[]; 
mean_firing_rate_PYR(remove_trials)=[]; 
mean_firing_rate_SOM(remove_trials)=[]; 
mean_phase(remove_trials,:)=[]; 
mean_proportion_PYR_inside_peaks(remove_trials)=[]; 
mean_proportion_PYR_outside_peaks(remove_trials)=[]; 
Moving_Proba_spike_PV_PSTH_Bin5(remove_trials(1:end-1),:)=[]; 
Moving_Proba_spike_PYR_PSTH_Bin5(remove_trials,:)=[]; 
Number_peaks(remove_trials)=[]; 
Period_peaks(remove_trials)=[]; 
Variance_Spike_Proba_PYR(remove_trials)=[]; 
parameters(remove_trials,:)=[]; 

% Sort according to firing rate
[new_mean_firing_rate_PYR,index_order] = sortrows(mean_firing_rate_PYR,1);
new_parameters=parameters(index_order,:);
Entropy_5ms_PYR=Entropy_5ms_PYR(index_order);
Variance_Spike_Proba_PYR=Variance_Spike_Proba_PYR(index_order);
mean_proportion_PYR_inside_peaks=mean_proportion_PYR_inside_peaks(index_order);
mean_proportion_PYR_outside_peaks=mean_proportion_PYR_outside_peaks(index_order);
Period_peaks=Period_peaks(index_order);

figure(); 
subplot(2,1,1); hold on; 
scatter(Variance_Spike_Proba_PYR(mean_proportion_PYR_outside_peaks>0.01),Entropy_5ms_PYR(mean_proportion_PYR_outside_peaks>0.01))
scatter(Variance_Spike_Proba_PYR(mean_proportion_PYR_outside_peaks<0.01),Entropy_5ms_PYR(mean_proportion_PYR_outside_peaks<0.01))
xlabel('Variance Spike Proba')
ylabel('Entropy 5 ms')
subplot(2,1,2); 
scatter(mean_proportion_PYR_inside_peaks,Variance_Spike_Proba_PYR)
ylabel('Variance Spike Proba')
xlabel('Proportion of spikes inside peaks')


figure(); 
subplot(2,2,1); hold on; 
scatter(new_mean_firing_rate_PYR(mean_proportion_PYR_outside_peaks<0.01),mean_proportion_PYR_inside_peaks(mean_proportion_PYR_outside_peaks<0.01))
scatter(new_mean_firing_rate_PYR(mean_proportion_PYR_outside_peaks>0.01),mean_proportion_PYR_inside_peaks(mean_proportion_PYR_outside_peaks>0.01))
xlabel('Firing rate PYR')
ylabel('Proportion of spikes inside peaks')
subplot(2,2,2); 
scatter(new_mean_firing_rate_PYR,Variance_Spike_Proba_PYR)
xlabel('Firing rate PYR')
ylabel('Variance Spike Proba')
subplot(2,2,3); 
scatter(new_mean_firing_rate_PYR,Entropy_5ms_PYR)
xlabel('Firing rate PYR')
ylabel('Entropy 5 ms')
subplot(2,2,4);
scatter(new_mean_firing_rate_PYR,mean_proportion_PYR_outside_peaks)
xlabel('Firing rate PYR')
xlabel('Proportion of spikes outside peaks')

matrix=[10 37]; 
new_mean_firing_rate_PYR=reshape(new_mean_firing_rate_PYR,matrix)';
params_1=reshape(new_parameters(:,1),matrix)';
params_2=reshape(new_parameters(:,2),matrix)';
params_3=reshape(new_parameters(:,3),matrix)';
params_4=reshape(new_parameters(:,4),matrix)';
params_5=reshape(new_parameters(:,5),matrix)';
Entropy_5ms_PYR=reshape(Entropy_5ms_PYR,matrix)';
mean_proportion_PYR_inside_peaks=reshape(mean_proportion_PYR_inside_peaks,matrix)';
mean_proportion_PYR_outside_peaks=reshape(mean_proportion_PYR_outside_peaks,matrix)';
Period_peaks=reshape(Period_peaks,matrix)'; 
Variance_Spike_Proba_PYR=reshape(Variance_Spike_Proba_PYR,matrix)';



%params order:  [taue,taui,Iext_PYR,we_PYR_PV,wi_PV_PYR]; 
for k=1:matrix(2)
    reordered_mean_firing_rate_PYR(k,:)=new_mean_firing_rate_PYR(k,index_order(k,:)); 
    reordered_params_1(k,:)=params_1(k,index_order(k,:)); 
    reordered_params_2(k,:)=params_2(k,index_order(k,:)); 
    reordered_params_3(k,:)=params_3(k,index_order(k,:)); 
    reordered_params_4(k,:)=params_4(k,index_order(k,:)); 
    reordered_params_5(k,:)=params_5(k,index_order(k,:)); 
end

figure(); 
subplot(1,2,1); 
imagesc(reordered_mean_firing_rate_PYR); 
caxis([0 5])
subplot(1,2,2); 
imagesc(new_Entropy_5ms_PYR); 

figure(); 
subplot(1,3,1); 
imagesc(reordered_params_3);
title('Iext on PYR')
subplot(1,3,2); 
imagesc(reordered_params_4); 
title('we')
subplot(1,3,3); 
imagesc(reordered_params_5);
title('wi')

figure(); 
subplot(1,2,1); 
imagesc(reordered_params_5 - reordered_params_4); 
title('wi -we')
subplot(1,2,2); 
imagesc(reordered_params_5./reordered_params_4); 
title('wi/we')


% Correlation matrices
figure(); imagesc(corrcoef([mean_firing_rate_PYR  Entropy_5ms_PYR  parameters(:,3:5)]))
caxis([-1 1])
map=bluewhitered(100);
colormap(map)

selected_trials=mean_firing_rate_PYR>2;
figure(); 
imagesc(corrcoef([mean_firing_rate_PYR(selected_trials)  Entropy_5ms_PYR(selected_trials)  parameters(selected_trials,3:5)]))
caxis([-1 1])
map=bluewhitered(100);
colormap(map)



% Effect of DBS 

load('Map_parameters_DBS_effect')
% Sort according to the previous sorting method: 
mean_firing_rate_PYR_under_DBS = mean_firing_rate_PYR_DBS(index_order_1); 
Entropy_5ms_under_DBS = Entropy_5ms_PYR_DBS(index_order_1); 
Variance_Spike_Proba_PYR_under_DBS=Variance_Spike_Proba_PYR_DBS(index_order_1);
Period_peaks_under_DBS=Period_peaks_DBS(index_order_1);
matrix=[10 37]; 
mean_firing_rate_PYR_under_DBS=reshape(mean_firing_rate_PYR_under_DBS,matrix)';
Entropy_5ms_under_DBS=reshape(Entropy_5ms_under_DBS,matrix)';
Period_peaks_under_DBS=reshape(Period_peaks_under_DBS,matrix)'; 
Variance_Spike_Proba_PYR_under_DBS=reshape(Variance_Spike_Proba_PYR_under_DBS,matrix)';

for k=1:matrix(2)
    reorder_Entropy_5ms_under_DBS(k,:)=Entropy_5ms_under_DBS(k,index_order(k,:));
    reordered_mean_firing_rate_PYR_under_DBS(k,:)=mean_firing_rate_PYR_under_DBS(k,index_order(k,:)); 
end

figure(); 
subplot(2,2,1); 
imagesc(reordered_mean_firing_rate_PYR); 
caxis([0 5])
subplot(2,2,2); 
imagesc(reordered_mean_firing_rate_PYR_under_DBS); 
caxis([0 5])
subplot(2,2,3); 
imagesc(new_Entropy_5ms_PYR); 
caxis([1 3.5])
subplot(2,2,4); 
imagesc(reorder_Entropy_5ms_under_DBS); 
caxis([1 3.5])

figure(); 
subplot(2,1,1); hold on; plot([0 16],[0 16],'k-')
scatter(reordered_mean_firing_rate_PYR(:),reordered_mean_firing_rate_PYR_under_DBS(:))
xlabel('Firing rate without DBS')
ylabel('Firing rate with DBS')
subplot(2,1,2); hold on; plot([0 4],[0 4],'k-')
scatter(new_Entropy_5ms_PYR(:),reorder_Entropy_5ms_under_DBS(:))
xlabel('Entropy rate without DBS')
ylabel('Entropy rate with DBS')


difference_firing = (reordered_mean_firing_rate_PYR_under_DBS(:) - reordered_mean_firing_rate_PYR(:))./reordered_mean_firing_rate_PYR(:); 
difference_entropy = reorder_Entropy_5ms_under_DBS(:) - new_Entropy_5ms_PYR(:); 
params_3_new = reordered_params_3(:); 
params_4_new = reordered_params_4(:); 
params_5_new = reordered_params_5(:); 


figure(); 
subplot(1,3,1); hold on; 
scatter(reordered_mean_firing_rate_PYR(:),difference_firing,60,params_3_new,'filled')
plot([0 9],[0 0],'k-')
xlabel('Firing without DBS')
ylabel('Difference in firing WITH - WITHOUT')
title('Colormap=  Iext')
subplot(1,3,2); hold on; 
scatter(reordered_mean_firing_rate_PYR(:),difference_firing,60,params_4_new,'filled')
plot([0 9],[0 0],'k-')
xlabel('Firing without DBS')
ylabel('Difference in firing WITH - WITHOUT')
title('Colormap=  We')
subplot(1,3,3); hold on; 
scatter(reordered_mean_firing_rate_PYR(:),difference_firing,60,params_4_new,'filled')
plot([0 9],[0 0],'k-')
xlabel('Firing without DBS')
ylabel('Difference in firing WITH - WITHOUT')
title('Colormap=  Wi')

figure(); 
scatter(Entropy_5ms_under_DBS,mean_firing_rate_PYR_under_DBS); hold on; 
scatter(Entropy_5ms_PYR,mean_firing_rate_PYR)
ylim([0 9])
xlim([1 4])

figure(); hold on; 
scatter(difference_firing,difference_entropy)
ylabel('Difference of entropy WITH-WITHOUT')
xlabel('Difference of firing WITH-WITHOUT')
plot([-1 1],[0 0],'k-')
plot([0 0],[-2.5 2],'k-')


figure(); 
subplot(1,3,1); hold on; 
scatter(new_Entropy_5ms_PYR(:),difference_entropy,60,params_3_new,'filled')
plot([1 4],[0 0],'k-')
xlabel('Entropy without DBS')
ylabel('Difference in entropy WITH - WITHOUT')
title('Colormap=  Iext')
subplot(1,3,2); hold on; 
scatter(new_Entropy_5ms_PYR(:),difference_entropy,60,params_4_new,'filled')
plot([1 4],[0 0],'k-')
xlabel('Entropy without DBS')
ylabel('Difference in entropy WITH - WITHOUT')
title('Colormap=  We')
subplot(1,3,3); hold on; 
scatter(new_Entropy_5ms_PYR(:),difference_entropy,60,params_4_new,'filled')
plot([1 4],[0 0],'k-')
xlabel('Entropy without DBS')
ylabel('Difference in entropy WITH - WITHOUT')
title('Colormap=  Wi')


figure(); imagesc(corrcoef([difference_firing  difference_entropy  params_3_new params_4_new params_5_new]))
caxis([-1 1])
set(gca,'xtick',1:5,'xticklabel',{'Diff Firing','Diff Entropy','Iext','We','Wi'})
set(gca,'ytick',1:5,'yticklabel',{'Diff Firing','Diff Entropy','Iext','We','Wi'})


figure(); 
subplot(2,1,1); hold on ;
scatter(new_Entropy_5ms_PYR(:),reordered_mean_firing_rate_PYR(:),60,difference_firing(:),'filled'); 
ylabel('Firing without DBS')
xlabel('Entropy without DBS')
title('Difference in Firing rate (normalized)')
map=bluewhitered(100);
colormap(map)
caxis([-1 1])
subplot(2,1,2); hold on ;
scatter(new_Entropy_5ms_PYR(:),reordered_mean_firing_rate_PYR(:),60,difference_entropy(:),'filled')
caxis([-1.5 1.5])
ylabel('Firing without DBS')
xlabel('Entropy without DBS')
title('Difference in entropy')
