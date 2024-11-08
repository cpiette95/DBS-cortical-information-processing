path1='/Networks'; % where the connectivity matrices are saved

trial_number = 500; % number of different regimes that will be simulated

dt = 0.05;
T = 700;
N_times=ceil(T/dt);
Time_vector = 0:dt:T;
Bin_Length = 5; % ms bin

N_PYR = 800;
N_SOM = 80;
N_PV = 120;
N_total = N_PYR + N_SOM + N_PV;


stimulus_duration = 500;
Time_position = 200;
number_cell = 200; % number of cells receiving the stimulus

% Constant parameters of the network:
EL = -60;
Ee = 0;
Ei = -80;

sigma = 5;

DeltaT_PYR = 1;
DeltaT_PV = 1;
DeltaT_SOM = 5;

taue = 5; 
taui = 8;
synaptic_delay = 1;

gL_PYR = 6;
gL_SOM = 5;
gL_PV = 5;
gL_Int = 5;
C_PYR =  180;
C_Int = 80;
taum_PYR = C_PYR/gL_PYR;
taum_Int = C_Int/gL_Int;

% Specific parameters
VT_PYR = -49;
VT_PV = -52;
VT_SOM = -53; %

VR_PYR = -60;
VR_PV = -60;
VR_SOM = -60;

Vcut = min([VT_PYR,VT_PV,VT_SOM])+40;

a_PYR = 4;
a_PV = 0;
a_SOM = 4;

b_PYR = 100;
b_PV = 0;
b_SOM = 90;

tauw_PYR = 100;
tauw_PV = 15;
tauw_SOM = 40;

% Connectivity weights
%coeff=2.2;
%coeff_1=1.2;
we_PYR_PYR = 0.5;
we_PYR_PV = 1;
wi_PV_PYR = 2.8;
wi_PV_PV = 2.5;
wi_SOM_PYR = 2.2;
wi_SOM_PV = 2.4;
wi_SOM_SOM = 1.6;
wi_PV_SOM = 1.6;

% External inputs
Iext_PV = 50;
Iext_SOM = 25;
Amp_spike = 30;

% Connectivity matrices (les lignes: pr?-synaptiques, les colonnes:
% post-synaptiques)
p_PYR_PYR = 0.5;
p_PV_PV = 0.6;
p_PYR_PV = 0.4;
p_PV_PYR = 0.4;
p_SOM_PYR = 0.4;
p_SOM_PV = 0.3;
p_SOM_SOM = 0.1;
p_PV_SOM = 0.1;


Spike_binning_PYR_PSTH = zeros(trial_number,N_PYR,stimulus_duration/Bin_Length);
Spike_binning_PV_PSTH = zeros(trial_number,N_PV,stimulus_duration/Bin_Length);
Moving_Proba_spike_PYR_PSTH_Bin5 = zeros(trial_number,stimulus_duration/dt);
Moving_Proba_spike_PYR_PSTH_Bin10 = zeros(trial_number,stimulus_duration/dt);
Moving_Proba_spike_PV_PSTH_Bin10 = zeros(trial_number,stimulus_duration/dt);

parameters = zeros(trial_number,5); 
Entropy_5ms_PYR=zeros(trial_number,1); 
Number_peaks = zeros(trial_number,1); 
Period_peaks=zeros(trial_number,1); 
Variance_Spike_Proba_PYR = zeros(trial_number,1); 
mean_proportion_PYR_inside_peaks = zeros(trial_number,1); 
mean_proportion_PYR_outside_peaks = zeros(trial_number,1); 
mean_firing_rate_PYR = zeros(trial_number,1); 
mean_firing_rate_PV = zeros(trial_number,1); 
mean_firing_rate_SOM = zeros(trial_number,1); 
mean_phase=NaN*ones(trial_number,N_PYR);
 

for p=1:trial_number
    
    
    coeff_a=10*ceil(10*rand(1)); 
    Iext_PYR = coeff_a+100;

    coeff=(3-1)*rand(1)+1;
    coeff_1=1.5*rand(1);
    we_PYR_PYR = 0.5;
    we_PYR_PV = coeff*1;
    wi_PV_PYR = coeff_1*2.8;
    
    cd(path1)
    Connect_PYR_PYR = load('Connect_PYR_PYR');
    Connect_PV_PV = load('Connect_PV_PV');
    Connect_SOM_SOM = load('Connect_SOM_SOM');
    Connect_PYR_PV = load('Connect_PYR_PV');
    Connect_PV_PYR = load('Connect_PV_PYR');
    Connect_SOM_PYR = load('Connect_SOM_PYR');
    Connect_SOM_PV = load('Connect_SOM_PV');
    Connect_PV_SOM = load('Connect_PV_SOM');
    
    V_PYR = zeros(N_PYR,N_times);
    w_PYR = zeros(N_PYR,N_times);
    V_PV = zeros(N_PV,N_times);
    w_PV = zeros(N_PV,N_times);
    V_SOM = zeros(N_SOM,N_times);
    w_SOM = zeros(N_SOM,N_times);
    
    % Initial conditions
    sigma_V_init=5;
    sigma_W_init=5;
    
    V_PYR(:,1) = EL + sigma_V_init*rand(N_PYR,1);
    V_PV(:,1) = EL + sigma_V_init*rand(N_PV,1);
    V_SOM_(:,1) = EL + sigma_V_init*rand(N_SOM,1);
    
    w_PYR(:,1) =  a_PYR*(V_PYR(:,1)-EL)+ sigma_W_init*rand(N_PYR,1);
    w_PV(:,1) =  a_PV*(V_PV(:,1)-EL)+ sigma_W_init*rand(N_PV,1);
    w_SOM(:,1) =  a_SOM*(V_SOM_(:,1)-EL)+ sigma_W_init*rand(N_SOM,1);
    
    n_spikes_init=1;
    
    ge_PYR = zeros(N_PYR,N_times);
    gi_PYR = zeros(N_PYR,N_times);
    ge_PV = zeros(N_PV,N_times);
    gi_PV = zeros(N_PV,N_times);
    gi_SOM = zeros(N_SOM,N_times);
    
    ge_PYR(:,1) = we_PYR_PYR*N_PYR/N_PYR * rand(N_PYR,1)*n_spikes_init;
    gi_PYR_(:,1) = (wi_PV_PYR*N_PV/N_PV + wi_SOM_PYR*N_SOM/N_SOM)/2*rand(N_PYR,1)*n_spikes_init;
    ge_PV(:,1) = we_PYR_PV*N_PYR/N_PYR * rand(N_PV,1)*n_spikes_init;
    gi_PV(:,1) = (wi_PV_PV*N_PV/N_PV + wi_SOM_PV*N_SOM/N_SOM)/2*rand(N_PV,1)*n_spikes_init;
    gi_SOM(:,1) = (wi_SOM_SOM*N_SOM/N_SOM+wi_PV_SOM*N_PV/N_PV)/2*rand(N_SOM,1)*n_spikes_init;
    
    result_exc_spikes_PYR = zeros(N_PYR,N_times);
    result_inh_spikes_PYR = zeros(N_PYR,N_times);
    result_exc_spikes_PV = zeros(N_PV,N_times);
    result_inh_spikes_PV = zeros(N_PV,N_times);
    result_inh_spikes_SOM = zeros(N_SOM,N_times);
    
    i=1;
    
    for t=1:dt:(T-dt)
        %% Update and save the voltage + adaptation variables
        
        dzeta_PYR = randn(1,N_PYR);  % random input to each neuron
        dzeta_PV = randn(1,N_PV);
        dzeta_SOM = randn(1,N_SOM);
        
        index_PYR = find(V_PYR(:,i)<Vcut); % distinction between spiking neurons (-> update to Vreset) and non-spiking neurons (-> differential equation)
        index_PV = find(V_PV(:,i)<Vcut);
        index_SOM = find(V_SOM_(:,i)<Vcut);
        
        V_PYR(index_PYR_,i+1)=V_PYR(index_PYR_,i)+dt*(gL_PYR*(EL-V_PYR(index_PYR_,i))+gL_PYR*DeltaT_PYR*exp((V_PYR(index_PYR_,i)-VT_PYR)/DeltaT_PYR) + ge_PYR(index_PYR_,i).*(Ee-V_PYR(index_PYR_,i)) + gi_PYR_(index_PYR_,i).*(Ei-V_PYR(index_PYR_,i)) + Iext_PYR  - w_PYR(index_PYR_,i))/C_PYR + (sigma*sqrt(dt)/sqrt(taum_PYR))*dzeta_PYR(index_PYR_)';
        w_PYR(index_PYR_,i+1)=w_PYR(index_PYR_,i)+dt*(a_PYR*(V_PYR(index_PYR_,i)-EL) - w_PYR(index_PYR_,i))/tauw_PYR ;
        
        V_PV(index_PV,i+1)=V_PV(index_PV,i)+dt*(gL_PV*(EL-V_PV(index_PV,i))+gL_PV*DeltaT_PV*exp((V_PV(index_PV,i)-VT_PV)/DeltaT_PV) + ge_PV(index_PV,i).*(Ee-V_PV(index_PV,i)) + gi_PV(index_PV,i).*(Ei-V_PV(index_PV,i)) + Iext_PV  - w_PV(index_PV,i))/C_Int + (sigma*sqrt(dt)/sqrt(taum_Int))*dzeta_PV(index_PV)';
        w_PV(index_PV,i+1)=w_PV(index_PV,i)+dt*(a_PV*(V_PV(index_PV,i)-EL) - w_PV(index_PV,i))/tauw_PV ;
        
        V_SOM_(index_SOM,i+1)=V_SOM_(index_SOM,i)+dt*(gL_SOM*(EL-V_SOM_(index_SOM,i))+gL_SOM*DeltaT_SOM*exp((V_SOM_(index_SOM,i)-VT_SOM)/DeltaT_SOM) + gi_SOM(index_SOM,i).*(Ei-V_SOM_(index_SOM,i))+ Iext_SOM  - w_SOM(index_SOM,i))/C_Int + (sigma*sqrt(dt)/sqrt(taum_Int))*dzeta_SOM(index_SOM)';
        w_SOM(index_SOM,i+1)=w_SOM(index_SOM,i)+dt*(a_SOM*(V_SOM_(index_SOM,i)-EL) - w_SOM(index_SOM,i))/tauw_SOM ;
        
        
        %% Update the inputs received at each synapses at every timestep
        
        Spikes_PYR_DBS = double(V_PYR(:,i)>Vcut);  % Boolean: spiking or not spiking
        Spikes_PV_DBS = double(V_PV(:,i)>Vcut);
        Spikes_SOM_DBS = double(V_SOM_(:,i)>Vcut);
        
        result_exc_spikes_PYR(:,i) = we_PYR_PYR*N_PYR/N_PYR.*(Connect_PYR_PYR'*Spikes_PYR_DBS(:)); % attention, utilisation de la transpose de Connect
        result_inh_spikes_PYR(:,i) = wi_SOM_PYR*N_SOM/N_SOM.*(Connect_SOM_PYR'*Spikes_SOM_DBS(:)) + wi_PV_PYR*N_PV/N_PV.*(Connect_PV_PYR'*Spikes_PV_DBS(:));
        result_exc_spikes_PV(:,i) = we_PYR_PV*N_PYR/N_PYR.*(Connect_PYR_PV'*Spikes_PYR_DBS(:));
        result_inh_spikes_PV(:,i) = wi_PV_PV*N_PV/N_PV.*(Connect_PV_PV'*Spikes_PV_DBS(:))+wi_SOM_PV*N_SOM/N_SOM.*(Connect_SOM_PV'*Spikes_SOM_DBS(:));
        result_inh_spikes_SOM(:,i) = wi_SOM_SOM*N_SOM/N_SOM.*(Connect_SOM_SOM'*Spikes_SOM_DBS(:))+wi_PV_SOM*N_PV/N_PV.*(Connect_PV_SOM'*Spikes_PV_DBS(:));
        
        
        if i>synaptic_delay/dt
            ge_PYR(:,i+1) = ge_PYR(:,i) + dt*(-ge_PYR(:,i)/taue) + result_exc_spikes_PYR(:,i-synaptic_delay/dt);
            gi_PYR_(:,i+1) = gi_PYR_(:,i) + dt*(-gi_PYR_(:,i)/taui) + result_inh_spikes_PYR(:,i-synaptic_delay/dt) ;
            ge_PV(:,i+1) = ge_PV(:,i) + dt*(-ge_PV(:,i)/taue) + result_exc_spikes_PV(:,i-synaptic_delay/dt);
            gi_PV(:,i+1) = gi_PV(:,i) + dt*(-gi_PV(:,i)/taui) + result_inh_spikes_PV(:,i-synaptic_delay/dt);
            gi_SOM(:,i+1) = gi_SOM(:,i) + dt*(-gi_SOM(:,i)/taui) + result_inh_spikes_SOM(:,i-synaptic_delay/dt);
            
        else
            ge_PYR(:,i+1) = ge_PYR(:,i) + dt*(-ge_PYR(:,i)/taue) ;
            gi_PYR_(:,i+1) = gi_PYR_(:,i) + dt*(-gi_PYR_(:,i)/taui) ;
            ge_PV(:,i+1) = ge_PV(:,i) + dt*(-ge_PV(:,i)/taue) ;
            gi_PV(:,i+1) = gi_PV(:,i) + dt*(-gi_PV(:,i)/taui) ;
            gi_SOM(:,i+1) = gi_SOM(:,i) + dt*(-gi_SOM(:,i)/taui) ;
            
        end
        
        
        index_PYR_s = find(Spikes_PYR_DBS);
        index_PV_s = find(Spikes_PV_DBS);
        index_SOM_s = find(Spikes_SOM_DBS);
        
        V_PYR(index_PYR_s,i)= Amp_spike;
        V_PYR(index_PYR_s,i+1)=  VR_PYR;
        w_PYR(index_PYR_s,i+1)= w_PYR(index_PYR_s,i)+b_PYR;
        
        V_PV(index_PV_s,i)= Amp_spike;
        V_PV(index_PV_s,i+1)=  VR_PV;
        w_PV(index_PV_s,i+1)= w_PV(index_PV_s,i)+b_PV;
        
        V_SOM_(index_SOM_s,i)= Amp_spike;
        V_SOM_(index_SOM_s,i+1)=  VR_SOM;
        w_SOM(index_SOM_s,i+1)= w_SOM(index_SOM_s,i)+b_SOM;
        
        
        i=i+1;
        
    end
    
    %% ANALYSIS
    
    PYR_spike_number_pre = length(find(V_PYR(:,1000:4000)==Amp_spike)); % removal of first 200 ms
    PYR_spike_number = length(find(V_PYR(:,4000:end)==Amp_spike)); 
    PV_spike_number = length(find(V_PV(:,4000:end)==Amp_spike));
    SOM_spike_number = length(find(V_SOM_(:,4000:end)==Amp_spike));
    
    idx = find(V_PYR==Amp_spike);
    [row_PYR_DBS,col]=ind2sub(size(V_PYR),idx);
    PYR_spike_time_DBS = col*dt;
    idx = find(V_PV==Amp_spike);
    [row_PV_DBS,col]=ind2sub(size(V_PV),idx);
    PV_spike_time_DBS = col*dt;
    idx = find(V_SOM_==Amp_spike);
    [row_SOM_DBS,col]=ind2sub(size(V_SOM_),idx);
    SOM_spike_time_DBS = col*dt;
    
    if PYR_spike_number/((T-200)*1e-3*N_PYR) > 100 || PYR_spike_number_pre/(200*1e-3*N_PYR) > 100 || PYR_spike_number/((T-200)*1e-3*N_PYR) <0.01
        continue
    end 
    
        
    mean_firing_rate_PYR(p) = PYR_spike_number/((T-200)*1e-3*N_PYR); 
    disp(['PYR =', num2str(mean_firing_rate_PYR(p))])
    mean_firing_rate_PV(p) = PV_spike_number/((T-200)*1e-3*N_PV);
    disp(['PV =', num2str(mean_firing_rate_PV(p))])
    mean_firing_rate_SOM(p) = SOM_spike_number/((T-200)*1e-3*N_SOM);
    disp(['SOM =', num2str(mean_firing_rate_SOM(p))])
    
    parameters(p,:) = [taue,taui,Iext_PYR,we_PYR_PV,wi_PV_PYR]; 
    
    
%     figure()
%     plot(PYR_spike_time_DBS,row_PYR_DBS,'m.','MarkerSize',10)
%     hold on
%     plot(PV_spike_time_DBS,N_PYR+row_PV_DBS,'g.','MarkerSize',10)
%     plot(SOM_spike_time_DBS,N_PYR+N_PV+row_SOM_DBS,'b.','MarkerSize',10)
%     xlabel('Time (ms)')
%     ylabel('Neuron index')
%     ylim([0 1010])
%     %title('WITH DBS')
%     hold off;
    
    V_PYR_binary=zeros(N_PYR,N_times); V_PYR_binary(V_PYR==Amp_spike)=1;
    V_PV_binary=zeros(N_PV,N_times);   V_PV_binary(V_PV==Amp_spike)=1;
    
    
    %% Average metric of population response to various stimuli
    Count_Spike_PSTH=[];
    Count_Spike_PSTH = movsum(V_PYR_binary(:,(Time_position)/dt:(Time_position+stimulus_duration)/dt-1),Bin_Length/dt,2);
    Moving_Proba_spike_PYR_PSTH_Bin5(p,:) = sum(Count_Spike_PSTH,1)/mean_firing_rate_PYR(p);
    
    bins=linspace(0,max(Moving_Proba_spike_PYR_PSTH_Bin5(p,:)),20);
    r=histcounts(Moving_Proba_spike_PYR_PSTH_Bin5(p,:),bins);
    for o=1:length(r)
        if r(o) ~=0
        Entropy_5ms_PYR(p) = Entropy_5ms_PYR(p) + r(o)./sum(r)*log2(r(o)./sum(r));
        end
    end
    Entropy_5ms_PYR(p) = -Entropy_5ms_PYR(p);
    Variance_Spike_Proba_PYR(p) = var(Moving_Proba_spike_PYR_PSTH_Bin5(p,:));
    
    [pks,locs]=findpeaks(Moving_Proba_spike_PYR_PSTH_Bin5(p,:),'MinPeakProminence',4,'MinPeakDistance',200);
    Period_peaks(p) = mean(diff(locs*dt));
    Number_peaks(p) = length(locs); 
    mean_diff = mean(diff(locs));
    for o=1:N_PYR
        timing=find(V_PYR_binary(o,(Time_position)/dt:(Time_position+stimulus_duration)/dt-1)>0); 
        if ~isempty(timing)
            phase=[];
            for m=1:length(timing)
                [~,idx]=min(abs(timing(m)-locs));
                phase(m)=(timing(m)-locs(idx))/mean_diff;
            end
            mean_phase(p,o) = nanmean(phase);
        end
    end
    Variance_Phase(p) = nanvar(mean_phase(p,:)); 
    
    V_PYR_binary_subset = V_PYR_binary(:,(Time_position)/dt:(Time_position+stimulus_duration)/dt-1);
    proportion_PYR_inside_peaks=zeros(length(pks),1); 
    for r=1:length(pks) % intervalle 10 ms for assembly
        for o=1:N_PYR
            if sum(V_PYR_binary_subset(o,max(1,locs(r)-10/dt):min(size(V_PYR_binary_subset,2),locs(r)+10/dt)))>0
                proportion_PYR_inside_peaks(r)=proportion_PYR_inside_peaks(r)+1; 
            end
        end
    end
   mean_proportion_PYR_inside_peaks(p) = mean(proportion_PYR_inside_peaks)*100./N_PYR; 
   proportion_PYR_outside_peaks=zeros(length(pks)-1,1); 
   for r=1:length(pks)-1 % intervalle 10 ms for assembly
        for o=1:N_PYR
            if sum(V_PYR_binary_subset(o,locs(r)+10/dt:locs(r+1)-10/dt))>0
                proportion_PYR_outside_peaks(r)=proportion_PYR_outside_peaks(r)+1; 
            end
        end
    end
   mean_proportion_PYR_outside_peaks(p) = mean(proportion_PYR_outside_peaks)*100./N_PYR; 
    
    
   
   Count_Spike_PSTH=[];
   Count_Spike_PSTH = movsum(V_PV_binary(:,(Time_position)/dt:(Time_position+stimulus_duration)/dt-1),Bin_Length/dt,2);
   Moving_Proba_spike_PV_PSTH_Bin5(p,:) = sum(Count_Spike_PSTH,1);
    
    
    
end


cd(path1)
save(strcat('Map_parameters'),'mean_firing_rate_PYR','mean_firing_rate_PV','mean_firing_rate_SOM','Number_peaks','mean_phase','mean_proportion_PYR_inside_peaks','mean_proportion_PYR_outside_peaks','parameters','Entropy_5ms_PYR','Period_peaks','Variance_Spike_Proba_PYR','Moving_Proba_spike_PYR_PSTH_Bin5','Moving_Proba_spike_PV_PSTH_Bin5')






