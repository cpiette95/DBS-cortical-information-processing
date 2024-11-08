clear

path1 = 'path1='/Networks'; % where the connectivity matrices + Maps_parameters are saved

load('Map_parameters.mat')
clearvars -except parameters path1

dt = 0.05;
T = 800;
N_times=ceil(T/dt);
Time_vector = 0:dt:T;
Bin_Length = 5; % ms bin (=> 100 bins for stimulus ON)

N_PYR = 800;
N_SOM = 80;
N_PV = 120;
N_total = N_PYR + N_SOM + N_PV;

stimulus_duration = 600;
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

taue = 5; %3
taui = 8; %5
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


Spike_binning_PYR_DBS_PSTH = zeros(trial_number,N_PYR,stimulus_duration/Bin_Length);
Spike_binning_PV_DBS_PSTH = zeros(trial_number,N_PV,stimulus_duration/Bin_Length);
Moving_Proba_spike_DBS_PYR_PSTH_Bin5 = zeros(trial_number,(stimulus_duration-100)/dt);
Moving_Proba_spike_DBS_PYR_PSTH_Bin10 = zeros(trial_number,(stimulus_duration-100)/dt);
Moving_Proba_spike_DBS_PV_PSTH_Bin10 = zeros(trial_number,(stimulus_duration-100)/dt);

Entropy_5ms_PYR_DBS=zeros(trial_number,1); 
Number_peaks_DBS = zeros(trial_number,1); 
Period_peaks_DBS=zeros(trial_number,1); 
Variance_Spike_Proba_PYR_DBS = zeros(trial_number,1); 
mean_proportion_PYR_inside_peaks_DBS = zeros(trial_number,1); 
mean_proportion_PYR_outside_peaks_DBS = zeros(trial_number,1); 
mean_firing_rate_PYR_DBS = zeros(trial_number,1); 
mean_firing_rate_PV_DBS = zeros(trial_number,1); 
mean_firing_rate_SOM_DBS = zeros(trial_number,1); 
mean_phase_DBS=NaN*ones(trial_number,N_PYR);
 

%% DBS parameters (pA)
I0_PYR = 200;
I0_PV = 200;
I0_SOM = 200;
delay_PV = 0.5;
delay_SOM = 2;
delay_PYR = 2;
frequency = 1e-3*130;
pulse_duration = 2;

stim_on_PYR = ones(round(pulse_duration/dt),400);
stim_off_PYR = zeros((round(1/(frequency*dt))-size(stim_on_PYR,1)),400);
stimulus_template_PYR = vertcat(stim_on_PYR,stim_off_PYR);
stimulus_total_PYR = repmat(I0_PYR*stimulus_template_PYR',1,ceil(ceil(T/dt)/size(stimulus_template_PYR,1)));

stim_off_pre_PYR_bis = zeros(round(delay_PYR/dt),400);
stim_on_PYR_bis = ones(round(pulse_duration/dt),400);
stim_off_PYR_bis = zeros(round(1/(frequency*dt) - size(stim_on_PYR_bis,1) -size(stim_off_pre_PYR_bis,1)),400);
stimulus_template_PYR_bis = vertcat(stim_off_pre_PYR_bis,stim_on_PYR_bis,stim_off_PYR_bis);
stimulus_total_PYR_bis= repmat(I0_PYR*stimulus_template_PYR_bis',1,ceil(ceil(T/dt)/size(stimulus_template_PYR_bis,1)));

stimulus_total_PYR = [stimulus_total_PYR;stimulus_total_PYR_bis];

stim_on_PV = ones(round(pulse_duration/dt),N_PV);
stim_off_PV = zeros(round(1/(frequency*dt)-size(stim_on_PYR,1)),N_PV);
stimulus_template_PV = vertcat(stim_on_PV,stim_off_PV);
stimulus_total_PV = repmat(I0_PV*stimulus_template_PV',1,ceil(ceil(T/dt)/size(stimulus_template_PV,1)));

stim_off_pre_SOM = zeros(round(delay_SOM/dt),N_SOM);
stim_on_SOM = ones(round(pulse_duration/dt),N_SOM);
stim_off_SOM = zeros(round(1/(frequency*dt) - size(stim_on_SOM,1) -size(stim_off_pre_SOM,1)),N_SOM);
stimulus_template_SOM = vertcat(stim_off_pre_SOM,stim_on_SOM,stim_off_SOM);
stimulus_total_SOM = repmat(I0_SOM*stimulus_template_SOM',1,ceil(ceil(T/dt)/size(stimulus_template_PV,1)));

% NODBS at the start of the stimulation (first 200 ms): 
stimulus_total_PYR = [zeros(N_PYR,Time_position/dt)  stimulus_total_PYR]; 
stimulus_total_PV = [zeros(N_PV,Time_position/dt)  stimulus_total_PV]; 
stimulus_total_SOM = [zeros(N_SOM,Time_position/dt)  stimulus_total_SOM]; 



for p=1:size(parameters,1)
    
    
    Iext_PYR = parameters(p,3);
    we_PYR_PV = parameters(p,4);
    wi_PV_PYR = parameters(p,5); 
    
           
    Connect_PYR_PYR = load('Connect_PYR_PYR');
    Connect_PV_PV = load('Connect_PV_PV');
    Connect_SOM_SOM = load('Connect_SOM_SOM');
    Connect_PYR_PV = load('Connect_PYR_PV');
    Connect_PV_PYR = load('Connect_PV_PYR');
    Connect_SOM_PYR = load('Connect_SOM_PYR');
    Connect_SOM_PV = load('Connect_SOM_PV');
    Connect_PV_SOM = load('Connect_PV_SOM');
    
    V_PYR_DBS = zeros(N_PYR,N_times);
    w_PYR_DBS = zeros(N_PYR,N_times);
    V_PV_DBS = zeros(N_PV,N_times);
    w_PV_DBS = zeros(N_PV,N_times);
    V_SOM_DBS = zeros(N_SOM,N_times);
    w_SOM_DBS = zeros(N_SOM,N_times);
    
    % Initial conditions
    sigma_V_init=5;
    sigma_W_init=5;
    
    V_PYR_DBS(:,1) = EL + sigma_V_init*rand(N_PYR,1);
    V_PV_DBS(:,1) = EL + sigma_V_init*rand(N_PV,1);
    V_SOM_DBS(:,1) = EL + sigma_V_init*rand(N_SOM,1);
    
    w_PYR_DBS(:,1) =  a_PYR*(V_PYR_DBS(:,1)-EL)+ sigma_W_init*rand(N_PYR,1);
    w_PV_DBS(:,1) =  a_PV*(V_PV_DBS(:,1)-EL)+ sigma_W_init*rand(N_PV,1);
    w_SOM_DBS(:,1) =  a_SOM*(V_SOM_DBS(:,1)-EL)+ sigma_W_init*rand(N_SOM,1);
    
    n_spikes_init=1;
    
    ge_PYR_DBS = zeros(N_PYR,N_times);
    gi_PYR_DBS = zeros(N_PYR,N_times);
    ge_PV_DBS = zeros(N_PV,N_times);
    gi_PV_DBS = zeros(N_PV,N_times);
    gi_SOM_DBS = zeros(N_SOM,N_times);
    
    ge_PYR_DBS(:,1) = we_PYR_PYR*N_PYR/N_PYR * rand(N_PYR,1)*n_spikes_init;
    gi_PYR_DBS(:,1) = (wi_PV_PYR*N_PV/N_PV + wi_SOM_PYR*N_SOM/N_SOM)/2*rand(N_PYR,1)*n_spikes_init;
    ge_PV_DBS(:,1) = we_PYR_PV*N_PYR/N_PYR * rand(N_PV,1)*n_spikes_init;
    gi_PV_DBS(:,1) = (wi_PV_PV*N_PV/N_PV + wi_SOM_PV*N_SOM/N_SOM)/2*rand(N_PV,1)*n_spikes_init;
    gi_SOM_DBS(:,1) = (wi_SOM_SOM*N_SOM/N_SOM+wi_PV_SOM*N_PV/N_PV)/2*rand(N_SOM,1)*n_spikes_init;
    
    result_exc_spikes_PYR_DBS = zeros(N_PYR,N_times);
    result_inh_spikes_PYR_DBS = zeros(N_PYR,N_times);
    result_exc_spikes_PV_DBS = zeros(N_PV,N_times);
    result_inh_spikes_PV_DBS = zeros(N_PV,N_times);
    result_inh_spikes_SOM_DBS = zeros(N_SOM,N_times);
    
    i=1;
    
    for t=1:dt:(T-dt)
        %% Update and save the voltage + adaptation variables
        
        dzeta_PYR = randn(1,N_PYR);  % random input to each neuron
        dzeta_PV = randn(1,N_PV);
        dzeta_SOM = randn(1,N_SOM);
        
        index_PYR_DBS = find(V_PYR_DBS(:,i)<Vcut); % distinction between spiking neurons (-> update to Vreset) and non-spiking neurons (-> differential equation)
        index_PV_DBS = find(V_PV_DBS(:,i)<Vcut);
        index_SOM_DBS = find(V_SOM_DBS(:,i)<Vcut);
        
        V_PYR_DBS(index_PYR_DBS,i+1)=V_PYR_DBS(index_PYR_DBS,i)+dt*(gL_PYR*(EL-V_PYR_DBS(index_PYR_DBS,i))+gL_PYR*DeltaT_PYR*exp((V_PYR_DBS(index_PYR_DBS,i)-VT_PYR)/DeltaT_PYR) + ge_PYR_DBS(index_PYR_DBS,i).*(Ee-V_PYR_DBS(index_PYR_DBS,i)) + gi_PYR_DBS(index_PYR_DBS,i).*(Ei-V_PYR_DBS(index_PYR_DBS,i)) + Iext_PYR  + stimulus_total_PYR(index_PYR_DBS,i) - w_PYR_DBS(index_PYR_DBS,i))/C_PYR + (sigma*sqrt(dt)/sqrt(taum_PYR))*dzeta_PYR(index_PYR_DBS)';
        w_PYR_DBS(index_PYR_DBS,i+1)=w_PYR_DBS(index_PYR_DBS,i)+dt*(a_PYR*(V_PYR_DBS(index_PYR_DBS,i)-EL) - w_PYR_DBS(index_PYR_DBS,i))/tauw_PYR ;
        
        V_PV_DBS(index_PV_DBS,i+1)=V_PV_DBS(index_PV_DBS,i)+dt*(gL_PV*(EL-V_PV_DBS(index_PV_DBS,i))+gL_PV*DeltaT_PV*exp((V_PV_DBS(index_PV_DBS,i)-VT_PV)/DeltaT_PV) + ge_PV_DBS(index_PV_DBS,i).*(Ee-V_PV_DBS(index_PV_DBS,i)) + gi_PV_DBS(index_PV_DBS,i).*(Ei-V_PV_DBS(index_PV_DBS,i)) + Iext_PV + stimulus_total_PV(index_PV_DBS,i) - w_PV_DBS(index_PV_DBS,i))/C_Int + (sigma*sqrt(dt)/sqrt(taum_Int))*dzeta_PV(index_PV_DBS)';
        w_PV_DBS(index_PV_DBS,i+1)=w_PV_DBS(index_PV_DBS,i)+dt*(a_PV*(V_PV_DBS(index_PV_DBS,i)-EL) - w_PV_DBS(index_PV_DBS,i))/tauw_PV ;
        
        V_SOM_DBS(index_SOM_DBS,i+1)=V_SOM_DBS(index_SOM_DBS,i)+dt*(gL_SOM*(EL-V_SOM_DBS(index_SOM_DBS,i))+gL_SOM*DeltaT_SOM*exp((V_SOM_DBS(index_SOM_DBS,i)-VT_SOM)/DeltaT_SOM) + gi_SOM_DBS(index_SOM_DBS,i).*(Ei-V_SOM_DBS(index_SOM_DBS,i))+ Iext_SOM + stimulus_total_SOM(index_SOM_DBS,i) - w_SOM_DBS(index_SOM_DBS,i))/C_Int + (sigma*sqrt(dt)/sqrt(taum_Int))*dzeta_SOM(index_SOM_DBS)';
        w_SOM_DBS(index_SOM_DBS,i+1)=w_SOM_DBS(index_SOM_DBS,i)+dt*(a_SOM*(V_SOM_DBS(index_SOM_DBS,i)-EL) - w_SOM_DBS(index_SOM_DBS,i))/tauw_SOM ;
        
        
        %% Update the inputs received at each synapses at every timestep
        
        Spikes_PYR_DBS = double(V_PYR_DBS(:,i)>Vcut);  % Boolean: spiking or not spiking
        Spikes_PV_DBS = double(V_PV_DBS(:,i)>Vcut);
        Spikes_SOM_DBS = double(V_SOM_DBS(:,i)>Vcut);
        
        result_exc_spikes_PYR_DBS(:,i) = we_PYR_PYR*N_PYR/N_PYR.*(Connect_PYR_PYR'*Spikes_PYR_DBS(:)); % attention, utilisation de la transpose de Connect
        result_inh_spikes_PYR_DBS(:,i) = wi_SOM_PYR*N_SOM/N_SOM.*(Connect_SOM_PYR'*Spikes_SOM_DBS(:)) + wi_PV_PYR*N_PV/N_PV.*(Connect_PV_PYR'*Spikes_PV_DBS(:));
        result_exc_spikes_PV_DBS(:,i) = we_PYR_PV*N_PYR/N_PYR.*(Connect_PYR_PV'*Spikes_PYR_DBS(:));
        result_inh_spikes_PV_DBS(:,i) = wi_PV_PV*N_PV/N_PV.*(Connect_PV_PV'*Spikes_PV_DBS(:))+wi_SOM_PV*N_SOM/N_SOM.*(Connect_SOM_PV'*Spikes_SOM_DBS(:));
        result_inh_spikes_SOM_DBS(:,i) = wi_SOM_SOM*N_SOM/N_SOM.*(Connect_SOM_SOM'*Spikes_SOM_DBS(:))+wi_PV_SOM*N_PV/N_PV.*(Connect_PV_SOM'*Spikes_PV_DBS(:));
        
        
        if i>synaptic_delay/dt
            ge_PYR_DBS(:,i+1) = ge_PYR_DBS(:,i) + dt*(-ge_PYR_DBS(:,i)/taue) + result_exc_spikes_PYR_DBS(:,i-synaptic_delay/dt);
            gi_PYR_DBS(:,i+1) = gi_PYR_DBS(:,i) + dt*(-gi_PYR_DBS(:,i)/taui) + result_inh_spikes_PYR_DBS(:,i-synaptic_delay/dt) ;
            ge_PV_DBS(:,i+1) = ge_PV_DBS(:,i) + dt*(-ge_PV_DBS(:,i)/taue) + result_exc_spikes_PV_DBS(:,i-synaptic_delay/dt);
            gi_PV_DBS(:,i+1) = gi_PV_DBS(:,i) + dt*(-gi_PV_DBS(:,i)/taui) + result_inh_spikes_PV_DBS(:,i-synaptic_delay/dt);
            gi_SOM_DBS(:,i+1) = gi_SOM_DBS(:,i) + dt*(-gi_SOM_DBS(:,i)/taui) + result_inh_spikes_SOM_DBS(:,i-synaptic_delay/dt);
            
        else
            ge_PYR_DBS(:,i+1) = ge_PYR_DBS(:,i) + dt*(-ge_PYR_DBS(:,i)/taue) ;
            gi_PYR_DBS(:,i+1) = gi_PYR_DBS(:,i) + dt*(-gi_PYR_DBS(:,i)/taui) ;
            ge_PV_DBS(:,i+1) = ge_PV_DBS(:,i) + dt*(-ge_PV_DBS(:,i)/taue) ;
            gi_PV_DBS(:,i+1) = gi_PV_DBS(:,i) + dt*(-gi_PV_DBS(:,i)/taui) ;
            gi_SOM_DBS(:,i+1) = gi_SOM_DBS(:,i) + dt*(-gi_SOM_DBS(:,i)/taui) ;
            
        end
        
        
        index_PYR_s_DBS = find(Spikes_PYR_DBS);
        index_PV_s_DBS = find(Spikes_PV_DBS);
        index_SOM_s_DBS = find(Spikes_SOM_DBS);
        
        V_PYR_DBS(index_PYR_s_DBS,i)= Amp_spike;
        V_PYR_DBS(index_PYR_s_DBS,i+1)=  VR_PYR;
        w_PYR_DBS(index_PYR_s_DBS,i+1)= w_PYR_DBS(index_PYR_s_DBS,i)+b_PYR;
        
        V_PV_DBS(index_PV_s_DBS,i)= Amp_spike;
        V_PV_DBS(index_PV_s_DBS,i+1)=  VR_PV;
        w_PV_DBS(index_PV_s_DBS,i+1)= w_PV_DBS(index_PV_s_DBS,i)+b_PV;
        
        V_SOM_DBS(index_SOM_s_DBS,i)= Amp_spike;
        V_SOM_DBS(index_SOM_s_DBS,i+1)=  VR_SOM;
        w_SOM_DBS(index_SOM_s_DBS,i+1)= w_SOM_DBS(index_SOM_s_DBS,i)+b_SOM;
        
        
        i=i+1;
        
    end
    
    %% ANALYSIS
    
    PYR_spike_number_DBS_pre = length(find(V_PYR_DBS(:,1000:4000)==Amp_spike)); 
    
    PYR_spike_number_DBS = length(find(V_PYR_DBS(:,6000:end)==Amp_spike)); % removal of first 200 ms (off DBS) + next 100 ms (once DBS on)
    PV_spike_number_DBS = length(find(V_PV_DBS(:,6000:end)==Amp_spike));
    SOM_spike_number_DBS = length(find(V_SOM_DBS(:,6000:end)==Amp_spike));
    
    idx = find(V_PYR_DBS==Amp_spike);
    [row_PYR_DBS,col]=ind2sub(size(V_PYR_DBS),idx);
    PYR_spike_time_DBS = col*dt;
    idx = find(V_PV_DBS==Amp_spike);
    [row_PV_DBS,col]=ind2sub(size(V_PV_DBS),idx);
    PV_spike_time_DBS = col*dt;
    idx = find(V_SOM_DBS==Amp_spike);
    [row_SOM_DBS,col]=ind2sub(size(V_SOM_DBS),idx);
    SOM_spike_time_DBS = col*dt;
    
    mean_firing_rate_PYR_DBS(p) = PYR_spike_number_DBS/((T-300)*1e-3*N_PYR); 
    disp(['PYR =', num2str(mean_firing_rate_PYR_DBS(p))])
    mean_firing_rate_PV_DBS(p) = PV_spike_number_DBS/((T-300)*1e-3*N_PV);
    disp(['PV =', num2str(mean_firing_rate_PV_DBS(p))])
    mean_firing_rate_SOM_DBS(p) = SOM_spike_number_DBS/((T-300)*1e-3*N_SOM);
    disp(['SOM =', num2str(mean_firing_rate_SOM_DBS(p))])
    
        
    V_PYR_binary_DBS=zeros(N_PYR,N_times); V_PYR_binary_DBS(V_PYR_DBS==Amp_spike)=1;
    V_PV_binary_DBS=zeros(N_PV,N_times);   V_PV_binary_DBS(V_PV_DBS==Amp_spike)=1;
    
    
    %% Average metric of population response to various stimuli        
    Count_Spike_DBS_PSTH=[];
    Count_Spike_DBS_PSTH = movsum(V_PYR_binary_DBS(:,(Time_position+100)/dt:(Time_position+stimulus_duration)/dt-1),Bin_Length/dt,2);
    Moving_Proba_spike_DBS_PYR_PSTH_Bin5(p,:) = sum(Count_Spike_DBS_PSTH,1)/mean_firing_rate_PYR_DBS(p);
    
    if mean_firing_rate_PYR_DBS(p)>0

        bins=linspace(0,max(Moving_Proba_spike_DBS_PYR_PSTH_Bin5(p,:)),20);
    r=histcounts(Moving_Proba_spike_DBS_PYR_PSTH_Bin5(p,:),bins);
    for o=1:length(r)
        if r(o) ~=0
        Entropy_5ms_PYR_DBS(p) = Entropy_5ms_PYR_DBS(p) + r(o)./sum(r)*log2(r(o)./sum(r));
        end
    end
    Entropy_5ms_PYR_DBS(p) = -Entropy_5ms_PYR_DBS(p);
    Variance_Spike_Proba_PYR_DBS(p) = var(Moving_Proba_spike_DBS_PYR_PSTH_Bin5(p,:));
    
    [pks,locs]=findpeaks(Moving_Proba_spike_DBS_PYR_PSTH_Bin5(p,:),'MinPeakProminence',4,'MinPeakDistance',200);
    Period_peaks_DBS(p) = mean(diff(locs*dt));
    Number_peaks_DBS(p) = length(locs); 
    mean_diff = mean(diff(locs));
    for o=1:N_PYR
        timing=find(V_PYR_binary_DBS(o,(Time_position+100)/dt:(Time_position+stimulus_duration)/dt-1)>0); 
        if ~isempty(timing)
            phase=[];
            for m=1:length(timing)
                [~,idx]=min(abs(timing(m)-locs));
                phase(m)=(timing(m)-locs(idx))/mean_diff;
            end
            mean_phase_DBS(p,o) = nanmean(phase);
        end
    end
    Variance_Phase(p) = nanvar(mean_phase_DBS(p,:)); 
    
    V_PYR_binary_DBS_subset = V_PYR_binary_DBS(:,(Time_position+100)/dt:(Time_position+stimulus_duration)/dt-1);
    proportion_PYR_inside_peaks=zeros(length(pks),1); 
    for r=1:length(pks) % intervalle 10 ms for assembly
        for o=1:N_PYR
            if sum(V_PYR_binary_DBS_subset(o,max(1,locs(r)-10/dt):min(size(V_PYR_binary_DBS_subset,2),locs(r)+10/dt)))>0
                proportion_PYR_inside_peaks(r)=proportion_PYR_inside_peaks(r)+1; 
            end
        end
    end
   mean_proportion_PYR_inside_peaks_DBS(p) = mean(proportion_PYR_inside_peaks)*100./N_PYR; 
   proportion_PYR_outside_peaks=zeros(length(pks)-1,1); 
   for r=1:length(pks)-1 % intervalle 10 ms for assembly
        for o=1:N_PYR
            if sum(V_PYR_binary_DBS_subset(o,locs(r)+10/dt:locs(r+1)-10/dt))>0
                proportion_PYR_outside_peaks(r)=proportion_PYR_outside_peaks(r)+1; 
            end
        end
    end
   mean_proportion_PYR_outside_peaks_DBS(p) = mean(proportion_PYR_outside_peaks)*100./N_PYR; 
    
    else
        mean_proportion_PYR_outside_peaks_DBS(p)=NaN;
        mean_proportion_PYR_inside_peaks_DBS(p)=NaN;
        Variance_Phase(p)=NaN;
        Period_peaks_DBS(p)=NaN;
        Period_peaks_DBS(p)=NaN;
        Entropy_5ms_PYR_DBS(p)=NaN;
        Variance_Spike_Proba_PYR_DBS(p)=NaN;
    end
   
   Count_Spike_DBS_PSTH=[];
   Count_Spike_DBS_PSTH = movsum(V_PV_binary_DBS(:,(Time_position)/dt:(Time_position+stimulus_duration)/dt-1),Bin_Length/dt,2);
   Moving_Proba_spike_DBS_PV_PSTH_Bin5(p,:) = sum(Count_Spike_DBS_PSTH,1);
    
    
    
end

save(strcat('Map_parameters_DBS_effect'),'mean_firing_rate_PYR_DBS','mean_firing_rate_PV_DBS','mean_firing_rate_SOM_DBS','Number_peaks_DBS','mean_phase_DBS','mean_proportion_PYR_inside_peaks_DBS','mean_proportion_PYR_outside_peaks_DBS','parameters','Entropy_5ms_PYR_DBS','Period_peaks_DBS','Variance_Spike_Proba_PYR_DBS','Moving_Proba_spike_DBS_PYR_PSTH_Bin5','Moving_Proba_spike_DBS_PV_PSTH_Bin5')


