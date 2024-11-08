path1 = '/Networks';
path2 = '/Constant_pulses';
path3 = '/Saved_Results';

recording_name='short_constant';


cd(path1)
load('parameters')


dt = 0.05;
T = 800;
N_times=ceil(T/dt);
Time_vector = 0:dt:T;
Bin_Length = 5; 

N_PYR = 800;
N_SOM = 80;
N_PV = 120;
N_total = N_PYR + N_SOM + N_PV;

constant_stimulus_intensity = [80,70,68,64,60,58,54,50,40];
stimulus_duration = 300; % short pulses to estimate firing rates and number of cells that are activated
Time_position = 500;
number_cell = 200; % number of cells receiving the stimulus
trial_number = 20; % number of repetitions


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
VT_SOM = -53; 

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


% DBS parameters (pA) : 
% OFF with I0_PYR=I0_PV=I0_SOM = 0 pA 
% ON  with I0_PYR=I0_PV=I0_SOM = 200 pA 
I0_PYR = 200; 
I0_PV = 200;
I0_SOM = 200;
delay_PV = 0.5;
delay_SOM = 2;
delay_PYR = 2;
frequency = 1e-3*130; % 130 Hz (or 30 Hz)
pulse_duration = 2;

% DBS from the start of the stimulation
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



parfor Z=1:length(parameters)
    
    disp(Z)
    Iext_PYR = parameters(Z,3);
    we_PYR_PV = parameters(Z,4);
    wi_PV_PYR = parameters(Z,5); 
    
    
         for A=1:length(constant_stimulus_intensity) 
            
            disp(strcat('Stimulus number = ',num2str(A))) ;
            
            Spike_binning_PYR_DBS_PSTH = zeros(trial_number,N_PYR,2*stimulus_duration/Bin_Length);
            Spike_binning_PV_DBS_PSTH = zeros(trial_number,N_PV,2*stimulus_duration/Bin_Length);
           
            for p=1:trial_number
                
                cd(path1)
                
                stimulus = zeros(N_PYR,N_times); % additional stimulus to Pyramidal cells
                stimulus_intensity = constant_stimulus_intensity(A);
                receptor_PYR = load(strcat('receptor_cells_',num2str(A)));

                i=1;non_receptor_PYR = zeros(600,1);
                for w=1:N_PYR
                    if ismember(w,receptor_PYR)==1
                        stimulus(w,Time_position/dt:Time_position/dt + stimulus_duration/dt) = stimulus_intensity;
                    else
                        non_receptor_PYR(i)=w ;
                        i=i+1;
                    end
                end
                
                
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
                    
                    V_PYR_DBS(index_PYR_DBS,i+1)=V_PYR_DBS(index_PYR_DBS,i)+dt*(gL_PYR*(EL-V_PYR_DBS(index_PYR_DBS,i))+gL_PYR*DeltaT_PYR*exp((V_PYR_DBS(index_PYR_DBS,i)-VT_PYR)/DeltaT_PYR) + ge_PYR_DBS(index_PYR_DBS,i).*(Ee-V_PYR_DBS(index_PYR_DBS,i)) + gi_PYR_DBS(index_PYR_DBS,i).*(Ei-V_PYR_DBS(index_PYR_DBS,i)) + Iext_PYR + + stimulus_total_PYR(index_PYR_DBS,i)  + stimulus(index_PYR_DBS,i) - w_PYR_DBS(index_PYR_DBS,i))/C_PYR + (sigma*sqrt(dt)/sqrt(taum_PYR))*dzeta_PYR(index_PYR_DBS)';
                    w_PYR_DBS(index_PYR_DBS,i+1)=w_PYR_DBS(index_PYR_DBS,i)+dt*(a_PYR*(V_PYR_DBS(index_PYR_DBS,i)-EL) - w_PYR_DBS(index_PYR_DBS,i))/tauw_PYR ;
                    
                    V_PV_DBS(index_PV_DBS,i+1)=V_PV_DBS(index_PV_DBS,i)+dt*(gL_PV*(EL-V_PV_DBS(index_PV_DBS,i))+gL_PV*DeltaT_PV*exp((V_PV_DBS(index_PV_DBS,i)-VT_PV)/DeltaT_PV) + ge_PV_DBS(index_PV_DBS,i).*(Ee-V_PV_DBS(index_PV_DBS,i)) + gi_PV_DBS(index_PV_DBS,i).*(Ei-V_PV_DBS(index_PV_DBS,i)) + stimulus_total_PV(index_PV_DBS,i) + Iext_PV  - w_PV_DBS(index_PV_DBS,i))/C_Int + (sigma*sqrt(dt)/sqrt(taum_Int))*dzeta_PV(index_PV_DBS)';
                    w_PV_DBS(index_PV_DBS,i+1)=w_PV_DBS(index_PV_DBS,i)+dt*(a_PV*(V_PV_DBS(index_PV_DBS,i)-EL) - w_PV_DBS(index_PV_DBS,i))/tauw_PV ;
                    
                    V_SOM_DBS(index_SOM_DBS,i+1)=V_SOM_DBS(index_SOM_DBS,i)+dt*(gL_SOM*(EL-V_SOM_DBS(index_SOM_DBS,i))+gL_SOM*DeltaT_SOM*exp((V_SOM_DBS(index_SOM_DBS,i)-VT_SOM)/DeltaT_SOM) + gi_SOM_DBS(index_SOM_DBS,i).*(Ei-V_SOM_DBS(index_SOM_DBS,i))+ Iext_SOM  + stimulus_total_SOM(index_SOM_DBS,i) - w_SOM_DBS(index_SOM_DBS,i))/C_Int + (sigma*sqrt(dt)/sqrt(taum_Int))*dzeta_SOM(index_SOM_DBS)';
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
                
               
                
                % We consider for analysis the last 300 ms before the stimulus onset and then the first 300 ms during stimulus presentation                
                V_PYR_binary_DBS=zeros(N_PYR,N_times); V_PYR_binary_DBS(V_PYR_DBS==Amp_spike)=1;
                V_PV_binary_DBS=zeros(N_PV,N_times);   V_PV_binary_DBS(V_PV_DBS==Amp_spike)=1;
                
                % Spike binning for PSTH
                Spike_binned_DBS_PSTH=zeros(N_PYR,ceil(2*stimulus_duration/Bin_Length));
                v=1;
                for z=0:((2*stimulus_duration)/Bin_Length-1)
                    Spike_binned_DBS_PSTH(:,v)=sum(V_PYR_binary_DBS(:,(Time_position-300)/dt+z*Bin_Length/dt:(Time_position-300)/dt+(z+1)*Bin_Length/dt),2);
                    v=v+1;
                end
                Spike_binned_DBS_PSTH=vertcat(Spike_binned_DBS_PSTH(receptor_PYR,:),Spike_binned_DBS_PSTH(non_receptor_PYR,:));
                Spike_binning_PYR_DBS_PSTH(p,:,:) = Spike_binned_DBS_PSTH;
                
                Spike_binned_PV_DBS_PSTH=zeros(N_PV,ceil(2*stimulus_duration/Bin_Length));
                v=1;
                for z=0:((2*stimulus_duration)/Bin_Length-1)
                    Spike_binned_PV_DBS_PSTH(:,v)=sum(V_PV_binary_DBS(:,(Time_position-300)/dt+z*Bin_Length/dt:(Time_position-300)/dt+(z+1)*Bin_Length/dt),2);
                    v=v+1;
                end
                Spike_binning_PV_DBS_PSTH(p,:,:) = Spike_binned_PV_DBS_PSTH;
                                
                
            end
            
            cd(path2)
            dlmwrite(strcat('V_PYR_binned_PSTH_DBS_130Hz_ON_Params_',num2str(Z),'_',recording_name,'_',num2str(A)),Spike_binning_PYR_DBS_PSTH)
            dlmwrite(strcat('V_PV_binned_PSTH_DBS_130Hz_ON_Params_',num2str(Z),'_',recording_name,'_',num2str(A)),Spike_binning_PV_DBS_PSTH)
            
         end
       
end
    


%% Analysis (example here for the case when DBS is ON, at 130 Hz) 

clear

path1 = '/Networks';
path2 = '/Constant_pulses';
path3 = '/Saved_Results';


T = 800;
N_PYR = 800;
receptor_cells = 200;
stimulus_duration = 300; 
Time_position = 500;
number_cell = 200; 
number_trials = 20;
Bin_Length = 5; 
dt=0.05;

constant_amplitude_stimuli = [80,70,68,64,60,58,54,50,40];
label=1:9; 

cd(path1)
load('parameters.mat')

for Z=1:length(parameters);
    
    for k=1:length(constant_amplitude_stimuli)
        
        cd(path3)

        M1 = importdata(strcat('V_PYR_binned_PSTH_DBS_130Hz_ON_Params_',num2str(Z),'_short_constant_',num2str(label(k))));
        for m=1:number_trials
            M1_bis = reshape(M1(m,:),[N_PYR stimulus_duration*2/Bin_Length]);
            
            spiking_R_neurons = movmean(sum(M1_bis(1:number_cell,61:120)~=0)/number_cell,1);
            spiking_NR_neurons = movmean(sum(M1_bis(number_cell+1:end,61:120)~=0)/(N_PYR-number_cell),1);

            [~,peaks]=findpeaks(spiking_R_neurons);
            %figure(); hold on; plot(spiking_R_neurons); plot(spiking_NR_neurons); 
            average_RC_cells(k,m) = mean(spiking_R_neurons(peaks)); 
            std_RC_cells(k,m) = std(spiking_R_neurons(peaks)); 
            average_NRC_cells(k,m)= mean(spiking_NR_neurons(peaks)); 
            std_NRC_cells(k,m)= std(spiking_NR_neurons(peaks)); 
            ratio_recruited(k,m)= mean(spiking_R_neurons(peaks)./spiking_NR_neurons(peaks));

            % changes in SNR linked to absolute firing rate
            firing_rate_RC_stimulus(k,m) = sum(sum(M1_bis(1:number_cell,61:120)))/(number_cell*stimulus_duration/1000); % in Hz
            firing_rate_NRC_stimulus(k,m) = sum(sum(M1_bis(number_cell+1:end,61:120)))/((N_PYR-number_cell)*stimulus_duration/1000);
            firing_rate_RC_off_stimulus(k,m) = sum(sum(M1_bis(1:number_cell,1:60)))/(number_cell*stimulus_duration/1000); % in Hz
            firing_rate_NRC_off_stimulus(k,m) = sum(sum(M1_bis(number_cell+1:end,1:60)))/((N_PYR-number_cell)*stimulus_duration/1000);
            firing_rate_off_stimulus(k,m) = sum(sum(M1_bis(:,1:60)))/(N_PYR*stimulus_duration/1000);
            firing_rate_on_stimulus(k,m) = sum(sum(M1_bis(:,61:120)))/(N_PYR*stimulus_duration/1000);

                        
            SNR_RC(k,m) = firing_rate_RC_stimulus(k,m)./firing_rate_RC_off_stimulus(k,m); 
            SNR_RC_NRC(k,m) = firing_rate_RC_stimulus(k,m)./firing_rate_NRC_stimulus(k,m); 
            
        end
        
    end
    
    save(strcat('Modele_Params_',num2str(Z),'_short_constant.mat'),'average_NRC_cells','average_RC_cells','std_RC_cells','std_NRC_cells','ratio_recruited','firing_rate_RC_stimulus','firing_rate_NRC_stimulus','firing_rate_RC_off_stimulus','firing_rate_NRC_off_stimulus','firing_rate_off_stimulus','firing_rate_on_stimulus','SNR_RC','SNR_RC_NRC')
    
   
    std_firing_off(Z,:) = std(firing_rate_off_stimulus,0,2); 
    std_firing_on(Z,:)= std(firing_rate_on_stimulus,0,2); 
    std_firing_on_RC(Z,:)= std(firing_rate_RC_stimulus,0,2); 
    
    mean_SNR_RC(Z,:) = mean(SNR_RC,2); 
    std_SNR_RC(Z,:) = std(SNR_RC,0,2); 
    mean_SNR_RC_NRC(Z,:) = mean(SNR_RC_NRC,2); 
    std_SNR_RC_NRC(Z,:) = std(SNR_RC_NRC,0,2); 
    
    total_mean_RC_cells(Z,:) = mean(average_RC_cells,2);
    total_std_RC_cells(Z,:) = mean(std_RC_cells,2);
    total_mean_NRC_cells(Z,:) = mean(average_NRC_cells,2);
    total_std_NRC_cells(Z,:) = mean(std_NRC_cells,2);
    ratio_recruited(~isfinite(ratio_recruited)) = NaN; 
    total_ratio_recruited(Z,:) =nanmean(ratio_recruited,2);
    
    mean_firing_rate_off_stimulus(Z,:) = mean(firing_rate_off_stimulus,2); 
    mean_firing_rate_on_stimulus(Z,:) = mean(firing_rate_on_stimulus,2); 
    mean_firing_rate_RC_on_stimulus(Z,:) = mean(firing_rate_RC_stimulus,2); 
    mean_firing_rate_NRC_on_stimulus(Z,:) = mean(firing_rate_NRC_stimulus,2); 
        
    
end

save('Constant_stimuli_DBS_130Hz','period_off','std_firing_off','std_firing_on','std_firing_on_RC','total_Entropy_5ms_PYR_off','total_variance_Spike_proba_off','total_mean_proportion_PYR_outside_peaks_on','total_mean_proportion_PYR_outside_peaks_off','total_mean_proportion_PYR_inside_peaks_on','total_mean_proportion_PYR_inside_peaks_off','mean_period_peaks_off','mean_period_peaks_on','total_mean_RC_cells','total_std_RC_cells','total_mean_NRC_cells','total_std_NRC_cells','total_ratio_recruited','mean_SNR_RC','mean_SNR_RC_NRC','mean_firing_rate_on_stimulus','mean_firing_rate_off_stimulus','mean_firing_rate_RC_on_stimulus','mean_firing_rate_NRC_on_stimulus','mean_synchrony_index_off_stimulus','mean_synchrony_index_RC_on_stimulus','mean_synchrony_index_RC_off_stimulus','mean_synchrony_index_NRC_on_stimulus','mean_synchrony_index_NRC_off_stimulus','mean_synchrony_index_on_stimulus','std_SNR_RC','std_SNR_RC_NRC')



% Disrimination (without gaussian hypothesis)
constant_amplitude_stimuli = [80,70,68,64,60,58,54,50,40];

number_trials = 20;
for Z=1:length(parameters)
    disp(Z)
    for k=1:length(constant_amplitude_stimuli)
        for j=1:length(constant_amplitude_stimuli)
            
            load(strcat('Modele_Params_DBS_130Hz_ON_',num2str(Z),'_short_constant.mat'))
            firing_subset = [firing_rate_on_stimulus(k,:) firing_rate_on_stimulus(j,:)]';
            binary_label = [zeros(number_trials,1); ones(number_trials,1)];
            mdl = fitglm(firing_subset,binary_label,'Distribution','binomial','Link','logit');
            scores = mdl.Fitted.Probability;
            [X,Y,T,AUC] = perfcurve(binary_label,scores,'1');
            discriminability_AUC{Z}(k,j) = AUC;
            
            firing_subset_RC = [firing_rate_RC_stimulus(k,:) firing_rate_RC_stimulus(j,:)]';
            binary_label = [zeros(number_trials,1); ones(number_trials,1)];
            mdl = fitglm(firing_subset_RC,binary_label,'Distribution','binomial','Link','logit');
            scores = mdl.Fitted.Probability;
            [X,Y,T,AUC] = perfcurve(binary_label,scores,'1');
            discriminability_AUC_RC{Z}(k,j) = AUC;

            recruitment_RC_cells = [average_RC_cells(k,:) average_RC_cells(j,:)]';
            binary_label = [zeros(number_trials,1); ones(number_trials,1)];
            mdl = fitglm(recruitment_RC_cells,binary_label,'Distribution','binomial','Link','logit');
            scores = mdl.Fitted.Probability;
            [X,Y,T,AUC] = perfcurve(binary_label,scores,'1');
            discriminability_recruited_RC{Z}(k,j) = AUC;


        end
    end
end
save('Discriminability_DBS_130Hz_ON','discriminability_AUC','discriminability_AUC_RC','discriminability_recruited_RC')



