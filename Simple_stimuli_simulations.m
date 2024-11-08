%% Code for running simulations using as external inputs simple stimuli

clear

path1='/Networks';
path2='/Saved_results'; % where to save the matrices
path5='/Simple_stimuli';

cd(path1)
load('parameters.mat')

dt = 0.05;
T = 1300;
N_times=ceil(T/dt);
Time_vector = 0:dt:T;
Bin_Length = 5; % ms bin (=> 100 bins for stimulus ON)

N_PYR = 800;
N_SOM = 80;
N_PV = 120;
N_total = N_PYR + N_SOM + N_PV;


% Different input types --- Constant input: type 1, Ramping input: type 2; OU input: type 3
constant_stimulus_intensity = [110,100,90,80,70,60,50,40,30,20,10];
ramping_stimulus_start = [0,0,60,0,0,0,0,0,130,120,100,90,80,70,60,50,40,30,20,0,0,0,0];
ramping_stimulus_end = [160,140,120,100,90,80,70,60,0,60,0,0,0,0,0,0,0,0,0,50,40,30,20];


stimulus_duration = 500;
Time_position = 700;
number_cell = 200; % number of cells receiving the stimulus


% Number of repetitions for each voice stimulus
trial_number = 60;


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


for Z=1:length(parameters)
    
    %% Variable parameters
    disp(strcat('Subset of parameters n?=',num2str(Z)))
    Iext_PYR = parameters(Z,3); 
    we_PYR_PV = parameters(Z,4);
    wi_PV_PYR = parameters(Z,5);
    
    %% Stimuli
    for F=1:3
        
        if F==1
            type=1;
            stimulus_name=[5,6] ; 
            recording_name='classic_constant';
        elseif F==2
            type=2;
            stimulus_name=[7,8,14,15];
            recording_name='classic_ramp';
        elseif F==3
            type=3;
            stimulus_name=[3,7,10]; 
            recording_name='classic_OU';
        end
        
        disp(recording_name)
        
        for A=stimulus_name
            
            disp(strcat('Stimulus number = ',num2str(A))) ;
            
            Spike_binning_PYR_DBS_PSTH = zeros(trial_number,N_PYR,stimulus_duration/Bin_Length);
            Spike_binning_PV_DBS_PSTH = zeros(trial_number,N_PV,stimulus_duration/Bin_Length);
            Moving_Proba_spike_DBS_PYR_PSTH_Bin5 = zeros(trial_number,stimulus_duration/dt);
            Moving_Proba_spike_DBS_PYR_PSTH_Bin10 = zeros(trial_number,stimulus_duration/dt);
            Moving_Proba_spike_DBS_PV_PSTH_Bin10 = zeros(trial_number,stimulus_duration/dt);
            
           
            for p=1:trial_number

                % Load external stimuli
                cd(path5)
                stimulus = zeros(N_PYR,N_times); % additional stimulus to Pyramidal cells
                if type==1
                    stimulus_intensity = constant_stimulus_intensity(A);
                    receptor_PYR = load(strcat('receptor_cells_constant_',num2str(A)));
                elseif type==2
                    stimulus_intensity = linspace(ramping_stimulus_start(A),ramping_stimulus_end(A),stimulus_duration/dt+1);
                    receptor_PYR = load(strcat('receptor_cells_ramp_',num2str(A)));
                elseif type==3
                    stimulus_intensity = load(strcat('OU_input_',num2str(A))); 
                    receptor_PYR = load(strcat('receptor_cells_OU_',num2str(A)));
                end
                
                
                i=1;
                for w=1:N_PYR
                    if ismember(w,receptor_PYR)==1
                        stimulus(w,Time_position/dt:Time_position/dt + stimulus_duration/dt) = stimulus_intensity;
                    else
                        non_receptor_PYR(i)=w ;
                        i=i+1;
                    end
                end
                
                cd(path1)
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
                    
                    V_PYR_DBS(index_PYR_DBS,i+1)=V_PYR_DBS(index_PYR_DBS,i)+dt*(gL_PYR*(EL-V_PYR_DBS(index_PYR_DBS,i))+gL_PYR*DeltaT_PYR*exp((V_PYR_DBS(index_PYR_DBS,i)-VT_PYR)/DeltaT_PYR) + ge_PYR_DBS(index_PYR_DBS,i).*(Ee-V_PYR_DBS(index_PYR_DBS,i)) + gi_PYR_DBS(index_PYR_DBS,i).*(Ei-V_PYR_DBS(index_PYR_DBS,i)) + Iext_PYR + stimulus(index_PYR_DBS,i) - w_PYR_DBS(index_PYR_DBS,i))/C_PYR + (sigma*sqrt(dt)/sqrt(taum_PYR))*dzeta_PYR(index_PYR_DBS)';
                    w_PYR_DBS(index_PYR_DBS,i+1)=w_PYR_DBS(index_PYR_DBS,i)+dt*(a_PYR*(V_PYR_DBS(index_PYR_DBS,i)-EL) - w_PYR_DBS(index_PYR_DBS,i))/tauw_PYR ;
                    
                    V_PV_DBS(index_PV_DBS,i+1)=V_PV_DBS(index_PV_DBS,i)+dt*(gL_PV*(EL-V_PV_DBS(index_PV_DBS,i))+gL_PV*DeltaT_PV*exp((V_PV_DBS(index_PV_DBS,i)-VT_PV)/DeltaT_PV) + ge_PV_DBS(index_PV_DBS,i).*(Ee-V_PV_DBS(index_PV_DBS,i)) + gi_PV_DBS(index_PV_DBS,i).*(Ei-V_PV_DBS(index_PV_DBS,i)) + Iext_PV  - w_PV_DBS(index_PV_DBS,i))/C_Int + (sigma*sqrt(dt)/sqrt(taum_Int))*dzeta_PV(index_PV_DBS)';
                    w_PV_DBS(index_PV_DBS,i+1)=w_PV_DBS(index_PV_DBS,i)+dt*(a_PV*(V_PV_DBS(index_PV_DBS,i)-EL) - w_PV_DBS(index_PV_DBS,i))/tauw_PV ;
                    
                    V_SOM_DBS(index_SOM_DBS,i+1)=V_SOM_DBS(index_SOM_DBS,i)+dt*(gL_SOM*(EL-V_SOM_DBS(index_SOM_DBS,i))+gL_SOM*DeltaT_SOM*exp((V_SOM_DBS(index_SOM_DBS,i)-VT_SOM)/DeltaT_SOM) + gi_SOM_DBS(index_SOM_DBS,i).*(Ei-V_SOM_DBS(index_SOM_DBS,i))+ Iext_SOM  - w_SOM_DBS(index_SOM_DBS,i))/C_Int + (sigma*sqrt(dt)/sqrt(taum_Int))*dzeta_SOM(index_SOM_DBS)';
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
                
                % During stimulus presentation
                PYR_spike_number_DBS = length(find(V_PYR_DBS(:,14000:24000)==Amp_spike));
                PV_spike_number_DBS = length(find(V_PV_DBS(:,14000:24000)==Amp_spike));
                SOM_spike_number_DBS = length(find(V_SOM_DBS(:,14000:24000)==Amp_spike));
                
                mean_firing_rate_PYR_DBS(A,p) = PYR_spike_number_DBS/((stimulus_duration)*1e-3*N_PYR); % retrait des 200 premi?res ms
                mean_firing_rate_PV_DBS(A,p) = PV_spike_number_DBS/((stimulus_duration)*1e-3*N_PV);
                mean_firing_rate_SOM_DBS(A,p) = SOM_spike_number_DBS/((stimulus_duration)*1e-3*N_SOM);
                
                V_PYR_binary_DBS=zeros(N_PYR,N_times); V_PYR_binary_DBS(V_PYR_DBS==Amp_spike)=1;
                V_PV_binary_DBS=zeros(N_PV,N_times);   V_PV_binary_DBS(V_PV_DBS==Amp_spike)=1;
                
                % Spike binning for PSTH
                Spike_binned_DBS_PSTH=zeros(N_PYR,ceil(stimulus_duration/Bin_Length)-1);
                v=1;
                for z=0:((stimulus_duration)/Bin_Length-1)
                    Spike_binned_DBS_PSTH(:,v)=sum(V_PYR_binary_DBS(:,(Time_position)/dt+z*Bin_Length/dt:(Time_position)/dt+(z+1)*Bin_Length/dt),2);
                    v=v+1;
                end
                Spike_binned_DBS_PSTH=vertcat(Spike_binned_DBS_PSTH(receptor_PYR,:),Spike_binned_DBS_PSTH(non_receptor_PYR,:));
                Spike_binning_PYR_DBS_PSTH(p,:,:) = Spike_binned_DBS_PSTH;
                
                Spike_binned_PV_DBS_PSTH=zeros(N_PV,ceil(stimulus_duration/Bin_Length)-1);
                v=1;
                for z=0:((stimulus_duration)/Bin_Length-1)
                    Spike_binned_PV_DBS_PSTH(:,v)=sum(V_PV_binary_DBS(:,(Time_position)/dt+z*Bin_Length/dt:(Time_position)/dt+(z+1)*Bin_Length/dt),2);
                    v=v+1;
                end
                Spike_binning_PV_DBS_PSTH(p,:,:) = Spike_binned_PV_DBS_PSTH;
                
                %% Average metric of population response to various stimuli
                Count_Spike_DBS_PSTH=[];
                Count_Spike_DBS_PSTH = movsum(V_PYR_binary_DBS(:,(Time_position)/dt:(Time_position+stimulus_duration)/dt-1),Bin_Length/dt,2);
                Moving_Proba_spike_DBS_PYR_PSTH_Bin5(p,:) = sum(Count_Spike_DBS_PSTH,1);
                
                Count_Spike_DBS_PSTH=[];
                Count_Spike_DBS_PSTH = movsum(V_PYR_binary_DBS(:,(Time_position)/dt:(Time_position+stimulus_duration)/dt-1),2*Bin_Length/dt,2);
                Moving_Proba_spike_DBS_PYR_PSTH_Bin10(p,:) = sum(Count_Spike_DBS_PSTH,1);
                
                Count_Spike_DBS_PSTH=[];
                Count_Spike_DBS_PSTH = movsum(V_PV_binary_DBS(:,(Time_position)/dt:(Time_position+stimulus_duration)/dt-1),2*Bin_Length/dt,2);
                Moving_Proba_spike_DBS_PV_PSTH_Bin10(p,:) = sum(Count_Spike_DBS_PSTH,1);
                
                
            end
            
            cd(path2)
            dlmwrite(strcat('V_PYR_binned_PSTH_Params_',num2str(Z),'_',recording_name,'_',num2str(A)),Spike_binning_PYR_DBS_PSTH)
            dlmwrite(strcat('V_PV_binned_PSTH_Params_',num2str(Z),'_',recording_name,'_',num2str(A)),Spike_binning_PV_DBS_PSTH)
            dlmwrite(strcat('Moving_Proba_PYR_PSTH_Bin10_Params_',num2str(Z),'_',recording_name,'_',num2str(A)),Moving_Proba_spike_DBS_PYR_PSTH_Bin10)
            dlmwrite(strcat('Moving_Proba_PV_PSTH_Bin10_Params_',num2str(Z),'_',recording_name,'_',num2str(A)),Moving_Proba_spike_DBS_PV_PSTH_Bin10)
            dlmwrite(strcat('Moving_Proba_PYR_PSTH_Bin5_Params_',num2str(Z),'_',recording_name,'_',num2str(A)),Moving_Proba_spike_DBS_PYR_PSTH_Bin5)
            
        end
                
    end
    
end

