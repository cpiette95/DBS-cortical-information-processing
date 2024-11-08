%% Code for running simulations using as external inputs Naturalistic stimuli from audio-recordings

clear

path1='/Networks';
path2='/Saved_results'; % where to save the matrices

% Simulations made on a subset of network configurations: 
trial_test=[2,5,7,17,26,40,41,52,58,60,62,67,72,81,85,87,91,104,105,112,134,138,139,151,166,174,180,184,188,200,219,222,223,246,260,268,273,289,317,303,304,321,334,3383,58];


dt = 0.05;
T = 2000;
N_times=ceil(T/dt);
Time_vector = 0:dt:T;
Bin_Length = 5; % ms bin 

N_PYR = 800;
N_SOM = 80;
N_PV = 120;
N_total = N_PYR + N_SOM + N_PV;


% Range of DBS parameters investigated
DBS_intensity=[0,200,400,600,800,1000,1200] ;
DBS_frequency=[110,130,150,170,190,210];  

stimulus_duration = 1000;
Time_position = 700;
number_cell = 200; % number of cells receiving the stimulus


% Number of repetitions for each voice stimulus
trial_number = 20;


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


folder={'Bertrand','Bertrand','Bertrand','Charlie','Charlie','Charlie','Charlotte','Charlotte','Charlotte','Charlotte','Charlotte','Elodie','Elodie','Elodie','Hannah','Hannah','Hannah','Jonathan','Jonathan','Jonathan','Jonathan','Laurent','Laurent','Laurent','Simon','Simon','Sylvie','Sylvie','Sylvie'};
stimulus_name = {'New_Bertrand_Avicenne','New_Bertrand_Habite','New_Bertrand_voix','New_Charlie_Brandeis_fr','New_Charlie_Paris','New_Charlie_enregistre','New_Charlotte_Brandeis','New_Charlotte_College','New_Waltham_VF','New_Paris_VF','New_Voix_charlotte','New_Elodie_College','New_Elodie_Habite','New_Elodie_voix','New_hannah_fr_Brandeis','New_hannah_fr_live','New_hannah_fr_ma_voix','New_Jonathan_Brandeis','New_Jonathan_Brookline','New_Jonathan_Nice','New_Jonathan_Voix','New_Laurent_College','New_Laurent_Habite','New_Laurent_voix','New_Simon_Brandeis','New_Simon_voix','New_Sylvie_College','New_Sylvie_Habite','New_Sylvie_voix'};
stimulus_intensity=2;


parfor Y=1:length(stimulus_name)
    
    disp(stimulus_name(Y))
    
    for N=1:length(trial_test)
        
        disp(trial_test(N))
                
        for Z=1:length(DBS_intensity)
            
            for R=1:length(DBS_frequency);
                
                cd(path1)
                Connect_PYR_PYR = load('Connect_PYR_PYR');
                Connect_PV_PV = load('Connect_PV_PV');
                Connect_SOM_SOM = load('Connect_SOM_SOM');
                Connect_PYR_PV = load('Connect_PYR_PV');
                Connect_PV_PYR = load('Connect_PV_PYR');
                Connect_SOM_PYR = load('Connect_SOM_PYR');
                Connect_SOM_PV = load('Connect_SOM_PV');
                Connect_PV_SOM = load('Connect_PV_SOM');
                receptor_PYR = load(strcat('receptor_natural_sounds'));
                
                
                Spike_binning_PYR_DBS_PSTH = zeros(trial_number,N_PYR,stimulus_duration/Bin_Length);
                Spike_binning_PV_DBS_PSTH = zeros(trial_number,N_PV,stimulus_duration/Bin_Length);
                Moving_Proba_spike_DBS_PYR_PSTH_Bin5 = zeros(trial_number,stimulus_duration/dt);
                Moving_Proba_spike_DBS_PYR_PSTH_Bin10 = zeros(trial_number,stimulus_duration/dt);
                Moving_Proba_spike_DBS_PV_PSTH_Bin10 = zeros(trial_number,stimulus_duration/dt);
                
                % DBS parameters (pA)
                I0_PYR = DBS_intensity(Z);
                I0_PV = DBS_intensity(Z);
                I0_SOM = DBS_intensity(Z);
                I0_PYR_bis=DBS_intensity(Z);
                delay_PV = 0.5;
                delay_SOM = 2;
                delay_PYR = 2;
                frequency = 1e-3*DBS_frequency(R);
                pulse_duration = 2;
                proportion_activated=0.5;
                
                pre_stim_PYR_template = vertcat(ones(round(pulse_duration/dt),1),zeros(round(1/(frequency*dt)-round(pulse_duration/dt)),1));
                stim_PYR_template = repmat(I0_PYR*pre_stim_PYR_template',1,ceil(ceil(T/dt)/length(pre_stim_PYR_template)));
                stim_PV_template = repmat(I0_PV*pre_stim_PYR_template',1,ceil(ceil(T/dt)/length(pre_stim_PYR_template)));
                
                pre_stim_PYR_delayed_template = vertcat(zeros(round(delay_PYR/dt),1),ones(round(pulse_duration/dt),1),zeros(round(1/(frequency*dt) - round(delay_PYR/dt) - round(pulse_duration/dt)),1));
                stim_PYR_delayed_template = repmat(I0_PYR_bis*pre_stim_PYR_delayed_template',1,ceil(ceil(T/dt)/length(pre_stim_PYR_delayed_template)));
                
                pre_stim_SOM_template = vertcat(zeros(round(delay_SOM/dt),1),ones(round(pulse_duration/dt),1),zeros(round(1/(frequency*dt) - round(delay_SOM/dt) - round(pulse_duration/dt)),1));
                stim_SOM_template = repmat(I0_SOM*pre_stim_SOM_template',1,ceil(ceil(T/dt)/length(pre_stim_SOM_template)));
                
                stimulus_total_PYR = [repmat(stim_PYR_template,proportion_activated*ceil(N_PYR/2),1) ; repmat(stim_PYR_delayed_template,proportion_activated*ceil(N_PYR/2),1) ; zeros(N_PYR - proportion_activated*N_PYR,length(stim_PYR_template))];
                stimulus_total_PV = [repmat(stim_PV_template,proportion_activated*N_PV,1) ; zeros(N_PV - proportion_activated*N_PV,length(stim_PV_template))];
                stimulus_total_SOM = [repmat(stim_SOM_template,proportion_activated*N_SOM,1); zeros(N_SOM - proportion_activated*N_SOM,length(stim_SOM_template))];
                
                cd(path1)
                parameters=load('parameters');
                
                disp(strcat('DBS intensity=',num2str(DBS_intensity(Z))))
                disp(strcat('DBS frequency=',num2str(DBS_frequency(R))))
                
                for p=1:trial_number
                    
                    
                    Iext_PYR = parameters(trial_test(N),3);
                    we_PYR_PV = parameters(trial_test(N),4);
                    wi_PV_PYR = parameters(trial_test(N),5);

                    input = load(strcat(stimulus_name{Y},num2str(p)));
                    
                    non_receptor_PYR=[];
                    stimulus = zeros(N_PYR,N_times); % additional stimulus to Pyramidal cells
                    i=1; j=1;
                    m=1;
                    for w=1:N_PYR
                        if ismember(w,receptor_PYR)==1
                            stimulus(w,Time_position/dt:Time_position/dt + stimulus_duration/dt -1) = stimulus_intensity*input(m,:);
                            stimulus(stimulus<0)=0;
                            j=j+1;
                            if mod(j,10)==1;
                                m=m+1;
                            end
                        else
                            non_receptor_PYR(i)=w ;
                            i=i+1;
                        end
                    end
                    
                    
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
                        
                        V_PYR_DBS(index_PYR_DBS,i+1)=V_PYR_DBS(index_PYR_DBS,i)+dt*(gL_PYR*(EL-V_PYR_DBS(index_PYR_DBS,i))+gL_PYR*DeltaT_PYR*exp((V_PYR_DBS(index_PYR_DBS,i)-VT_PYR)/DeltaT_PYR) + ge_PYR_DBS(index_PYR_DBS,i).*(Ee-V_PYR_DBS(index_PYR_DBS,i)) + gi_PYR_DBS(index_PYR_DBS,i).*(Ei-V_PYR_DBS(index_PYR_DBS,i)) + Iext_PYR + stimulus_total_PYR(index_PYR_DBS,i) + stimulus(index_PYR_DBS,i) - w_PYR_DBS(index_PYR_DBS,i))/C_PYR + (sigma*sqrt(dt)/sqrt(taum_PYR))*dzeta_PYR(index_PYR_DBS)';
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
                    
                    
                    
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Rapid display for Raster plots
                    %             PYR_spike_number_DBS = length(find(V_PYR_DBS(:,4000:end)==Amp_spike)); % retrait des 200 premi?res ms
                    %             PV_spike_number_DBS = length(find(V_PV_DBS(:,4000:end)==Amp_spike));
                    %             SOM_spike_number_DBS = length(find(V_SOM_DBS(:,4000:end)==Amp_spike));
                    %
                    %             idx = find(V_PYR_DBS==Amp_spike);
                    %             [row_PYR_DBS,col]=ind2sub(size(V_PYR_DBS),idx);
                    %             PYR_spike_time_DBS = col*dt;
                    %             idx = find(V_PV_DBS==Amp_spike);
                    %             [row_PV_DBS,col]=ind2sub(size(V_PV_DBS),idx);
                    %             PV_spike_time_DBS = col*dt;
                    %             idx = find(V_SOM_DBS==Amp_spike);
                    %             [row_SOM_DBS,col]=ind2sub(size(V_SOM_DBS),idx);
                    %             SOM_spike_time_DBS = col*dt;
                    %
                    %             mean_firing_rate_PYR_DBS = PYR_spike_number_DBS/((T-200)*1e-3*N_PYR); % retrait des 200 premi?res ms
                    %             disp(['PYR =', num2str(mean_firing_rate_PYR_DBS)])
                    %             mean_firing_rate_PV_DBS = PV_spike_number_DBS/((T-200)*1e-3*N_PV);
                    %             disp(['PV =', num2str(mean_firing_rate_PV_DBS)])
                    %             mean_firing_rate_SOM_DBS = SOM_spike_number_DBS/((T-200)*1e-3*N_SOM);
                    %             disp(['SOM =', num2str(mean_firing_rate_SOM_DBS)])
                    %
                    %             figure()
                    %             plot(PYR_spike_time_DBS,row_PYR_DBS,'m.','MarkerSize',10)
                    %             hold on
                    %             plot(PV_spike_time_DBS,N_PYR+row_PV_DBS,'g.','MarkerSize',10)
                    %             plot(SOM_spike_time_DBS,N_PYR+N_PV+row_SOM_DBS,'b.','MarkerSize',10)
                    %             xlabel('Time (ms)')
                    %             ylabel('Neuron index')
                    %             ylim([0 1010])
                    %             %title('WITH DBS')
                    %             hold off;
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    
                    % Analysis During stimulus presentation
                    PYR_spike_number_DBS = length(find(V_PYR_DBS(:,14000:34000)==Amp_spike));
                    PV_spike_number_DBS = length(find(V_PV_DBS(:,14000:34000)==Amp_spike));
                    SOM_spike_number_DBS = length(find(V_SOM_DBS(:,14000:34000)==Amp_spike));
                    
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
                    
                    % Average metric of population response to various stimuli
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
                dlmwrite(strcat('Modele_',num2str(trial_test(N)),'_V_PYR_binned_PSTH_DBS_',num2str(DBS_intensity(Z)),'pA_',num2str(DBS_frequency(R)),'Hz_stimulus_intensity_2_',stimulus_name{Y}),Spike_binning_PYR_DBS_PSTH)
                
            end
        end
    end
    
end

