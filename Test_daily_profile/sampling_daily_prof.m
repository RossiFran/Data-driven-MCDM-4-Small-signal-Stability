open_data_DE=readtable('time_series_15min_singleindex_filtered.csv')

n_loads=4;
n_gens=3;
nlhs=height(open_data_DE);

Pnom_gens=[300e6, 100e6, 150e6]; % Nominal power of generators: G1: wind on-shore, G2: PV, G3: wind off-shore
GENs_min= 0.05*[Pnom_gens]; % Minimum active power: 5% of nominal power
GENs_max= 0.95*[Pnom_gens]; % Maximum active power: 95% of nominal power

alpha_P1=(open_data_DE.DE_wind_onshore_generation_actual*1e6-min(open_data_DE.DE_wind_onshore_generation_actual*1e6))/(max(open_data_DE.DE_wind_onshore_generation_actual*1e6)-min(open_data_DE.DE_wind_onshore_generation_actual*1e6));
alpha_P3=(open_data_DE.DE_wind_offshore_generation_actual*1e6-min(open_data_DE.DE_wind_offshore_generation_actual*1e6))/(max(open_data_DE.DE_wind_offshore_generation_actual*1e6)-min(open_data_DE.DE_wind_offshore_generation_actual*1e6));
alpha_P2=(open_data_DE.DE_solar_generation_actual*1e6-min(open_data_DE.DE_solar_generation_actual*1e6))/(max(open_data_DE.DE_solar_generation_actual*1e6)-min(open_data_DE.DE_solar_generation_actual*1e6));

P1_day_ahead= alpha_P1*(GENs_max(1)-GENs_min(1))+GENs_min(1);
P3_day_ahead= alpha_P3*(GENs_max(3)-GENs_min(3))+GENs_min(3);
P2_day_ahead= alpha_P2*(GENs_max(2)-GENs_min(2))+GENs_min(2);


wind_forecast_std= 0.0534 % according to B.-M. Hodge, D. Lew, M. Milligan, E. Gómez-Lázaro, X. G. Larsén,
                                        % G. Giebel, H. Holttinen, S. Sillanpää, R. Scharff, L. Söder, et al., Wind
                                        % power forecasting error distributions: An international comparison, in:
                                        % 11th International Workshop on Large-Scale Integration of Wind Power
                                        % into Power Systems as well as on Transmission Networks for Offshore Wind
                                        % Power Plants, 2012.

P1_intra_day=P1_day_ahead-2*wind_forecast_std*Pnom_gens(1);
P1_intra_day(P1_intra_day<0)=0.05*Pnom_gens(1);
P3_intra_day=P3_day_ahead-2*wind_forecast_std*Pnom_gens(3);
P3_intra_day(P3_intra_day<0)=0.05*Pnom_gens(3);

P2_day_ahead=P2_day_ahead-0.055*Pnom_gens(2);
P2_day_ahead(P2_day_ahead<0)=0.05*Pnom_gens(2);

pf_gen_max= 0.95;
pf_gen_min= 0.8;

new_samples=lhsdesign(nlhs,n_var);
pf_gen_arr(:,1)=(pf_gen_max-pf_gen_min)*new_samples(:,1)+pf_gen_min;
pf_gen_arr(:,2)=(pf_gen_max-pf_gen_min)*new_samples(:,2)+pf_gen_min;
pf_gen_arr(:,3)=(pf_gen_max-pf_gen_min)*new_samples(:,3)+pf_gen_min;

Q3g_arr=P1_intra_day(:,1).*tan(acos(pf_gen_arr(:,1)));
Q6g_arr=P2_day_ahead(:,1).*tan(acos(pf_gen_arr(:,2)));
Q11g_arr=P3_intra_day(:,1).*tan(acos(pf_gen_arr(:,3)));

new_mmc3(:,1)=new_samples(:,4).*(MMC3_max-MMC3_min)+MMC3_min;

isd=[30,20,20,30];
isd_up=isd*1.3;
isd_lw=isd*0.7;

loads_stress_dir_0=lhsdesign(height(open_data_DE),n_loads,'Criterion','maximin');
loads_stress_dir=(isd_up-isd_lw).*loads_stress_dir_0+isd_lw;

clear stress_dir

for csd=1:height(open_data_DE)
    sum_loads_sd(csd,1)=sum(loads_stress_dir(csd,1:end));
    stress_dir(csd,:)=loads_stress_dir(csd,:)/sum_loads_sd(csd,1);
end

Pload_tot_vect_unscaled=open_data_DE.DE_load_actual_entsoe_transparency*1e6;

PL_min=200e6;
PL_max=700e6;

Pload_tot_vect_std= (Pload_tot_vect_unscaled-min(Pload_tot_vect_unscaled))/(max(Pload_tot_vect_unscaled)-min(Pload_tot_vect_unscaled));
Pload_tot_vect=Pload_tot_vect_std*(PL_max-PL_min)+PL_min;

P2l_arr=-stress_dir(:,1).*Pload_tot_vect;
P5l_arr=-stress_dir(:,2).*Pload_tot_vect;
P7l_arr=-stress_dir(:,3).*Pload_tot_vect;
P9l_arr=-stress_dir(:,4).*Pload_tot_vect;

pf_loads=0.8;
pf_tg_loads=tan(acos(pf_loads));

Q2l_arr=pf_tg_loads*P2l_arr;
Q5l_arr=pf_tg_loads*P5l_arr;
Q7l_arr=pf_tg_loads*P7l_arr;
Q9l_arr=pf_tg_loads*P9l_arr;

Pacref_n3_arr=new_mmc3;

sampling_struct.Pacref_n3_arr=Pacref_n3_arr;
sampling_struct.P2l_arr=P2l_arr;
sampling_struct.Q2l_arr=Q2l_arr;
sampling_struct.P5l_arr=P5l_arr;
sampling_struct.Q5l_arr=Q5l_arr;
sampling_struct.P7l_arr=P7l_arr;
sampling_struct.Q7l_arr=Q7l_arr;
sampling_struct.P9l_arr=P9l_arr;
sampling_struct.Q9l_arr=Q9l_arr;
sampling_struct.P3g_arr=P1_intra_day;
sampling_struct.Q3g_arr=Q3g_arr;
sampling_struct.P6g_arr=P2_day_ahead;
sampling_struct.Q6g_arr=Q6g_arr;
sampling_struct.P11g_arr=P3_intra_day;
sampling_struct.Q11g_arr=Q11g_arr;
%sampling_struct.instance=[1:length(Pload_tot_vect)];