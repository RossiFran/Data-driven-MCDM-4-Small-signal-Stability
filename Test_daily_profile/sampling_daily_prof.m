clear all
close all
clc

%% Load Open Data Denmark RES generation and consumption: 24H, every 15 min data for the day 14/07/2020
open_data_DE=readtable('time_series_15min_singleindex_filtered.csv')

%% Adapt Open Data measures to the test system
%Generation
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
P2_intra_day=P2_day_ahead-0.055*Pnom_gens(2);
P2_intra_day(P2_intra_day<0)=0.05*Pnom_gens(2);

pf_gen_max= 0.95;% power factor (cosphi)
pf_gen_min= 0.8;% power factor (cosphi)

new_samples=lhsdesign(nlhs,n_gens);
pf_gen_arr(:,1)=(pf_gen_max-pf_gen_min)*new_samples(:,1)+pf_gen_min;
pf_gen_arr(:,2)=(pf_gen_max-pf_gen_min)*new_samples(:,2)+pf_gen_min;
pf_gen_arr(:,3)=(pf_gen_max-pf_gen_min)*new_samples(:,3)+pf_gen_min;

Q1_intra_day=P1_intra_day(:,1).*tan(acos(pf_gen_arr(:,1)));
Q2_intra_day=P2_intra_day(:,1).*tan(acos(pf_gen_arr(:,2)));
Q3_intra_day=P3_intra_day(:,1).*tan(acos(pf_gen_arr(:,3)));

%Demand
n_loads=4;
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

Pdemand_unscaled=open_data_DE.DE_load_actual_entsoe_transparency*1e6;

PL_min=200e6; %minimum total demand
PL_max=700e6; %maximum totoal demand

alpha_Pdemand= (Pdemand_unscaled-min(Pdemand_unscaled))/(max(Pdemand_unscaled)-min(Pdemand_unscaled));
Pdemand_dayahead=alpha_Pdemand*(PL_max-PL_min)+PL_min;

PL2_dayahead=-stress_dir(:,1).*Pdemand_dayahead;
PL5_dayahead=-stress_dir(:,2).*Pdemand_dayahead;
PL7_dayahead=-stress_dir(:,3).*Pdemand_dayahead;
PL9_dayahead=-stress_dir(:,4).*Pdemand_dayahead;

pf_loads=0.8; % power factor
pf_tg_loads=tan(acos(pf_loads));

QL2_dayahead=pf_tg_loads*PL2_dayahead;
QL5_dayahead=pf_tg_loads*PL5_dayahead;
QL7_dayahead=pf_tg_loads*PL7_dayahead;
QL9_dayahead=pf_tg_loads*PL9_dayahead;

day_ahead_intra_day_forecast=table();
day_ahead_intra_day_forecast.P1_day_ahead=P1_day_ahead;
day_ahead_intra_day_forecast.P2_day_ahead=P2_day_ahead;
day_ahead_intra_day_forecast.P3_day_ahead=P3_day_ahead;
day_ahead_intra_day_forecast.P1_intra_day=P1_intra_day;
day_ahead_intra_day_forecast.P2_intra_day=P2_intra_day;
day_ahead_intra_day_forecast.P3_intra_day=P3_intra_day;
day_ahead_intra_day_forecast.Q1_intra_day=Q1_intra_day;
day_ahead_intra_day_forecast.Q2_intra_day=Q2_intra_day;
day_ahead_intra_day_forecast.Q3_intra_day=Q3_intra_day;

day_ahead_intra_day_forecast.Pdemand_dayahead=Pdemand_dayahead;
day_ahead_intra_day_forecast.PL2_dayahead=PL2_dayahead;
day_ahead_intra_day_forecast.PL5_dayahead=PL5_dayahead;
day_ahead_intra_day_forecast.PL7_dayahead=PL7_dayahead;
day_ahead_intra_day_forecast.QL2_dayahead=QL2_dayahead;
day_ahead_intra_day_forecast.QL5_dayahead=QL5_dayahead;
day_ahead_intra_day_forecast.QL7_dayahead=QL7_dayahead;
day_ahead_intra_day_forecast.QL9_dayahead=QL9_dayahead;

writetable(day_ahead_intra_day_forecast,'DayAhead_IntraDay_Forecast.xlsx');
