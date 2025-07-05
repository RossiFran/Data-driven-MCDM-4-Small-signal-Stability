import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#%%
fs=20
plt.rcParams.update({"figure.figsize" : [8, 6],
                     "text.usetex": True,
                     "font.family": "serif",
                     "font.serif": "Computer Modern",
                     "axes.labelsize": fs,
                     "axes.titlesize": fs,
                     "legend.fontsize": fs,
                     "xtick.labelsize": fs,
                     "ytick.labelsize": fs,
                     "savefig.dpi": 130,
                    'legend.fontsize': fs,
                    'legend.handlelength': 2,
                    'legend.loc': 'upper right'})
#%%
day_pf=pd.read_excel('PFtab_red_daily_prof.xlsx')
day_pf_dev=pd.read_excel('G://Il mio Drive/gridofgridsML-main/gridofgridsML-main/Graph GoG_v5/Nominal/Data_30_12_2022/DailyProf_V2/PFtab_red_daily_prof_comb6.xlsx')

Pg1=day_pf['Pg1']/1e6
Pg2=day_pf['Pg2']/1e6
Pg3=day_pf['Pg3']/1e6

Pl1=day_pf['P2l']/1e6
Pl2=day_pf['P5l']/1e6
Pl3=day_pf['Pl7']/1e6
Pl4=day_pf['P9l']/1e6

Pltot=Pl1+Pl2+Pl3+Pl4

Pg1_dev=day_pf_dev['Pg1']/1e6
Pg2_dev=day_pf_dev['Pg2']/1e6
Pg3_dev=day_pf_dev['Pg3']/1e6

Pl1_dev=day_pf_dev['P2l']/1e6
Pl2_dev=day_pf_dev['P5l']/1e6
Pl3_dev=day_pf_dev['Pl7']/1e6
Pl4_dev=day_pf_dev['P9l']/1e6

Pltot_dev=Pl1_dev+Pl2_dev+Pl3_dev+Pl4_dev

time_intervals = [f"{hour}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 15)]

#%%
colors=["#A2142F","#4DBEEE","#77AC30","#7E2F8E","#EDB120","#D95319","#0072BD"]
fig=plt.figure(figsize=(8,5))
ax=fig.add_subplot()
ax.plot(time_intervals,Pg1_dev, linewidth=4, color=colors[0], label='On-shore\nwind ($G_1$)')
ax.plot(time_intervals,Pg2_dev, linewidth=4,color=colors[1],  label='PV ($G_2$)')
ax.plot(time_intervals,Pg3_dev, linewidth=4,color=colors[2],  label='Off-shore\nwind ($G_3$)')
ax.plot(time_intervals,-Pltot_dev, linewidth=4,color=colors[3],  label='Total\nDemand')

ax.plot(time_intervals,Pg1, '--',alpha=0.3, color=colors[0], linewidth=4, label='1 day-ahead\nforecast')
ax.plot(time_intervals,Pg2, '--',alpha=0.3,color=colors[1],  linewidth=4, label='Intra-day\nforecast')
ax.plot(time_intervals,Pg3,'--',alpha=0.3, color=colors[2], linewidth=4)
ax.plot(time_intervals,-Pltot,'--',alpha=0.1, color=colors[3],  linewidth=4)

# ax.plot(time_intervals,Pl1, linewidth=4)
# ax.plot(time_intervals,Pl2, linewidth=4)
# ax.plot(time_intervals,Pl3, linewidth=4)
# ax.plot(time_intervals,Pl4, linewidth=4)
ax.set_ylabel('P [MW]')
ax.set_xticks(time_intervals)
time_intervals_lab=[t if ':00' in t else '' for t in time_intervals]
time_intervals_lab=[t if t=='0:00' or t == '4:00' or t=='8:00' or t=='12:00' or t=='16:00' or t=='20:00' else '' for t in time_intervals_lab]
ax.set_xticklabels(time_intervals_lab, rotation=45)
# Enable grid for major ticks only
ax.grid(True, which='major')#, linestyle='--', linewidth=0.7, alpha=0.7)
ax.minorticks_on()
# Show major ticks
ax.xaxis.set_major_locator(plt.MultipleLocator(4))  # Set major tick intervals
ax.set_xlim(0,96)
ax.set_title('Generation and Demand Forecast')
plt.tight_layout(rect=[0, 0.3, 1, 1])  # Reserve space for the legend

fig.legend(loc="lower center",ncols=3)#, bbox_to_anchor=(1.3, 1.1))
#%%
plt.savefig('DayGenDem.pdf', format='pdf')
plt.savefig('DayGenDem.png', format='png')
#%%
colors=["#EDB120","#D95319","#0072BD","#77AC30"]
fig=plt.figure(figsize=(8,5))
ax=fig.add_subplot()
ax.plot(time_intervals,-Pl1_dev, linewidth=4, color=colors[0], label='$L_1$')
ax.plot(time_intervals,-Pl2_dev, linewidth=4,color=colors[1],  label='$L_2$')
ax.plot(time_intervals,-Pl3_dev, linewidth=4,color=colors[2],  label='$L_3$')
ax.plot(time_intervals,-Pl4, linewidth=4,color=colors[3],  label='$L_4$')

ax.plot(time_intervals,-Pl1, '--',alpha=0.3, color=colors[0], linewidth=4, label='1 day-ahead\nforecast')
ax.plot(time_intervals,-Pl2, '--',alpha=0.3,color=colors[1],  linewidth=4, label='Intra-day\nforecast')
ax.plot(time_intervals,-Pl3,'--',alpha=0.3, color=colors[2], linewidth=4)
ax.plot(time_intervals,-Pl4,'--',alpha=0.1, color=colors[3],  linewidth=4)

# ax.plot(time_intervals,Pl1, linewidth=4)
# ax.plot(time_intervals,Pl2, linewidth=4)
# ax.plot(time_intervals,Pl3, linewidth=4)
# ax.plot(time_intervals,Pl4, linewidth=4)
ax.set_ylabel('P [MW]')
ax.set_xticks(time_intervals)
time_intervals_lab=[t if ':00' in t else '' for t in time_intervals]
time_intervals_lab=[t if t=='0:00' or t == '4:00' or t=='8:00' or t=='12:00' or t=='16:00' or t=='20:00' else '' for t in time_intervals_lab]
ax.set_xticklabels(time_intervals_lab, rotation=45)
# Enable grid for major ticks only
ax.grid(True, which='major')#, linestyle='--', linewidth=0.7, alpha=0.7)
ax.minorticks_on()
# Show major ticks
ax.xaxis.set_major_locator(plt.MultipleLocator(4))  # Set major tick intervals
ax.set_xlim(0,96)
ax.set_title('Daily Generation and Demand Profile')
plt.tight_layout(rect=[0, 0.3, 1, 1])  # Reserve space for the legend

fig.legend(loc="lower center",ncols=3)#, bbox_to_anchor=(1.3, 1.1))
