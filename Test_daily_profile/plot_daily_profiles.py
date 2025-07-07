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

forecast=pd.read_excel('DayAhead_IntraDay_Forecast.xlsx')

#Day Ahead (DA) forecast
Pg1_DA=forecast['P1_day_ahead']/1e6
Pg2_DA=forecast['P2_day_ahead']/1e6
Pg3_DA=forecast['P3_day_ahead']/1e6

Pltot=forecast['Pdemand_dayahead']/1e6

#Intra Day (ID) forecast
Pg1_ID=forecast['P1_intra_day']/1e6
Pg2_ID=forecast['P2_intra_day']/1e6
Pg3_ID=forecast['P3_intra_day']/1e6


#%%

time_intervals = [f"{hour}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 15)]

colors=["#A2142F","#4DBEEE","#77AC30","#7E2F8E","#EDB120","#D95319","#0072BD"]
fig=plt.figure(figsize=(8,5))
ax=fig.add_subplot()
ax.plot(time_intervals,Pg1_DA, linewidth=4, color=colors[0], label='On-shore\nwind ($G_1$) DA')
ax.plot(time_intervals,Pg2_DA, linewidth=4,color=colors[1],  label='PV ($G_2$) DA')
ax.plot(time_intervals,Pg3_DA, linewidth=4,color=colors[2],  label='Off-shore\nwind ($G_3$) DA')
ax.plot(time_intervals,Pltot, linewidth=4,color=colors[3],  label='Total\nDemand')

ax.plot(time_intervals,Pg1_ID, '--',alpha=0.3, color=colors[0], linewidth=4, label='On-shore\nwind ($G_1$) ID')
ax.plot(time_intervals,Pg2_ID, '--',alpha=0.3,color=colors[1],  linewidth=4, label='PV ($G_2$) ID')
ax.plot(time_intervals,Pg3_ID,'--',alpha=0.3, color=colors[2], linewidth=4, label='Off-shore\nwind ($G_3$)ID')

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

