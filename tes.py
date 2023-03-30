import pandas as pd
import matplotlib.pyplot as plt
from cb_set_pars import hp_settings, orc_settings
from cb_network import hp_network, orc_network
from tes_func import time_analysis_TES
from config import V, P_out, P_in, P_min, rho


hp = hp_network(['R245fa', 'water'])
orc = orc_network(['R245fa', 'water'])
hp_settings(hp, P_in)
orc_settings(orc, P_out)
hp.solve(mode='design')
hp_s = hp.results['Connection']
orc.solve(mode='design')
orc_s = orc.results['Connection']

V_in_t = hp_s.loc["7", "m"] / rho * 3600  # [m^3/h] charging volume stream
V_out_t = orc_s.loc["17", "m"] / rho * 3600  # [m^3/h] discharging volume stream
t_charge = V / V_in_t  # [h] charging time
t_discharge = V / V_out_t  # [h] discharging time

plt.rcParams.update({'font.size': 14})

df_2022 = pd.read_csv("wind_pv_entose_DE_2022.csv", index_col=0, parse_dates=[0])  # [MW]
df_2022 = df_2022 / 1e3  # [GW]
df_2022["RES"] = df_2022["Wind Offshore"] + df_2022["Wind Onshore"] + df_2022["Solar"]
df_2022["Residual"] = df_2022["Last"] - df_2022["RES"]
df_2022["RES new"] = df_2022["RES"] * 2  # assumption: wind and pv capacity doubled
df_2022["Residual new"] = df_2022["Last"] - df_2022["RES new"]
df_2022["Residual for CB"] = df_2022["Residual new"] / df_2022["Residual new"].max() * P_in / 1e6  # [MW]

fig1 = plt.figure(figsize=(14, 7))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(df_2022["RES"].resample("D").mean(), color="green")
ax1.plot(df_2022["Residual"].resample("D").mean(), color="black")
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.set_ylabel('Power [GW]')
ax1.legend(["RES", "Load"])
ax1.set_ylim(0, 100)
ax1.set_title("Residual load and power generation from RES in Germany (daily mean, 2022)", fontsize=16, pad=20)
plt.show()

fig2 = plt.figure(figsize=(14, 7))
ax1 = fig2.add_subplot(1, 1, 1)
ax1.plot(df_2022["RES new"].resample("D").mean(), color="green")
ax1.plot(df_2022["Residual"].resample("D").mean(), color="black")
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.set_ylabel('Power [GW]')
ax1.legend(["RES", "Load"])
ax1.set_ylim(0, 100)
ax1.set_title("Residual load and power generation from RES in Germany (daily mean, future scenario)", fontsize=16, pad=20)
plt.show()

fig3 = plt.figure(figsize=(14, 7))
ax1 = fig3.add_subplot(1, 1, 1)
ax1.plot(df_2022["Residual for CB"].resample("H").mean(), color="black")
ax1.axhline(y=0, linestyle='--', color="grey")
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.set_ylabel('Power [MW]')
ax1.set_ylim(-2, 2)
ax1.set_title("Example of demand curve for a CB (hourly)", fontsize=16, pad=20)
plt.show()

summer_week = df_2022.loc["2022-07-01":"2022-07-07"]

fig4 = plt.figure(figsize=(14, 7))
ax1 = fig4.add_subplot(1, 2, 1)
ax1.plot(summer_week["Residual for CB"].resample("H").mean(), color="black")
ax1.axhline(y=0, linestyle='--', color="grey")
ax1.set_xlabel('Time')
plt.tight_layout(pad=6.0)
plt.xticks(rotation=90)
ax1.grid(True)
ax1.set_ylabel('Power [MW]')
ax1.set_ylim(-2, 2)
ax1.set_title("Demand curve for CB (hourly, summer week)", fontsize=16, pad=20)

winter_week = df_2022.loc["2022-01-01":"2022-01-07"]

ax2 = fig4.add_subplot(1, 2, 2)
ax2.plot(winter_week["Residual for CB"].resample("H").mean(), color="black")
ax2.axhline(y=0, linestyle='--', color="grey")
ax2.set_xlabel('Time')
plt.xticks(rotation=90)
ax2.grid(True)
ax2.set_ylabel('Power [MW]')
ax2.set_ylim(-2, 2)
ax2.set_title("Demand curve for CB (hourly, winter week)", fontsize=16, pad=20)
plt.show()


# Dispatch in the summer
begin_period = "2022-07-01"
end_period = "2022-07-21"
df_period = df_2022[begin_period:end_period].resample("H").mean()

V_t = V  # at the beginning fully charged
df_period = time_analysis_TES(df_period, hp, orc, V_t, V_in_t, V_out_t)
df_period["SOC"] = V - (df_period["Discharging volume"] - df_period["Charging volume"]).cumsum()

fig5 = plt.figure(figsize=(14, 7))
ax1 = fig5.add_subplot(1, 2, 1)
ax1.fill_between(df_period.index, -df_period["Charging volume"], 0, color="blue")
ax1.fill_between(df_period.index, 0, df_period["Discharging volume"], color="red")
ax1.axhline(y=0, linestyle='--', color="grey")
ax1.set_xlabel('Time')
plt.tight_layout(pad=6.0)
plt.xticks(rotation=90)
plt.legend(["Charging", "Discharging"], loc='lower right')
ax1.grid(True)
ax1.set_ylabel('Volume flow [m3/h]')
ax1.set_ylim(-60, 60)
ax1.set_title("Charging and discharging of the CB", fontsize=16, pad=20)

ax2 = fig5.add_subplot(1, 2, 2)
ax2.plot(df_2022.loc[begin_period:end_period, "Residual for CB"].resample("H").mean(), color="black")
ax2.axhline(y=0, linestyle='--', color="grey")
ax2.set_xlabel('Time')
plt.tight_layout(pad=6.0)
plt.xticks(rotation=90)
ax2.grid(True)
ax2.set_ylabel('Power [MW]')
ax2.set_ylim(-2, 2)
ax2.set_title("Demand curve (positive deficit, negative surplus)", fontsize=16, pad=20)
plt.show()

fig6 = plt.figure(figsize=(14, 7))
ax1 = fig6.add_subplot(1, 1, 1)
ax1.fill_between(df_period.index, df_period["SOC"], 0, color="green")
ax1.set_xlabel('Time')
plt.tight_layout(pad=6.0)
plt.xticks(rotation=90)
ax1.grid(True)
ax1.set_ylabel('SOC [m3]')
ax1.set_title("State of charge of the TES", fontsize=16, pad=20)
plt.show()


# Dispatch in the winter
begin_period = "2022-01-01"
end_period = "2022-01-21"
df_period = df_2022[begin_period:end_period].resample("H").mean()

V_t = V  # at the beginning fully charged
df_period = time_analysis_TES(df_period, hp, orc, V_t, V_in_t, V_out_t)
df_period["SOC"] = V - (df_period["Discharging volume"] - df_period["Charging volume"]).cumsum()

fig7 = plt.figure(figsize=(14, 7))
ax1 = fig7.add_subplot(1, 2, 1)
ax1.fill_between(df_period.index, -df_period["Charging volume"], 0, color="blue")
ax1.fill_between(df_period.index, 0, df_period["Discharging volume"], color="red")
ax1.axhline(y=0, linestyle='--', color="grey")
ax1.set_xlabel('Time')
plt.tight_layout(pad=6.0)
plt.xticks(rotation=90)
plt.legend(["Charging", "Discharging"], loc='lower right')
ax1.grid(True)
ax1.set_ylabel('Volume flow [m3/h]')
ax1.set_ylim(-60, 60)
ax1.set_title("Charging and discharging of the CB", fontsize=16, pad=20)

ax2 = fig7.add_subplot(1, 2, 2)
ax2.plot(df_2022.loc[begin_period:end_period, "Residual for CB"].resample("H").mean(), color="black")
ax2.axhline(y=0, linestyle='--', color="grey")
ax2.set_xlabel('Time')
plt.tight_layout(pad=6.0)
plt.xticks(rotation=90)
ax2.grid(True)
ax2.set_ylabel('Power [MW]')
ax2.set_ylim(-2, 2)
ax2.set_title("Demand curve (positive deficit, negative surplus)", fontsize=16, pad=20)
plt.show()

fig8 = plt.figure(figsize=(14, 7))
ax1 = fig8.add_subplot(1, 1, 1)
ax1.fill_between(df_period.index, df_period["SOC"], 0, color="green")
ax1.set_xlabel('Time')
plt.tight_layout(pad=6.0)
plt.xticks(rotation=90)
ax1.grid(True)
ax1.set_ylabel('SOC [m3]')
ax1.set_title("State of charge of the TES", fontsize=16, pad=20)
plt.show()

# TODO:
#  interesting to see for how many hours the CB operates in charging and discharging mode
#  thermal losses?
#  time series over the year? (one month takes less than one minute)
#  change temperature of the ambient?
#  analyze COP and round-trip efficiency
