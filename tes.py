import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from cb_set_pars import hp_settings, orc_settings
from cb_network import hp_network, orc_network

P_in = 1.5e6  # [e6 = MW]

hp = hp_network(['R245fa', 'water'])
orc = orc_network(['R245fa', 'water'])
hp_settings(hp, P_in)
orc_settings(orc)
hp.solve(mode='design')
orc.solve(mode='design')

res = pd.read_csv("residual_load_entose_DE_2022.csv", parse_dates=[0], index_col=0)  # [MW]
res = res / 1e3  # [GW]

fig1 = plt.figure(figsize=(14, 7))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(res.resample("H").mean(), color='black')
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.set_ylabel('Residual load [GW]')
ax1.set_ylim(0, 75)
ax1.set_title("Residual load in Germany (2022)")
plt.show()

P_in_t = res / res.max() * P_in / 1e6  # [MW]

fig2 = plt.figure(figsize=(14, 7))
ax1 = fig2.add_subplot(1, 1, 1)
ax1.plot(P_in_t.resample("H").mean(), color='red')
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.set_ylabel('Power [MW]')
ax1.set_ylim(0, 2)
ax1.set_title("Power inlet of the heat pump")
plt.show()

fig3 = plt.figure(figsize=(14, 7))
ax1 = fig3.add_subplot(1, 1, 1)
ax1.plot(P_in_t.loc["2022-07-01":"2022-07-05"].resample("H").mean(), color='red')
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.set_ylabel('Power [MW]')
ax1.set_ylim(0, 2)
ax1.set_title("Power inlet of the heat pump")
plt.show()

fig4 = plt.figure(figsize=(14, 7))
ax1 = fig4.add_subplot(1, 1, 1)
ax1.plot(P_in_t.loc["2022-02-01":"2022-02-05"].resample("H").mean(), color='red')
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.set_ylabel('Power [MW]')
ax1.set_ylim(0, 2)
ax1.set_title("Power inlet of the heat pump")
plt.show()

# TODO: Idea is to run the code for a give time range (day, week, month, year?) and to analyze the charging and discharging behaviour

df_2022 = pd.read_csv("wind_pv_entose_DE_2022.csv", parse_dates=[0], index_col=0)  # [MW]
df_2022 = df_2022 / 1e3  # [GW]
df_2022["RES"] = df_2022["Wind Offshore"] + df_2022["Wind Onshore"] + df_2022["Solar"]
df_2022["Residual"] = df_2022["Last"] - df_2022["RES"]
df_2022["RES new"] = df_2022["RES"] * 2  # assumption: wind and pv capacity doubled
df_2022["Residual new"] = df_2022["Last"] - df_2022["RES new"]
df_2022["Power inlet"] = df_2022["Residual new"] / df_2022["Residual new"].max() * P_in / 1e6  # [MW]

fig5 = plt.figure(figsize=(14, 7))
ax1 = fig5.add_subplot(1, 1, 1)
ax1.plot(df_2022["RES"].resample("D").mean(), color="green")
ax1.plot(df_2022["Residual"].resample("D").mean(), color="black")
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.set_ylabel('Power [GW]')
ax1.legend(["RES", "Load"])
ax1.set_ylim(0, 100)
ax1.set_title("Residual load and power generation from RES in Germany (daily mean, 2022)")
plt.show()

fig6 = plt.figure(figsize=(14, 7))
ax1 = fig6.add_subplot(1, 1, 1)
ax1.plot(df_2022["RES new"].resample("D").mean(), color="green")
ax1.plot(df_2022["Residual"].resample("D").mean(), color="black")
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.set_ylabel('Power [GW]')
ax1.legend(["RES", "Load"])
ax1.set_ylim(0, 100)
ax1.set_title("Residual load and power generation from RES in Germany (daily mean, future scenario)")
plt.show()

fig7 = plt.figure(figsize=(14, 7))
ax1 = fig7.add_subplot(1, 1, 1)
ax1.plot(df_2022["Power inlet"].resample("D").mean(), color="black")
ax1.set_xlabel('Time')
ax1.grid(True)
ax1.set_ylabel('Power [GW]')
ax1.set_ylim(-2, 2)
ax1.set_title("Power inlet of the heat pump")
plt.show()
