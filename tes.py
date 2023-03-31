import pandas as pd
import matplotlib.pyplot as plt
from cb_set_pars import hp_settings, orc_settings
from cb_network import hp_network, orc_network
from tes_func import time_analysis_TES
from config import V, P_out, P_in, P_min, rho
from tes_plot import tes_plot_init, tes_plot_period


hp = hp_network(['R245fa', 'water'])
orc = orc_network(['R245fa', 'water'])
hp_settings(hp, P_in)
orc_settings(orc, P_out)
hp.solve(mode='design')
hp_s = hp.results['Connection']
orc.solve(mode='design')
orc_s = orc.results['Connection']

# read design dimensions from simulation, if you want to change the size of the CB, change vars in "config.py" and rerun
V_in = hp_s.loc["7", "m"] / rho * 3600  # [m^3/h] charging volume stream
V_out = orc_s.loc["17", "m"] / rho * 3600  # [m^3/h] discharging volume stream
t_charge = V / V_in  # [h] charging time
t_discharge = V / V_out  # [h] discharging time

# read data for power generation, load demand and temperature
df_2022 = pd.read_csv("wind_pv_entose_DE_2022.csv", index_col=0, parse_dates=[0])  # [MW]
df_2022 = df_2022 / 1e3  # [GW]
df_2022["RES"] = df_2022["Wind Offshore"] + df_2022["Wind Onshore"] + df_2022["Solar"]
df_2022["Residual"] = df_2022["Last"] - df_2022["RES"]
df_2022["RES new"] = df_2022["RES"] / df_2022["RES"].sum() * df_2022["Last"].sum()  # assumption: wind and pv capacity cover the whole demand
df_2022["Residual new"] = df_2022["Last"] - df_2022["RES new"]
df_2022["Residual for CB"] = df_2022["Residual new"] / df_2022["Residual new"].max() * P_in / 1e6  # [MW]

dwd_2022 = pd.read_csv("stundenwerte_TU_03987_akt/produkt_tu_stunde_20210926_20230329_03987.txt", delimiter=';')
dwd_2022['MESS_DATUM'] = pd.to_datetime(dwd_2022['MESS_DATUM'], format='%Y%m%d%H')  # convert MESS_DATUM to datetime format
dwd_2022.set_index('MESS_DATUM', inplace=True)  # set MESS_DATUM as the time index
temp_2022 = dwd_2022.loc["2022-01-01":"2022-12-31", "TT_TU"]

tes_plot_init(df_2022, temp_2022)

# Dispatch in the summer
begin_period = "2022-07-01"
end_period = "2022-08-31"
df_period = df_2022[begin_period:end_period].resample("H").mean()

V_t = V / 2  # at the beginning half charged
df_period = time_analysis_TES(df_period, hp, orc, V_t, V_in, V_out)
df_period["SOC"] = V_t - (df_period["Discharging volume"] - df_period["Charging volume"]).cumsum()

tes_plot_period(df_2022, df_period, begin_period, end_period)

# Dispatch in the winter
begin_period = "2022-01-01"
end_period = "2022-02-28"
df_period = df_2022[begin_period:end_period].resample("H").mean()

V_t = V / 2  # at the beginning half charged
df_period = time_analysis_TES(df_period, hp, orc, V_t, V_in, V_out)
df_period["SOC"] = V_t - (df_period["Discharging volume"] - df_period["Charging volume"]).cumsum()

tes_plot_period(df_2022, df_period, begin_period, end_period)

# TODO:
#  1) adapt/optimize the ORC and HP low pressure to the ambient temperature over time period
#     ==> time/temperature sensitive COP
#     one month around 10 seconds run time with hourly time step
#  2) calculate how many hours CB operates in charging and discharging mode
#  3) analyze COP and round-trip efficiency
#  4) consider thermal losses?
