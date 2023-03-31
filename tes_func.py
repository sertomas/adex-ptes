from cb_set_pars import hp_settings, orc_settings
from config import V, P_out, P_in, P_min, rho


def time_analysis_TES(df, hp, orc, V_t, V_in, V_out):

    for t in df.index:
        P = df.loc[t, "Residual for CB"] * 1e6

        # check if SOC is high enough for discharging
        if V_t > V_out:
            if P > P_min:
                if P < P_out:
                    P_orc = P
                else:
                    P_orc = P_out
                orc_settings(orc, P_orc)
                orc.solve(mode='design')
                df.loc[t, "Discharging volume"] = orc.results['Connection'].loc["17", "m"] / rho * 3600
                V_t -= df.loc[t, "Discharging volume"]

        # check if SOC is low enough for charging
        if V_t < V - V_in:
            if P < - P_min:
                if P < P_in:
                    P_hp = P
                else:
                    P_hp = P_in
                hp_settings(hp, -P_hp)
                hp.solve(mode='design')
                df.loc[t, "Charging volume"] = hp.results['Connection'].loc["7", "m"] / rho * 3600
                V_t += df.loc[t, "Charging volume"]

    df = df.fillna(0)

    return df
