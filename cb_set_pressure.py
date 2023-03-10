from cb_qt_diagram import qt_sens_latent
import pandas as pd


def set_pressure(hp, orc, data, delta_t_min):

    sens_analysis_p_high_orc = pd.DataFrame()

    for p in data['p_high_orc']:
        orc.get_conn('14').set_attr(p=p)
        orc.solve('design')
        delta_t_he = qt_sens_latent(orc.get_conn('17'), orc.get_conn('18'), orc.get_conn('14'), orc.get_conn('13'), delta_t_min, False)
        sens_analysis_p_high_orc.loc[f"p13 = {p} bar", "Min. temp. diff. Evaporator ORC [K]"] = delta_t_he
        sens_analysis_p_high_orc.loc[f"p13 = {p} bar", "Efficiency ORC [%]"] = abs(orc.get_comp('steam turbine').P.val - orc.get_comp('pump').P.val) / abs(orc.get_comp('evaporator orc').Q.val) * 100

    orc.get_conn('14').set_attr(p=6.5)  # value from cb_set_pars

    sens_analysis_p_high_hp = pd.DataFrame()

    for p in data['p_high_hp']:
        hp.get_conn('4').set_attr(p=p)
        hp.solve('design')
        delta_t_he = qt_sens_latent(hp.get_conn('8'), hp.get_conn('7'), hp.get_conn('3'), hp.get_conn('4'), delta_t_min, False)
        sens_analysis_p_high_hp.loc[f"p4 = {p} bar", "Min. temp. diff. Condenser HP [K]"] = delta_t_he
        sens_analysis_p_high_hp.loc[f"p4 = {p} bar", "COP HP [-]"] = abs(hp.get_comp('condenser hp').Q.val) / hp.get_comp('compressor').P.val

    hp.get_conn('4').set_attr(p=26)  # value from cb_set_pars

    sens_analysis_p_low_orc = pd.DataFrame()

    for p in data['p_low_orc']:
        orc.get_conn('12').set_attr(p=p)
        orc.solve('design')
        delta_t_he = qt_sens_latent(orc.get_conn('16'), orc.get_conn('15'), orc.get_conn('11'), orc.get_conn('12'), delta_t_min, False)
        sens_analysis_p_low_orc.loc[f"p12 = {round(p,1)} bar", "Min. temp. diff. Condenser ORC [K]"] = delta_t_he
        sens_analysis_p_low_orc.loc[f"p12 = {round(p,1)} bar", "Efficiency ORC [%]"] = abs(orc.get_comp('steam turbine').P.val - orc.get_comp('pump').P.val) / abs(orc.get_comp('evaporator orc').Q.val) * 100

    round(sens_analysis_p_high_hp, 3).to_csv("hp_p_high.csv")
    round(sens_analysis_p_high_orc, 3).to_csv("orc_p_high.csv")
    round(sens_analysis_p_low_orc, 3).to_csv("orc_p_low.csv")
