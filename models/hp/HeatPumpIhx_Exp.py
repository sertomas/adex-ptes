import numpy as np
import pandas as pd
import os
from CoolProp.CoolProp import PropsSI as PSI
from tespy.components import (Compressor, CycleCloser, HeatExchanger, Sink, Source, Turbine)
from tespy.connections import Bus, Connection
from tespy.tools import ExergyAnalysis
from models.hp.HeatPump import HeatPump
import plotly.graph_objects as go
import plotly.io as pio


class HeatPumpIhx_Exp(HeatPump):

    # Initialize components of heat pump with ideal expander instead of valve.
    def generate_components(self):
        """
        Initialize components of the heat pump, including condenser, evaporator, valve, compressor,
        internal heat exchanger, cycle closer, heat sink, and heat source.
        """

        # Main Cycle
        self.comps['cond'] = HeatExchanger('Condenser')
        self.comps['eva'] = HeatExchanger('Evaporator')
        self.comps['exp'] = Turbine('Expander')
        self.comps['comp'] = Compressor('Compressor')
        self.comps['ihx'] = HeatExchanger("Internal Heat Exchanger")
        self.comps['cc'] = CycleCloser('CycleCloser')

        # Heat sink
        self.comps['si_in'] = Source('Sink in')
        self.comps['si_out'] = Sink('Sink out')

        # Heat source
        self.comps['sou_in'] = Source('Source in')
        self.comps['sou_out'] = Sink('Source out')

    # Initialize and add connections and busses to the network.
    def generate_connections(self):
        """
        Initialize and add connections and busses to the network for the heat pump components and streams.
        """

        # Main Cycle
        self.conns['c31'] = Connection(self.comps['cc'], 'out1', self.comps['cond'], 'in1', label="31")
        self.conns['c32'] = Connection(self.comps['cond'], 'out1', self.comps['ihx'], 'in1', label="32")
        self.conns['c33'] = Connection(self.comps['ihx'], 'out1', self.comps['exp'], 'in1', label="33")
        self.conns['c34'] = Connection(self.comps['exp'], 'out1', self.comps['eva'], 'in2', label="34")
        self.conns['c35'] = Connection(self.comps['eva'], 'out2', self.comps['ihx'], 'in2', label="35")
        self.conns['c36'] = Connection(self.comps['ihx'], 'out2', self.comps['comp'], 'in1', label="36")
        self.conns['c30'] = Connection(self.comps['comp'], 'out1', self.comps['cc'], 'in1', label="30")

        # Heat Sink
        self.conns['c21'] = Connection(self.comps['si_in'], 'out1', self.comps['cond'], 'in2', label="21")
        self.conns['c22'] = Connection(self.comps['cond'], 'out2', self.comps['si_out'], 'in1', label="22")

        # Heat Source
        self.conns['c11'] = Connection(self.comps['sou_in'], 'out1', self.comps['eva'], 'in1', label="11")
        self.conns['c12'] = Connection(self.comps['eva'], 'out1', self.comps['sou_out'], 'in1', label="12")

        self.network.add_conns(*[conn for conn in self.conns.values()])

    # Perform initial parametrization with starting values.
    def init_simulation(self):
        """
        Perform initial parametrization with starting values for the heat pump components and streams.
        """

        # Pressure drops
        self.comps['cond'].set_attr(pr1=self.params['cond']['pr1'],
                                    pr2=self.params['cond']['pr2'])
        self.comps['eva'].set_attr(pr1=self.params['eva']['pr1'],
                                   pr2=self.params['eva']['pr2'])
        self.comps['ihx'].set_attr(pr1=self.params['ihx']['pr1'],
                                   pr2=self.params['ihx']['pr1'])

        # For the dimensioning there are many options:
        # self.conns['c31'].set_attr(m=self.params['dimension_parameters']['m_31'])
        # self.comps['cond'].set_attr(Q=self.params['dimension_parameters']['Q_cond'])
        # self.comps['comp'].set_attr(P=self.params['dimension_parameters']['P_comp'])
        self.conns['c21'].set_attr(m=self.params['dimension_parameters']['m_21'])

        # Technical restrictions
        self.comps['comp'].set_attr(eta_s=self.params['comp']['eta_s'])
        self.comps['exp'].set_attr(eta_s=1)  # ideal expander

        # Main cycle
        h_c36_start = PSI('H',
                          'P', self.params['c35']['p_start'] * 1e5,
                          'T', self.params['c36']['T_start'] + 273.15,
                          self.working_fluid
                          ) * 1e-3
        self.conns['c36'].set_attr(h=h_c36_start,
                                   p=self.params['c35']['p_start'],
                                   fluid=self.working_fluid_vec)
        h_c32_start = PSI('H',
                          'P', self.params['c32']['p'] * 1e5,
                          'T', self.params['c32']['T_start'] + 273.15,
                          self.working_fluid
                          ) * 1e-3
        self.conns['c32'].set_attr(h=h_c32_start,
                                   p=self.params['c32']['p'])

        h_c35_start = PSI('H',
                          'P', self.params['c35']['p_start'] * 1e5,
                          'T', self.params['c35']['T_start'] + 273.15,
                          self.working_fluid
                          ) * 1e-3
        self.conns['c35'].set_attr(h=h_c35_start)

        # Heat Sink
        self.conns['c21'].set_attr(T=self.config['TES']['T_low'],
                                   p=self.config['TES']['p'],
                                   fluid=self.water_vec)
        self.conns['c22'].set_attr(T=self.config['TES']['T_high'])

        # Heat Source
        self.conns['c11'].set_attr(T=self.params['ambient']['T_in'],
                                   p=self.params['ambient']['p'],
                                   fluid=self.water_vec)
        self.conns['c12'].set_attr(T=self.params['ambient']['T_out'])

        # Busses
        self.busses['power input'] = Bus('power input')
        self.busses['power input'].add_comps({'comp': self.comps['comp'], 'base': 'bus'})

        self.busses['heat input'] = Bus('heat input')
        self.busses['heat input'].add_comps({'comp': self.comps['sou_in'], 'base': 'bus'},
                                            {'comp': self.comps['sou_out'], 'base': 'component'})

        self.busses['heat output'] = Bus('heat output')
        self.busses['heat output'].add_comps({'comp': self.comps['si_in'], 'base': 'bus'},
                                             {'comp': self.comps['si_out'], 'base': 'component'})

        self.network.add_busses(self.busses['power input'],
                                self.busses['heat input'],
                                self.busses['heat output'])

        # Solve model
        self.solve_model()

    # Perform final parametrization and design simulation.
    def design_simulation(self):
        """
        Perform final parametrization and design simulation for the heat pump, including dimensioning and solving.
        """

        self.conns['c36'].set_attr(h=None, p=None)
        self.conns['c32'].set_attr(h=None)

        # self.comps['eva'].set_attr(ttd_l=self.params['eva']['ttd_l'])
        # self.comps['ihx'].set_attr(ttd_u=self.params['ihx']['ttd_u'])
        self.comps['cond'].set_attr(ttd_l=self.params['cond']['ttd_l'])
        # self.conns['c35'].set_attr(h=None, Td_bp=self.params['c35']['Td_bp'])
        self.conns['c35'].set_attr(h=None, Td_bp=self.params['c35']['Td_bp'])
        # self.conns['c32'].set_attr(T=75)
        self.comps['ihx'].set_attr(ttd_u=self.params['ihx']['ttd_u'])
        self.comps['eva'].set_attr(ttd_l=self.params['eva']['ttd_l'])

        # Solve model
        self.solve_model()

        # Calculate COP
        self.cop = abs(self.comps['cond'].Q.val) / self.busses['power input'].P.val

        # Print relevant information
        print('HP base case: \n'
              'COP = ' + str(round(self.cop, 3)) + '.')
        print('eta_s = ' + str(round(self.comps['comp'].eta_s.val, 3)) + '.')
        print('m_31 = ' + str(round(self.network.results['Connection'].loc['31', 'm'], 3)) + ' kg/s.')

    # Perform exergy analysis.
    def perform_exergy_analysis(self, path=None):
        """
        Perform exergy analysis for the heat pump and optionally save the analysis results to a file.

        Args:
            path (str, optional): Path to save the analysis results. Default is None.

        Returns:
            None
        """

        self.ean = ExergyAnalysis(self.network,
                                  E_F=[self.busses['power input'], self.busses['heat input']],
                                  E_P=[self.busses['heat output']])
        self.ean.analyse(pamb=self.config['ambient']['p'], Tamb=self.config['ambient']['T'])

        if path is not None:
            (self.ean.connection_data / 1e3).round(5).to_csv(path + "connections.csv")  # in kJ/kg
            self.ean.component_data[self.ean.component_data["E_F"] > 0].round(5).to_csv(
                path + "components.csv")

        self.epsilon = self.ean.network_data['epsilon']

        links, nodes = self.ean.generate_plotly_sankey_input()

        fig_hp = go.Figure(go.Sankey(
            arrangement='snap',
            node={
                'label': nodes,
                'pad': 11,
                'color': 'orange'
            },
            link=links
        ))
        pio.write_image(fig_hp, 'sankey_hp.png')

    # Perform a sensitivity analysis of the high pressure of the heat pump.
    def perform_sens_anal_p_high(self, p_min, p_max, stepwidth=1, plot=False):
        """
        Perform a sensitivity analysis of the high pressure of the heat pump over a specified range of values.

        Args:
            p_min (float): Minimum value of the high pressure.
            p_max (float): Maximum value of the high pressure.
            stepwidth (float, optional): Step size for incrementing the high pressure. Default is 1.
            plot (bool, optional): Whether to generate plots during the analysis. Default is False.

        Returns:
            None
        """

        T_TES_high = self.config['TES']['T_high']
        p_high_range = [*np.arange(p_min, p_max, stepwidth)]
        table = pd.DataFrame(index=p_high_range,
                             columns=['COP', 'epsilon', 'min_temp_diff_eva [K]', 'min_temp_diff_cond [K]',
                                      'm_31 [kg/s]', 'm_21 [kg/s]', 'Q_cond [MW]'])
        table = table.rename_axis('p32')
        path = 'outputs/sens_analysis_hp/high_pressure/TES_' + str(T_TES_high) + 'C_ihx/'
        os.makedirs(path, exist_ok=True)
        for p in np.arange(p_min, p_max, stepwidth):
            self.params['c32']['p'] = p
            self.conns['c32'].set_attr(p=self.params['c32']['p'])
            self.network.solve('design')
            self.network.results['Connection'].round(5).to_csv(path + 'results_' + str(p).replace('.', '_') + 'bar.csv')
            self.perform_exergy_analysis()  # for epsilon
            [min_temp_diff_eva, max_temp_diff_eva] = self.qt_diagram(11, 12, 35, 34,
                                                                     self.config['tech_specs']['delta_T_min'],
                                                                     False, path + 'eva_' + str(p).replace('.',
                                                                                                           '_') + 'bar.png')
            [min_temp_diff_cond, max_temp_diff_cond] = self.qt_diagram(22, 21, 31, 32,
                                                                       self.config['tech_specs']['delta_T_min'],
                                                                       False, path + 'cond_' + str(p).replace('.',
                                                                                                              '_') + 'bar.png')
            table.loc[p, 'COP'] = (round(abs(self.comps['cond'].Q.val) / self.busses['power input'].P.val, 3))
            table.loc[p, 'epsilon'] = round(abs(self.epsilon), 3)
            table.loc[p, 'min_temp_diff_eva [K]'] = (round(min_temp_diff_eva, 3))
            table.loc[p, 'min_temp_diff_cond [K]'] = (round(min_temp_diff_cond, 3))
            table.loc[p, 'm_31 [kg/s]'] = (round(self.network.results['Connection'].loc['31', 'm'], 3))
            table.loc[p, 'm_21 [kg/s]'] = (round(self.network.results['Connection'].loc['21', 'm'], 3))
            table.loc[p, 'Q_cond [MW]'] = round(abs(self.comps['cond'].Q.val * 1e-6), 3)
            if plot is True:
                self.plot_logph({}, path=path + str(p).replace('.', '_') + 'bar_logph.png')
        table.to_csv(path + str(T_TES_high) + 'C_results_table_ihx.csv')

    # Get states to generate log(p)-h-diagram of the heat pump process.
    def get_states(self):
        """
        Get states to generate log(p)-h-diagram of the heat pump process.

        Returns:
            None
        """

        # Get component plotting data
        result_dict = dict()
        ihx_label = self.comps['ihx'].label
        result_dict.update({self.comps['eva'].label: self.comps['eva'].get_plotting_data()[2]})
        result_dict.update({f'{ihx_label} (cold)': self.comps['ihx'].get_plotting_data()[2]})
        result_dict.update({self.comps['comp'].label: self.comps['comp'].get_plotting_data()[1]})
        result_dict.update({self.comps['cond'].label: self.comps['cond'].get_plotting_data()[1]})
        result_dict.update({f'{ihx_label} (hot)': self.comps['ihx'].get_plotting_data()[1]})
        result_dict.update({self.comps['val'].label: self.comps['val'].get_plotting_data()[1]})
        return result_dict
