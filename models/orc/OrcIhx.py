import pandas as pd
from CoolProp.CoolProp import PropsSI as PSI
from tespy.components import (Pump, CycleCloser, HeatExchanger, Sink, Source, Turbine)
from tespy.connections import Bus, Connection
import plotly.graph_objects as go
from tespy.tools import ExergyAnalysis
from models.orc.Orc import Orc
import plotly.io as pio


class OrcIhx(Orc):

    def generate_components(self):
        """Initialize components of heat pump."""

        # Main Cycle
        self.comps['cond'] = HeatExchanger('Condenser')
        self.comps['eva'] = HeatExchanger('Evaporator')
        self.comps['exp'] = Turbine('Expander')
        self.comps['pump'] = Pump('Pump')
        self.comps['ihx'] = HeatExchanger("Internal Heat Exchanger")
        self.comps['cc'] = CycleCloser('CycleCloser')

        # Heat sink
        self.comps['si_in'] = Source('Sink in')
        self.comps['si_out'] = Sink('Sink out')

        # Heat source
        self.comps['sou_in'] = Source('Source in')
        self.comps['sou_out'] = Sink('Source out')

    def generate_connections(self):
        """Initialize and add connections and busses to network."""

        # Main Cycle
        self.conns['c61'] = Connection(self.comps['cc'], 'out1', self.comps['pump'], 'in1', label="61")
        self.conns['c62'] = Connection(self.comps['pump'], 'out1', self.comps['ihx'], 'in2', label="62")
        self.conns['c63'] = Connection(self.comps['ihx'], 'out2', self.comps['eva'], 'in2', label="63")
        self.conns['c64'] = Connection(self.comps['eva'], 'out2', self.comps['exp'], 'in1', label="64")
        self.conns['c65'] = Connection(self.comps['exp'], 'out1', self.comps['ihx'], 'in1', label="65")
        self.conns['c66'] = Connection(self.comps['ihx'], 'out1', self.comps['cond'], 'in1', label="66")
        self.conns['c60'] = Connection(self.comps['cond'], 'out1', self.comps['cc'], 'in1', label="60")

        # Heat Sink
        self.conns['c41'] = Connection(self.comps['si_in'], 'out1', self.comps['cond'], 'in2', label="41")
        self.conns['c42'] = Connection(self.comps['cond'], 'out2', self.comps['si_out'], 'in1', label="42")

        # Heat Source
        self.conns['c51'] = Connection(self.comps['sou_in'], 'out1', self.comps['eva'], 'in1', label="51")
        self.conns['c52'] = Connection(self.comps['eva'], 'out1', self.comps['sou_out'], 'in1', label="52")

        self.network.add_conns(*[conn for conn in self.conns.values()])

    def init_simulation(self):
        """
        Perform initial parametrization with starting values.
        """

        # Pressure drops
        self.comps['cond'].set_attr(pr1=self.params['cond']['pr1'],
                                    pr2=self.params['cond']['pr2'])
        self.comps['eva'].set_attr(pr1=self.params['eva']['pr1'],
                                   pr2=self.params['eva']['pr2'])
        self.comps['ihx'].set_attr(pr1=self.params['ihx']['pr1'],
                                   pr2=self.params['ihx']['pr2'])

        # For the dimensioning there are many options:
        # self.conns['c61'].set_attr(m=self.params['dimension_parameters']['m_61'])
        # self.comps['eva'].set_attr(Q=self.params['dimension_parameters']['Q_eva'])
        # self.comps['exp'].set_attr(P=self.params['dimension_parameters']['P_exp'])
        self.conns['c51'].set_attr(m=self.params['dimension_parameters']['m_51'])

        # Technical restrictions
        self.comps['exp'].set_attr(eta_s=self.params['exp']['eta_s'])
        self.comps['pump'].set_attr(eta_s=self.params['pump']['eta_s'])

        # Main cycle
        h_c61_start = PSI('H',
                          'P', self.params['c61']['p_start'] * 1e5,
                          'T', self.params['c61']['T_start'] + 273.15,
                          self.working_fluid
                          ) * 1e-3
        h_c64_start = PSI('H',
                          'P', self.params['c64']['p_start'] * 1e5,
                          'T', self.params['c64']['T_start'] + 273.15,
                          self.working_fluid
                          ) * 1e-3
        h_c66_start = PSI('H',
                          'P', self.params['c66']['p_start'] * 1e5,
                          'T', self.params['c66']['T_start'] + 273.15,
                          self.working_fluid
                          ) * 1e-3

        self.conns['c61'].set_attr(p=self.params['c61']['p_start'],
                                   h=h_c61_start,
                                   fluid=self.working_fluid_vec)
        self.conns['c62'].set_attr(p=self.params['c62']['p'])
        self.conns['c64'].set_attr(h=h_c64_start)
        self.conns['c66'].set_attr(h=h_c66_start)

        # Heat Sink
        self.conns['c41'].set_attr(T=self.params['ambient']['T_in'],
                                   p=self.params['ambient']['p'],
                                   fluid=self.fluid_ambient_vec)
        self.conns['c42'].set_attr(T=self.params['ambient']['T_out'])

        # Heat Source
        self.conns['c51'].set_attr(T=self.config['TES']['T_high'],
                                   p=self.config['TES']['p'],
                                   fluid=self.fluid_tes_vec)
        self.conns['c52'].set_attr(T=self.config['TES']['T_low'])

        # Busses
        self.busses['power output'] = Bus('power output')
        self.busses['power output'].add_comps({'comp': self.comps['exp'], 'base': 'component'},
                                              {'comp': self.comps['pump'], 'base': 'bus'})

        self.busses['heat input'] = Bus('heat input')
        self.busses['heat input'].add_comps({'comp': self.comps['sou_in'], 'base': 'bus'},
                                            {'comp': self.comps['sou_out'], 'base': 'component'})

        self.busses['heat output'] = Bus('heat output')
        self.busses['heat output'].add_comps({'comp': self.comps['si_in'], 'base': 'bus'},
                                             {'comp': self.comps['si_out'], 'base': 'component'})

        self.network.add_busses(self.busses['power output'],
                                self.busses['heat input'],
                                self.busses['heat output'])

        # Solve model
        self.solve_model()

    def design_simulation(self):
        """
        Perform final parametrization and design simulation.
        """

        self.conns['c61'].set_attr(h=None, p=None, Td_bp=self.params['c61']['Td_bp'])
        self.conns['c62'].set_attr(h=None)
        self.conns['c64'].set_attr(h=None)
        self.conns['c66'].set_attr(h=None)
        self.comps['cond'].set_attr(ttd_l=self.params['cond']['ttd_l'])
        self.comps['eva'].set_attr(ttd_u=self.params['eva']['ttd_u'])
        self.comps['ihx'].set_attr(ttd_l=self.params['ihx']['ttd_l'])

        # Solve model
        self.solve_model()

        # Calculate COP
        self.eta = abs(self.busses['power output'].P.val / self.comps['eva'].Q.val)

        # Print relevant information
        print('ORC base case: \n'
              'eta_tot = ' + str(round(self.eta, 4)) + '.')

    def perform_exergy_analysis(self, path=None):
        """
        Perform exergy analysis.

        Args:
            save_results (bool, optional): Whether to save the analysis results. Defaults to True.

        Returns:
            None
        """

        self.ean = ExergyAnalysis(self.network,
                                  E_F=[self.busses['heat input']],
                                  E_P=[self.busses['power output']],
                                  E_L=[self.busses['heat output']])
        self.ean.analyse(pamb=self.config['ambient']['p'], Tamb=self.config['ambient']['T'])

        if path is not None:
            (self.ean.connection_data / 1e3).round(5).to_csv(path + "connections.csv")  # in kJ/kg
            self.ean.component_data[self.ean.component_data["E_F"] > 0].round(5).to_csv(
                path + "components.csv")

        self.epsilon = self.ean.network_data['epsilon']

        links, nodes = self.ean.generate_plotly_sankey_input()

        fig_orc = go.Figure(go.Sankey(
            arrangement='snap',
            node={
                'label': nodes,
                'pad': 11,
                'color': 'orange'
            },
            link=links
        ))
        pio.write_image(fig_orc, 'sankey_hp.png')

    def perform_sens_anal_p_high(self, p_min, p_max, stepwidth=1, plot=False):
        """
        Perform a sensitivity analysis of the high pressure of the heat pump.

        Args:
            p_min (float): Minimum value of the high pressure.
            p_max (float): Maximum value of the high pressure.
            stepwidth (float, optional): Step size for incrementing the high pressure. Defaults to 1.
            plot (bool, optional): Whether to generate plots during the analysis. Defaults to False.

        Returns:
            None
        """

    # Get states to generate log(p)-h-diagram of the ORC system.
    def get_states(self):
        """
        Get states to generate log(p)-h-diagram of the ORC system.

        Returns:
            None
        """

        # Get component plotting data
        result_dict = dict()
        ihx_label = self.comps['ihx'].label
        result_dict.update({self.comps['eva'].label: self.comps['eva'].get_plotting_data()[2]})
        result_dict.update({f'{ihx_label} (cold)': self.comps['ihx'].get_plotting_data()[2]})
        result_dict.update({self.comps['pump'].label: self.comps['pump'].get_plotting_data()[1]})
        result_dict.update({self.comps['cond'].label: self.comps['cond'].get_plotting_data()[1]})
        result_dict.update({f'{ihx_label} (hot)': self.comps['ihx'].get_plotting_data()[1]})
        result_dict.update({self.comps['exp'].label: self.comps['exp'].get_plotting_data()[1]})
        return result_dict
