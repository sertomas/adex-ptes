from tespy.networks import Network
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from fluprodia import FluidPropertyDiagram
from CoolProp.CoolProp import PropsSI as PSI
# adexlog = logging.getLogger("adex-cb")


class HeatPump:
    """
    A class representing a heat pump.

    Args:
        config (dict): Configuration parameters for the heat pump.
        params (dict): Parameters defining the ORC components and properties.

    Attributes:
        working_fluid (str): The working fluid used in the heat pump.
        fluid_tes (str): The secondary fluid (typically water) used in the heat pump.
        working_fluid_vec (dict): A dictionary specifying the composition of the working fluid.
        fluid_tes_vec (dict): A dictionary specifying the composition of the secondary fluid.
        comps (dict): A dictionary to store components of the heat pump.
        conns (dict): A dictionary to store connections in the heat pump.
        busses (dict): A dictionary to store busses in the heat pump.
        network (Network): An instance of the Network class for modeling the heat pump.
        eta (float): The efficiency of the heat pump (initialized to NaN).
        epsilon (float): The exergy efficiency of the heat pump (initialized to NaN).

    Methods:
        generate_components: Initialize components of the heat pump.
        generate_connections: Initialize and add connections and busses to the network.
        init_simulation: Perform initial parametrization with starting values.
        design_simulation: Perform final parametrization and design simulation.
        run_model: Run the initialization and design simulation routine.
        solve_model: Solve the model in design mode.
        perform_sens_anal_p_high: Perform a sensitivity analysis of the high pressure of the heat pump.
        get_states: Get states to generate log(p)-h-diagram of the heat pump process.
        perform_exergy_analysis: Perform exergy analysis.
        plot_logph: Generate log(p)-h-diagram of the heat pump.
        qt_diagram: Generate a QT diagram of the heat pump.

    """
    
    # Constructor to initialize the HeatPump object with configuration and parameters.
    def __init__(self, config, params):
        """
        Initialize a HeatPump object with the given configuration and parameters.

        Args:
            config (dict): Configuration settings for the heat pump.
            params (dict): Parameters for the heat pump components and fluids.
        """
        self.params = params
        self.config = config

        self.working_fluid = self.params['fluids']['working_fluid']
        self.fluid_tes = self.params['fluids']['fluid_TES']
        self.fluid_ambient = self.params['fluids']['fluid_ambient']

        self.working_fluid_vec = {self.working_fluid: 1}
        self.fluid_tes_vec = {self.fluid_tes: 1}
        self.fluid_ambient_vec = {self.fluid_ambient: 1}

        self.comps = dict()
        self.conns = dict()
        self.busses = dict()

        self.network = Network(
            fluids=[self.working_fluid, self.fluid_tes, self.fluid_ambient],
            T_unit='C', p_unit='bar', h_unit='kJ / kg', m_unit='kg / s'
        )

        self.cop = np.nan
        self.epsilon = np.nan

    # Initialize components of the heat pump.
    def generate_components(self):
        """
        Initialize components of the heat pump.
        """

    # Initialize and add connections and busses to the network.
    def generate_connections(self):
        """
        Initialize and add connections and busses to the network.
        """

    # Perform initial parametrization with starting values.
    def init_simulation(self):
        """
        Perform initial parametrization with starting values.
        """

    # Perform final parametrization and design simulation.
    def design_simulation(self):
        """
        Perform final parametrization and design simulation.
        """

    # Run the initialization and design simulation routine.
    def run_model(self):
        """
        Run the initialization and design simulation routine.
        """

        self.generate_components()
        self.generate_connections()
        self.init_simulation()
        self.design_simulation()

    # Solve the model in design mode.
    def solve_model(self):
        """
        Solve the model in design mode.
        """

        self.network.solve('design')

    # Perform a sensitivity analysis of the high pressure of the heat pump.
    def perform_sens_anal_p_high(self, p_min, p_max, stepwidth=1):
        """
        Perform a sensitivity analysis of the high pressure of the heat pump.

        Args:
            p_min (float): Minimum high pressure value.
            p_max (float): Maximum high pressure value.
            stepwidth (float): Step size for sensitivity analysis.

        Returns:
            None
        """

    # Get states to generate log(p)-h-diagram of the heat pump process.
    def get_states(self):
        """
        Get states to generate log(p)-h-diagram of the heat pump process.

        Returns:
            None
        """

    # Perform exergy analysis.
    def perform_exergy_analysis(self, save_results=True):
        """
        Perform exergy analysis.

        Args:
            save_results (bool): Whether to save the analysis results.

        Returns:
            None
        """

    # Generate and plot log(p)-h-diagram of the heat pump process.
    def plot_logph(self, path=None, return_diagram=False):
        """
        Generate and plot log(p)-h-diagram of heat pump process.

        Args:
            result_dict (dict): Dictionary containing result data.
            path (str): Path to save the diagram image.
            return_diagram (bool): Whether to return the diagram object.

        Returns:
            FluidPropertyDiagram or None: The diagram object if return_diagram is True, else None.
        """

        result_dict = self.get_states()

        # Initialize fluid property diagram
        diagram = FluidPropertyDiagram(self.params['setup']['refrig'])
        diagram.set_unit_system(T='°C', p='bar', h='kJ/kg')
        diagram.set_limits(x_min=self.params['logph']['x_min'],
                           x_max=self.params['logph']['x_max'],
                           y_min=self.params['logph']['y_min'],
                           y_max=self.params['logph']['y_max'])

        # Calculate components process data
        for compdata in result_dict.values():
            compdata['datapoints'] = (diagram.calc_individual_isoline(**compdata))
        # breakpoint()

        # Isolines
        diagram.calc_isolines()
        diagram.draw_isolines('logph')

        # Draw heat pump process over fluid property diagram
        for compdata in result_dict.values():
            datapoints = compdata['datapoints']
            diagram.ax.plot(datapoints['h'], datapoints['p'], color='#EC6707')
            diagram.ax.scatter(datapoints['h'][0], datapoints['p'][0], color='#B54036')

        if path is not None:
            diagram.save(path, dpi=300)

        if return_diagram:
            return diagram

    # Generate and plot T-s-diagram of the heat pump process.
    def plot_Ts(self, path=None, return_diagram=False):
        """
        Generate and plot T-s-diagram of heat pump process.

        Args:
            path (str): Path to save the diagram image.
            return_diagram (bool): Whether to return the diagram object.

        Returns:
            FluidPropertyDiagram or None: The diagram object if return_diagram is True, else None.
        """

        result_dict = self.get_states()

        # Initialize fluid property diagram
        diagram = FluidPropertyDiagram(self.params['setup']['refrig'])
        diagram.set_unit_system(T='°C', p='bar', h='kJ/kg')
        diagram.set_limits(x_min=self.params['Ts']['x_min'],
                           x_max=self.params['Ts']['x_max'],
                           y_min=self.params['Ts']['y_min'],
                           y_max=self.params['Ts']['y_max'])

        # Calculate components process data
        for compdata in result_dict.values():
            compdata['datapoints'] = (diagram.calc_individual_isoline(**compdata))
        # breakpoint()

        # Isolines
        diagram.calc_isolines()
        diagram.draw_isolines('Ts')

        # Draw heat pump process over fluid property diagram
        for compdata in result_dict.values():
            datapoints = compdata['datapoints']
            diagram.ax.plot(datapoints['s'], datapoints['T'], color='#EC6707')
            diagram.ax.scatter(datapoints['s'][0], datapoints['T'][0], color='#B54036')

        if path is not None:
            diagram.save(path, dpi=300)

        if return_diagram:
            return diagram

    # Generate a QT diagram for the heat pump component.
    def qt_diagram(self, hot_in, hot_out, cold_in, cold_out, delta_t_min, case, plot=False, path=None):
        """
        Generate a QT diagram for the heat pump component.

        Args:
            hot_in: Input hot stream
            hot_out: Output hot stream
            cold_in: Input cold stream
            cold_out: Output cold stream
            delta_t_min (float): Minimum temperature difference.
            plot (bool): Whether to display the diagram.
            path (str): Path to save the diagram image.

        Returns:
            list: List containing the minimum and maximum temperature differences.
        """

        conn_hot_in = self.network.get_conn(str(hot_in))
        conn_cold_in = self.network.get_conn(str(cold_in))

        #  get the name of the heat exchanger
        component_name = []
        for component in [hot_in, hot_out, cold_in, cold_out]:
            component_name.append(self.network.get_conn(str(component)).target.label)
        component_name = [item for item in set(component_name) if component_name.count(item) > 1][0]

        # read the fluid of the streams
        fluid_hot = next(key for key, value in conn_hot_in.fluid.val.items() if value == 1)
        fluid_cold = next(key for key, value in conn_cold_in.fluid.val.items() if value == 1)

        # get start and end values for the diagram
        T_hot_out = self.network.results['Connection'].loc[str(hot_out), 'T']
        T_cold_in = self.network.results['Connection'].loc[str(cold_in), 'T']
        h_hot_in = self.network.results['Connection'].loc[str(hot_in), 'h']
        h_hot_out = self.network.results['Connection'].loc[str(hot_out), 'h']
        h_cold_in = self.network.results['Connection'].loc[str(cold_in), 'h']
        h_cold_out = self.network.results['Connection'].loc[str(cold_out), 'h']
        p_hot_in = self.network.results['Connection'].loc[str(hot_in), 'p']
        p_hot_out = self.network.results['Connection'].loc[str(hot_out), 'p']
        p_cold_in = self.network.results['Connection'].loc[str(cold_in), 'p']
        p_cold_out = self.network.results['Connection'].loc[str(cold_out), 'p']
        m = self.network.results['Connection'].loc[str(cold_in), 'm']

        step_number = 200
        T_cold = [T_cold_in]
        T_hot = [T_hot_out]
        H_plot = [0]

        for i in np.linspace(1, step_number, step_number):
            h_hot = h_hot_out + (h_hot_in-h_hot_out)/step_number*i
            p_hot = p_hot_out + (p_hot_in-p_hot_out)/step_number*i
            T_hot.append(PSI('T', 'H', h_hot * 1e3, 'P', p_hot * 1e5, f'REFPROP::{fluid_hot}') - 273.15)
            h_cold = h_cold_in + (h_cold_out-h_cold_in)/step_number*i
            p_cold = p_cold_in - (p_cold_in-p_cold_out)/step_number*i
            T_cold.append(PSI('T', 'H', h_cold * 1e3, 'P', p_cold * 1e5, f'REFPROP::{fluid_cold}') - 273.15)
            H_plot.append((h_cold-h_cold_in)*m)

        difference = [x - y for x, y in zip(T_hot, T_cold)]

        # plot the results
        mpl.rcParams['font.size'] = 16
        plt.figure(figsize=(14, 7))
        plt.plot(H_plot, T_hot, color='red')
        plt.plot(H_plot, T_cold, color='blue')
        plt.legend(["Hot side", "Cold side"])
        plt.xlabel('Q [kW]')
        plt.grid(True)
        plt.ylabel('T [°C]')
        plt.xlim(0, max(H_plot))
        plt.ylim(min(T_cold)-10, max(T_hot)+10)
        plt.title("QT diagram of the " + component_name + f" of the HP \nfor the case: {case}")

        if path is not None:
            plt.savefig(path)
        if plot:
            plt.show()


        if min(difference) > delta_t_min - 1e-3:
            print("The min. temperature difference of the " + component_name + " of the HP is " + str(round(min(difference), 2)) + "K and is equal or higher than the allowed min. (" + str(
                delta_t_min) + "K).")
        else:
            print("The min. temperature difference of the " + component_name + " of the HP is " + str(round(min(difference), 2)) + "K and is lower than the allowed min. (" + str(
                delta_t_min) + "K).")

        return [min(difference), max(difference)]
