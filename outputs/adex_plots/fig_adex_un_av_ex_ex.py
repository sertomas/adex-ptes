import pandas as pd
import os
import matplotlib.pyplot as plt

plt.rc('font', family='Arial', size=18)

scenarios = [
    "140_70",
    "130_70",
    "120_70",
    "95_70",
    "95_60",
    "95_50"
]


def plot_exergy_avoid_endo(ax, df, name_mapping, components, title, colors, labels):
    """
    Plots exergy destruction for various components of a system, differentiating between unavoidable and avoidable parts.

    This function creates a horizontal bar plot for each specified component, showing the contribution of unavoidable endogenous,
    unavoidable exogenous, avoidable endogenous, and avoidable exogenous exergy destruction. Positive and negative values are plotted
    separately to align correctly.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes on which to plot.
    - df (DataFrame): A DataFrame containing the exergy destruction data for the components.
    - name_mapping (dict): A dictionary mapping component names to their display names.
    - components (list): A list of component names to plot.
    - title (str): The title for the plot.
    - colors (list): A list of colors to use for the different parts of exergy destruction.
    - labels (list): A list of labels corresponding to the different parts of exergy destruction.

    Returns:
    None
    """
    # Mapping of component names for display purposes
    components_mapped = [name_mapping.get(component, component).upper() for component in components]

    # Plotting each component
    for i, component in enumerate(components):
        # Gather positive and negative totals separately to manage left offset for negatives
        totals_positive = []
        totals_negative = []

        # Define the values to plot
        values = [
            df.loc[component, 'ED UN EN [kW]'].sum(),
            df.loc[component, 'ED UN EX [kW]'].sum(),
            df.loc[component, 'ED AV EN [kW]'].sum(),
            df.loc[component, 'ED AV EX [kW]'].sum()
        ]

        # Accumulate positive and negative values separately
        for value in values:
            if value >= 0:
                totals_positive.append(value)
                totals_negative.append(0)  # Placeholder for alignment
            else:
                totals_positive.append(0)  # Placeholder for alignment
                totals_negative.append(value)

        # Plot positive values
        left_positive = 0
        for j, total in enumerate(totals_positive):
            if total > 0:
                ax.barh(components_mapped[i], total, left=left_positive, color=colors[j], edgecolor="black", linewidth=1, height=0.9)
                left_positive += total

        # Plot negative values
        left_negative = 0
        for j, total in enumerate(totals_negative):
            if total < 0:
                ax.barh(components_mapped[i], total, left=left_negative, color=colors[j], edgecolor="black", linewidth=1, height=0.9)
                left_negative += total

    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Exergy destruction [kW]')
    ax.text(0.97, 0.95, title, ha='right', va='top', transform=ax.transAxes, fontsize=18)


components_hp = ["comp", "cond", "ihx", "val", "eva"]
components_orc = ["pump", "eva", "ihx", "exp", "cond"]
colors = ['#ff6666', '#ffcccc', '#66cc66', '#ccffcc']  # other nice colors: ['#ff6666', '#ffcccc', '#9090ff', '#ccccff']
labels = [r'$\dot{E}^\mathrm{UN,EN}_{\mathrm{D},k}$', r'$\dot{E}^\mathrm{UN,EX}_{\mathrm{D},k}$', r'$\dot{E}^\mathrm{AV,EN}_{\mathrm{D},k}$', '$\dot{E}^\mathrm{AV,EX}_{\mathrm{D},k}$']

# Iterate over scenarios
for scenario in scenarios:
    hp_path = f"../hp_ihx_R1336MZZZ_{scenario}/hp_adex_analysis.csv"
    orc_path = f"../orc_ihx_R152a_{scenario}/orc_adex_analysis.csv"
    if not os.path.exists(hp_path) and not os.path.exists(orc_path):
        hp_path = f"../hp_ihx_R245fa_{scenario}/hp_adex_analysis.csv"
        orc_path = f"../orc_ihx_R134a_{scenario}/orc_adex_analysis.csv"

    if os.path.exists(hp_path) and os.path.exists(orc_path):
        df_hp_mexo = pd.read_csv(hp_path, index_col=[0, 1])
        df_orc_mexo = pd.read_csv(orc_path, index_col=[0, 1])

        fig, axs = plt.subplots(1, 2, figsize=(14.5, 3.2))

        # Plot HP
        name_mapping_hp = {'eva': 'EVA1', 'comp': 'COMP', 'cond': 'COND1', 'ihx': 'IHX1', 'val': 'VAL'}
        plot_exergy_avoid_endo(axs[0], df_hp_mexo, name_mapping_hp, components_hp, "HP", colors, labels)

        # Plot ORC
        name_mapping_orc = {'eva': 'EVA2', 'pump': 'PUMP', 'exp': 'EXP', 'cond': 'COND2', 'ihx': 'IHX2'}
        plot_exergy_avoid_endo(axs[1], df_orc_mexo, name_mapping_orc, components_orc, "ORC", colors, labels)

        # Add legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for color, label in
                   zip(colors[:4], labels[:4])]
        fig.legend(handles=handles, bbox_to_anchor=(0.58, 0.85), ncol=1, frameon=False)

        # Set x-limits
        axs[0].set_xlim([-15, df_orc_mexo["ED [kW]"].max()+25])
        axs[1].set_xlim([-15, df_orc_mexo["ED [kW]"].max()+25])

        # Adjust layout
        plt.subplots_adjust(left=0.08, right=0.99, bottom=0.21, top=0.97, wspace=0.65)

        # Save the figure
        plt.savefig(f'adex_un_av_en_ex_{scenario}.pdf')
        plt.show()