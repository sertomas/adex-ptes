import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


figure_size = (20, 12)
font_size = 18

plt.rc('font', family='Arial', size=font_size)

# Define scenarios and titles for subplots with HT on left and LT on right
scenario_order = ["130_70", "95_70", "120_70", "95_65", "110_70", "95_60"]
scenario_titles = {
    "130_70": "HT-PTES (130-70)",
    "120_70": "HT-PTES (120-70)",
    "110_70": "HT-PTES (110-70)",
    "95_70": "LT-PTES (95-70)",
    "95_65": "LT-PTES (95-65)",
    "95_60": "LT-PTES (95-60)"
}

# Define components, colors, labels for legend, and function for plotting exergy destruction
components_hp = ["comp", "cond", "ihx", "val", "eva"]
components_orc = ["pump", "eva", "ihx", "exp", "cond"]
colors = ['#ccccff', '#ccffcc', '#ff6666', '#ffcc99']
labels = [r'$\dot{E}^\mathrm{EN}_{\mathrm{D},k}$',
          r'$\dot{E}^\mathrm{MEXO}_{\mathrm{D},k}$',
          r'$\dot{E}^{\mathrm{EX},kl}_{\mathrm{D},k}$',
          'ED AV EX [kW]']

def plot_exergy_destruction(ax, df, name_mapping, components, title, colors):
    # Initialize arrays for plotting
    totals_positive = []
    totals_negative = []
    ed_ex_l_positive = {}
    ed_ex_l_negative = {}

    # Mapping of component names for display purposes with numbering preserved
    components_mapped = [name_mapping.get(component, component).upper() for component in components]

    # Calculate totals for each component
    for component in components:
        total_positive = [df.loc[component, "ED EN [kW]"].sum()]
        total_negative = [0]  # Placeholder for symmetry in plotting

        mexo_values = df.loc[component, "ED MX [kW]"].sum()
        if mexo_values > 0:
            total_positive.append(mexo_values)
            total_negative.append(0)
        else:
            total_positive.append(0)
            total_negative.append(mexo_values)

        for other_component in components:
            if other_component != component:
                key = (component, other_component)
                value = df.loc[(component, other_component), "ED EX kl [kW]"].sum()
                if value < 0:
                    ed_ex_l_negative[key] = value
                else:
                    ed_ex_l_positive[key] = value

        positive_ex_l = sum(value for (comp1, comp2), value in ed_ex_l_positive.items() if comp1 == component)
        negative_ex_l = sum(value for (comp1, comp2), value in ed_ex_l_negative.items() if comp1 == component)

        total_positive.append(positive_ex_l)
        total_negative.append(negative_ex_l)

        totals_positive.append(total_positive)
        totals_negative.append(total_negative)

    # Use the mapped component labels with numbering for plotting
    components_labels = components_mapped

    # Plotting positive values
    for i, totals in enumerate(totals_positive):
        left = 0
        for j, total in enumerate(totals):
            if total > 0:
                ax.barh(components_labels[i], total, left=left, color=colors[j],
                        edgecolor="black", linewidth=1, height=0.9)
                left += total

    # Plotting negative values
    for i, totals in enumerate(totals_negative):
        left = 0
        for j, total in enumerate(totals):
            if total < 0:
                ax.barh(components_labels[i], total, left=left, color=colors[j],
                        edgecolor="black", linewidth=1, height=0.9)
                left += total

    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Exergy destruction [kW]')
    ax.text(0.57, 1.1, title, transform=ax.transAxes, fontsize=font_size,
            verticalalignment='top', horizontalalignment='right')

# Create figure and 3x2 grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=figure_size)

for idx, scenario in enumerate(scenario_order):
    row = idx // 2
    col = idx % 2
    ax_main = axes[row, col]
    ax_main.set_title(scenario_titles.get(scenario, scenario), fontsize=font_size)

    # Create inset axes for HP and ORC within the current subplot
    ax_hp = inset_axes(ax_main, width="43%", height="80%", loc='center left', borderpad=0)
    ax_orc = inset_axes(ax_main, width="43%", height="80%", loc='center right', borderpad=0)

    # Load data for current scenario
    hp_path = f"../hp_ihx_R1336MZZZ_{scenario}/hp_adex_analysis.csv"
    orc_path = f"../orc_ihx_R152a_{scenario}/orc_adex_analysis.csv"
    if not os.path.exists(hp_path) or not os.path.exists(orc_path):
        hp_path = f"../hp_ihx_R245fa_{scenario}/hp_adex_analysis.csv"
        orc_path = f"../orc_ihx_R134a_{scenario}/orc_adex_analysis.csv"

    if os.path.exists(hp_path) and os.path.exists(orc_path):
        df_hp_mexo = pd.read_csv(hp_path, index_col=[0, 1])
        df_orc_mexo = pd.read_csv(orc_path, index_col=[0, 1])

        # Plot HP exergy destruction for this scenario
        name_mapping_hp = {
            'eva': 'EVA1',
            'comp': 'COMP',
            'cond': 'COND1',
            'ihx': 'IHX1',
            'val': 'VAL',
            'pump': 'PUMP',
            'exp': 'EXP'
        }
        plot_exergy_destruction(ax_hp, df_hp_mexo, name_mapping_hp, components_hp, "HP", colors)

        # Plot ORC exergy destruction for this scenario
        name_mapping_orc = {
            'eva': 'EVA2',
            'pump': 'PUMP',
            'ihx': 'IHX2',
            'exp': 'EXP',
            'cond': 'COND2'
        }
        plot_exergy_destruction(ax_orc, df_orc_mexo, name_mapping_orc, components_orc, "ORC", colors)

        # Set x-limits for consistency using [-15, max_val]
        max_val_hp = df_hp_mexo.filter(like='ED').max().max()
        max_val_orc = df_orc_mexo.filter(like='ED').max().max()
        max_val = max(max_val_hp, max_val_orc) + 25
        ax_hp.set_xlim([-15, max_val])
        ax_orc.set_xlim([-15, max_val])

    # Remove axis visuals from main container axis
    ax_main.set_axis_off()

legend_handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors[:3]))]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=font_size, frameon=False)

plt.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.1, wspace=0.15, hspace=0.25)

legend_handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors[:3]))]

# Save the combined figure
plt.savefig("advanced_exergy_endo_exo_mexo.pdf", dpi=300, bbox_inches='tight')
plt.show()
