import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


figure_size = (20, 12)
font_size = 18

plt.rc('font', family='Arial', size=font_size)

# Alternate HT and LT scenarios for left and right columns
scenario_order = ["130_70", "95_70", "120_70", "95_65", "110_70", "95_60"]
scenario_titles = {
    "130_70": "HT-PTES (130-70)",
    "120_70": "HT-PTES (120-70)",
    "110_70": "HT-PTES (110-70)",
    "95_70": "LT-PTES (95-70)",
    "95_65": "LT-PTES (95-65)",
    "95_60": "LT-PTES (95-60)"
}

# Define components, colors, labels, and function for plotting exergy destruction
components_hp = ["comp", "cond", "ihx", "val", "eva"]
components_orc = ["pump", "eva", "ihx", "exp", "cond"]
colors = ['#ff6666', '#ffcccc', '#66cc66', '#ccffcc']
labels = [r'$\dot{E}^\mathrm{UN,EN}_{\mathrm{D},k}$',
          r'$\dot{E}^\mathrm{UN,EX}_{\mathrm{D},k}$',
          r'$\dot{E}^\mathrm{AV,EN}_{\mathrm{D},k}$',
          r'$\dot{E}^\mathrm{AV,EX}_{\mathrm{D},k}$']

def plot_exergy_avoid_endo(ax, df, name_mapping, components, title, colors, labels):
    components_mapped = [name_mapping.get(comp, comp).upper() for comp in components]
    for i, comp in enumerate(components):
        totals_positive = []
        totals_negative = []
        values = [
            df.loc[comp, 'ED UN EN [kW]'].sum(),
            df.loc[comp, 'ED UN EX [kW]'].sum(),
            df.loc[comp, 'ED AV EN [kW]'].sum(),
            df.loc[comp, 'ED AV EX [kW]'].sum()
        ]
        for val in values:
            if val >= 0:
                totals_positive.append(val)
                totals_negative.append(0)
            else:
                totals_positive.append(0)
                totals_negative.append(val)
        left_positive = 0
        for j, total in enumerate(totals_positive):
            if total > 0:
                ax.barh(components_mapped[i], total, left=left_positive, color=colors[j],
                        edgecolor="black", linewidth=1, height=0.9)
                left_positive += total
        left_negative = 0
        for j, total in enumerate(totals_negative):
            if total < 0:
                ax.barh(components_mapped[i], total, left=left_negative, color=colors[j],
                        edgecolor="black", linewidth=1, height=0.9)
                left_negative += total
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Exergy destruction [kW]')
    ax.text(0.97, 0.95, title, ha='right', va='top', transform=ax.transAxes, fontsize=font_size)

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

        # Plot HP exergy destruction
        name_mapping_hp = {'eva': 'EVA1', 'comp': 'COMP', 'cond': 'COND1', 'ihx': 'IHX1', 'val': 'VAL'}
        plot_exergy_avoid_endo(ax_hp, df_hp_mexo, name_mapping_hp, components_hp, "HP", colors, labels)

        # Plot ORC exergy destruction
        name_mapping_orc = {'eva': 'EVA2', 'pump': 'PUMP', 'exp': 'EXP', 'cond': 'COND2', 'ihx': 'IHX2'}
        plot_exergy_avoid_endo(ax_orc, df_orc_mexo, name_mapping_orc, components_orc, "ORC", colors, labels)

        # Set x-limits for consistency
        max_val_hp = df_hp_mexo.filter(like='ED').max().max()
        max_val_orc = df_orc_mexo.filter(like='ED').max().max()
        max_val = max(max_val_hp, max_val_orc) + 25
        ax_hp.set_xlim([-15, max_val])
        ax_orc.set_xlim([-15, max_val])

    # Remove axis visuals from main container axis
    ax_main.set_axis_off()

# Create a single legend for the entire figure
legend_handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=font_size, frameon=False)

plt.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.1, wspace=0.15, hspace=0.25)

plt.savefig("advanced_exergy_avoid_unavoid.pdf", dpi=300, bbox_inches='tight')
plt.show()
