import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython.display import display, Math

# Figure and font setting s
figure_size = (20, 12)
font_size = 18
plt.rc('font', family='Arial', size=font_size)

# Scenarios and titles
scenario_order = ["130_70", "95_70", "120_70", "95_65", "110_70", "95_60"]
scenario_titles = {
    "130_70": "HT-PTES (130-70)",
    "120_70": "HT-PTES (120-70)",
    "110_70": "HT-PTES (110-70)",
    "95_70": "LT-PTES (95-70)",
    "95_65": "LT-PTES (95-65)",
    "95_60": "LT-PTES (95-60)"
}

# Components and colors
components_hp = ["comp", "cond", "ihx", "val", "eva"]
components_orc = ["pump", "eva", "ihx", "exp", "cond"]
colors_hp = ['#ffcccc', '#ff9999', '#ff6666', '#cc0000', '#990000']
colors_orc = ['#ccccff', '#9999ff', '#6666ff', '#3333ff', '#0000ff']

# Plot function
def plot_ed_av_sum(ax, df, name_mapping, components, title, color):
    component_labels = [name_mapping.get(comp, comp).upper() for comp in components]
    values = []
    for comp in components:
        try:
            value = df.loc[(comp.lower(), slice(None)), "ED AV SUM [kW]"].sum()
            values.append(value)
        except KeyError:
            values.append(0)

    # Plot bars
    ax.barh(component_labels, values, color=color, edgecolor="black", linewidth=1, height=0.9)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel(r"Exergy destruction $\dot{E}_{{D,k}}^{{AV},\Sigma}$")
    ax.text(0.57, 1.1, title, ha='right', va='top', transform=ax.transAxes, fontsize=font_size)

# Create figure and subplots
fig, axes = plt.subplots(3, 2, figsize=figure_size)

for idx, scenario in enumerate(scenario_order):
    row, col = divmod(idx, 2)
    ax_main = axes[row, col]
    ax_main.set_title(scenario_titles[scenario], fontsize=font_size)

    # Create inset axes for HP and ORC
    ax_hp = inset_axes(ax_main, width="43%", height="80%", loc='center left', borderpad=0)
    ax_orc = inset_axes(ax_main, width="43%", height="80%", loc='center right', borderpad=0)

    # Paths for the data
    hp_path = f"../hp_ihx_R1336MZZZ_{scenario}/hp_adex_analysis.csv"
    orc_path = f"../orc_ihx_R152a_{scenario}/orc_adex_analysis.csv"

    # Fallback paths
    if not os.path.exists(hp_path) or not os.path.exists(orc_path):
        hp_path = f"../hp_ihx_R245fa_{scenario}/hp_adex_analysis.csv"
        orc_path = f"../orc_ihx_R134a_{scenario}/orc_adex_analysis.csv"

    # Load and plot data
    if os.path.exists(hp_path):
        df_hp = pd.read_csv(hp_path, index_col=[0, 1])
        name_mapping_hp = {'eva': 'EVA1', 'comp': 'COMP', 'cond': 'COND1', 'ihx': 'IHX1', 'val': 'VAL'}
        plot_ed_av_sum(ax_hp, df_hp, name_mapping_hp, components_hp, "HP", colors_hp[2])

    if os.path.exists(orc_path):
        df_orc = pd.read_csv(orc_path, index_col=[0, 1])
        name_mapping_orc = {'eva': 'EVA2', 'pump': 'PUMP', 'exp': 'EXP', 'cond': 'COND2', 'ihx': 'IHX2'}
        plot_ed_av_sum(ax_orc, df_orc, name_mapping_orc, components_orc, "ORC", colors_orc[2])

    # Set x-limits for consistency
    ax_hp.set_xlim(left=0)
    ax_orc.set_xlim(left=0)

    # Remove axis visuals for main container
    ax_main.set_axis_off()

# Adjust layout and save
plt.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.05, wspace=0.15, hspace=0.25)
plt.savefig("ed_av_sum.pdf", dpi=300, bbox_inches='tight')
plt.show()
