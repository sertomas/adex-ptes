import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='Arial', size=18)


def plot_exergy_destruction(ax, df, name_mapping, components, title, colors):
    # Initialize arrays for plotting
    totals = {'positive': [], 'negative': []}

    # Mapping of component names for display purposes
    components_mapped = [name_mapping.get(component, component).upper() for component in components]

    # Calculate totals for each component
    for component in components:
        totals_comp = {'positive': [0, 0, 0, 0], 'negative': [0, 0, 0, 0]}

        for i, col in enumerate(['ED EN UN [kW]', 'ED EX UN [kW]', 'ED EN AV [kW]', 'ED EX AV [kW]']):
            value = df.loc[component, col]
            if value > 0:
                totals_comp['positive'][i] = value
            else:
                totals_comp['negative'][i] = value

        totals['positive'].append(totals_comp['positive'])
        totals['negative'].append(totals_comp['negative'])

    # Plotting using mapped component names
    for i, component_totals in enumerate(zip(totals['positive'], totals['negative'])):
        comp_mapped = components_mapped[i]
        for totals in component_totals:
            left = 0
            for total in totals:
                if total != 0:
                    color_index = ['ED EN UN [kW]', 'ED EX UN [kW]', 'ED EN AV [kW]', 'ED EX AV [kW]'].index(
                        ['ED EN UN [kW]', 'ED EX UN [kW]', 'ED EN AV [kW]', 'ED EX AV [kW]'][totals.index(total)])
                    ax.barh(comp_mapped, total, left=left, color=colors[color_index], edgecolor="black", linewidth=1,
                            height=0.9)
                    left += total

    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Exergy destruction [kW]')
    ax.text(0.97, 0.95, title, ha='right', va='top', transform=ax.transAxes, fontsize=18)


# Ensure to define or load your DataFrame `df_hp_mexo` and `df_orc_mexo` here
df_hp_mexo = pd.read_csv('../adex_hp/final_results_hp.csv', index_col=[0, 1])
df_orc_mexo = pd.read_csv('../adex_orc/final_results_orc.csv', index_col=[0, 1])

# Define your components and other variables as before, keeping the original data labels
components_hp = ["comp", "cond", "ihx", "val", "eva"]  # Original component names
components_orc = ["pump", "eva", "ihx", "exp", "cond"]
colors = ['#ccccff', '#ccffcc', '#ff6666', '#ffcc99']
labels = [r'$\dot{E}^\mathrm{EN}_{\mathrm{D},k}$', r'$\dot{E}^\mathrm{MEXO}_{\mathrm{D},k}$', r'$\dot{E}^{\mathrm{EX},kl}_{\mathrm{D},k}$', 'ED EX AV [kW]']

# Assuming `df_hp_mexo` and `df_orc_mexo` are your DataFrame variables

fig, axs = plt.subplots(1, 2, figsize=(14.5, 3.2))

# Plot for Heat Pump
name_mapping_hp = {'eva': 'EVA1', 'comp': 'COMP', 'cond': 'COND1', 'ihx': 'IHX1', 'val': 'VAL', 'pump': 'PUMP', 'exp': 'EXP'}
plot_exergy_destruction(axs[0], df_hp_mexo, name_mapping_hp, components_hp, "HP", colors, labels)

# Plot for ORC
name_mapping_orc = {'eva': 'EVA2', 'comp': 'COMP', 'cond': 'COND2', 'ihx': 'IHX2', 'val': 'VAL', 'pump': 'PUMP', 'exp': 'EXP'}
plot_exergy_destruction(axs[1], df_orc_mexo, name_mapping_orc, components_orc, "ORC", colors, labels)

# Create handles for the legend and add it to the last plot
handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for color, label in zip(colors[:3], labels[:3])]
fig.legend(handles=handles, bbox_to_anchor=(0.58, 0.85), ncol=1, frameon=False)
axs[0].set_xlim([-25, 175])
axs[1].set_xlim([-25, 175])
# plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.99, bottom=0.21, top=0.97, wspace=0.65)

plt.savefig('adex_combined_unavoid.pdf')
plt.show()
