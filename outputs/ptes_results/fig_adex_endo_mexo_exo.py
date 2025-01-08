import pandas as pd
import matplotlib.pyplot as plt

plt.rc('font', family='Arial', size=18)


def plot_exergy_destruction(ax, df, name_mapping, components, title, colors):
    # Initialize arrays for plotting
    totals_positive = []
    totals_negative = []
    ed_ex_l_positive = {}
    ed_ex_l_negative = {}

    # Mapping of component names for display purposes
    components_mapped = [name_mapping.get(component, component).upper() for component in components]

    # Calculate totals for each component
    for component in components:
        total_positive = [df.loc[component, "ED EN [kW]"].sum()]
        total_negative = [0]  # Placeholder for symmetry in plotting

        mexo_values = df.loc[component, "ED MX [kW]"].sum()
        if mexo_values > 0:
            total_positive.append(mexo_values)
            total_negative.append(0)  # Placeholder
        else:
            total_positive.append(0)  # Placeholder
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

    components_uppercase = [component.upper() for component in components]

    # Plotting
    for i, totals in enumerate(totals_positive):
        left = 0
        for j, total in enumerate(totals):
            if total > 0:
                ax.barh(components_uppercase[i], total, left=left, color=colors[j], edgecolor="black", linewidth=1,
                        height=0.9)
                left += total

    for i, totals in enumerate(totals_negative):
        left = 0
        for j, total in enumerate(totals):
            if total < 0:
                ax.barh(components_uppercase[i], total, left=left, color=colors[j], edgecolor="black", linewidth=1,
                        height=0.9)
                left += total

    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Exergy destruction [kW]')

    # Adding text in the top right corner
    ax.text(0.97, 0.95, title, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='right')

# Ensure to define or load your DataFrame `df_hp_mexo` and `df_orc_mexo` here
df_hp_mexo = pd.read_csv('../adex_hp/hp_adex_analysis.csv', index_col=[0, 1])
df_orc_mexo = pd.read_csv('../adex_orc/orc_adex_analysis.csv', index_col=[0, 1])

# Define your components and other variables as before, keeping the original data labels
components_hp = ["comp", "cond", "ihx", "val", "eva"]  # Original component names
components_orc = ["pump", "eva", "ihx", "exp", "cond"]
colors = ['#ccccff', '#ccffcc', '#ff6666', '#ffcc99']
labels = [r'$\dot{E}^\mathrm{EN}_{\mathrm{D},k}$', r'$\dot{E}^\mathrm{MEXO}_{\mathrm{D},k}$',
          r'$\dot{E}^{\mathrm{EX},kl}_{\mathrm{D},k}$', 'ED AV EX [kW]']

# Assuming `df_hp_mexo` and `df_orc_mexo` are your DataFrame variables

fig, axs = plt.subplots(1, 2, figsize=(14.5, 3.2))

# Plot for Heat Pump
name_mapping_hp = {'eva': 'EVA1', 'comp': 'COMP', 'cond': 'COND1', 'ihx': 'IHX1', 'val': 'VAL', 'pump': 'PUMP',
                   'exp': 'EXP'}
plot_exergy_destruction(axs[0], df_hp_mexo, name_mapping_hp, components_hp, "HP", colors)

# Plot for ORC
name_mapping_orc = {'eva': 'EVA2', 'comp': 'COMP', 'cond': 'COND2', 'ihx': 'IHX2', 'val': 'VAL', 'pump': 'PUMP',
                    'exp': 'EXP'}
plot_exergy_destruction(axs[1], df_orc_mexo, name_mapping_orc, components_orc, "ORC", colors)

# Create handles for the legend and add it to the last plot
handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=label) for color, label in zip(colors[:3], labels[:3])]
fig.legend(handles=handles, bbox_to_anchor=(0.58, 0.85), ncol=1, frameon=False)
axs[0].set_xlim([-25, 175])
axs[1].set_xlim([-25, 175])
# plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.99, bottom=0.21, top=0.97, wspace=0.65)

plt.savefig('adex_endo_mexo_exo.pdf')
plt.show()
