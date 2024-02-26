import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rc('font', family='Times New Roman', size=20)
plt.rc('mathtext', fontset='stix')

# Load the data
df_hp_comps_base = pd.read_csv('../adex_hp/hp_comps_real.csv', index_col=0)
df_orc_comps_base = pd.read_csv('../adex_orc/orc_comps_real.csv', index_col=0)

ep = [100,
      df_hp_comps_base.loc['tot', 'epsilon']*100,
      df_hp_comps_base.loc['tot', 'epsilon']*100,
      df_hp_comps_base.loc['tot', 'epsilon']*df_orc_comps_base.loc['tot', 'epsilon']*100,
      df_hp_comps_base.loc['tot', 'epsilon']*df_orc_comps_base.loc['tot', 'epsilon']*100]

# Detailed breakdown of ED for HP components
ed_hp = [df_hp_comps_base.loc['comp', 'ED [kW]']/df_hp_comps_base.loc['tot', 'EF [kW]']*100,
         df_hp_comps_base.loc['cond', 'ED [kW]']/df_hp_comps_base.loc['tot', 'EF [kW]']*100,
         df_hp_comps_base.loc['ihx', 'ED [kW]']/df_hp_comps_base.loc['tot', 'EF [kW]']*100,
         df_hp_comps_base.loc['val', 'ED [kW]']/df_hp_comps_base.loc['tot', 'EF [kW]']*100,
         df_hp_comps_base.loc['eva', 'ED [kW]']/df_hp_comps_base.loc['tot', 'EF [kW]']*100
         ]
# Assuming an arbitrary breakdown for ORC components for demonstration
ed_orc = [df_orc_comps_base.loc['pump', 'ED [kW]']/df_orc_comps_base.loc['tot', 'EF [kW]']*100*df_hp_comps_base.loc['tot', 'epsilon'],
         df_orc_comps_base.loc['eva', 'ED [kW]']/df_orc_comps_base.loc['tot', 'EF [kW]']*100*df_hp_comps_base.loc['tot', 'epsilon'],
         df_orc_comps_base.loc['ihx', 'ED [kW]']/df_orc_comps_base.loc['tot', 'EF [kW]']*100*df_hp_comps_base.loc['tot', 'epsilon'],
         df_orc_comps_base.loc['exp', 'ED [kW]']/df_orc_comps_base.loc['tot', 'EF [kW]']*100*df_hp_comps_base.loc['tot', 'epsilon'],
         df_orc_comps_base.loc['cond', 'ED [kW]']/df_orc_comps_base.loc['tot', 'EF [kW]']*100*df_hp_comps_base.loc['tot', 'epsilon']
         ]

# Creating custom labels for each component of ed_hp and ed_orc
labels_hp = ['COMP', 'COND', 'IHX', 'VAL', 'EVA']
labels_orc = ['PUMP', 'EVA', 'IHX', 'EXP', 'COND']

fig, ax = plt.subplots()
fig.set_size_inches(12, 3.5)

# Reverse the order of groups for plotting
groups = ['Output', 'ORC', 'TES', 'HP', 'Input']

# Reversed EP for horizontal plotting, to match the reversed groups
ep_reversed = [ep[i] for i in [4, 3, 2, 1, 0]]

# Since we're reversing the group order, no need to reverse each component's data,
# but we need to adjust the starting points accordingly

# Left for ED_HP starts with EP, adjusted for reversed order
left_for_ed_hp_reversed = ep_reversed.copy()

# Left for ED_ORC starts with EP plus total ED of HP (since ORC is after HP), adjusted for reversed order
left_for_ed_orc_reversed = [ep_reversed[i] + sum(ed_hp) if i == 3 else ep_reversed[i] for i in range(len(ep_reversed))]

# Plotting EP horizontally with reversed order
ax.barh(groups, ep_reversed, color="lightgrey", edgecolor="black", linewidth=1, height=0.9)

#               comp        cond        ihx      val        eva
#               pump        cond        ihx      exp        cond
colors_hp = ['#ffcccc', '#ff9999', '#ff6666', '#cc0000', '#990000']
colors_orc = ['#ccccff', '#9999ff', '#6666ff', '#3333ff', '#0000ff']

# Plotting detailed ED for HP components horizontally, adjusted for reversed order
for i, value in enumerate(ed_hp):
    ax.barh(groups[3], value, left=left_for_ed_hp_reversed[3], edgecolor="black", linewidth=1, color=colors_hp[i], height=0.9)
    left_for_ed_hp_reversed[3] += value

# Plotting detailed ED for ORC components horizontally, adjusted for reversed order
for i, value in enumerate(ed_orc):
    ax.barh(groups[1], value, left=left_for_ed_orc_reversed[1], edgecolor="black", linewidth=1, color=colors_orc[i], height=0.9)
    left_for_ed_orc_reversed[1] += value

ep_reversed = ep[::-1]

# Adding labels inside the bars for reversed ep values only
for i, bar in enumerate(ax.patches[:len(ep_reversed)]):  # Iterate only through the first bars corresponding to 'ep'
    ax.text(x=bar.get_x() + bar.get_width() / 2,  # Position text at the center of the bar's width
            y=bar.get_y() + bar.get_height() / 2,  # Center text vertically in the bar
            s=f"{ep_reversed[i]:.1f}",  # Use the corresponding reversed 'ep' value for the text
            ha='center',  # Center text horizontally
            va='center',  # Center text vertically
            color='black',  # Set text color for visibility
            # weight='bold',  # Bold text
            )

# Creating custom legend
legend_patches = [mpatches.Patch(color=colors_hp[i], label=labels_hp[i]) for i in range(len(labels_hp))] + \
                 [mpatches.Patch(color=colors_orc[i], label=labels_orc[i]) for i in range(len(labels_orc))]

# Displaying the legend
ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1), ncol=2)

# Setting the x-axis label and limits
label_text = "Exergetic efficiency of the overall system $\\varepsilon = 1 - y_\\mathrm{D} - y_\\mathrm{L}$ [%]"
ax.set_xlabel(label_text)
ax.set_xlim(left=0)

# Adjust subplot parameters to give more space for the legend
plt.subplots_adjust(left=0.08)  # Adjust this value as needed to fit the legend
plt.subplots_adjust(right=0.65)  # Adjust this value as needed to fit the legend
plt.subplots_adjust(bottom=0.25)

plt.savefig('exergy_conversion.png')
plt.show()
