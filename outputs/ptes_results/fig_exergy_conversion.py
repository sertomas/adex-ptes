import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.rc('font', family='Arial', size=18)

scenarios = [
    "140_70",
    "130_70",
    "120_70",
    "95_70",
    "95_60",
    "95_50"
]

# Load the data
for scenario in scenarios:
    hp_path =  f"../hp_ihx_R1336MZZZ_{scenario}/comps/hp_comps_all_real.csv"
    orc_path = f"../orc_ihx_R152a_{scenario}/comps/orc_comps_all_real.csv"
    if os.path.exists(hp_path) is False and os.path.exists(orc_path) is False:
        hp_path = f"../hp_ihx_R245fa_{scenario}/comps/hp_comps_all_real.csv"
        orc_path = f"../orc_ihx_R134a_{scenario}/comps/orc_comps_all_real.csv"

    df_hp_comps_base = pd.read_csv(hp_path, index_col=0)
    df_orc_comps_base = pd.read_csv(orc_path, index_col=0)

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
    labels_hp = ['COMP', 'COND1', 'IHX1', 'VAL', 'EVA1']
    labels_orc = ['PUMP', 'EVA2', 'IHX2', 'EXP', 'COND2']

    fig, ax = plt.subplots()
    fig.set_size_inches(14.5, 3.5)

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

    # Adding labels inside the bars for HP components
    left_position_hp = ep_reversed[3]
    for i, value in enumerate(ed_hp):
        if labels_hp[i] in ['COND1', 'COMP']:
            ax.text(x=left_position_hp + value / 2,
                    y=ax.patches[len(ep_reversed) + i].get_y() + ax.patches[len(ep_reversed) + i].get_height() / 2,
                    s=f"{value:.1f}",
                    ha='center',
                    va='center',
                    color='black')
        left_position_hp += value

    # Adding labels inside the bars for ORC components
    left_position_orc = ep_reversed[1] + sum(ed_hp)
    orc_index = len(ep_reversed) + len(ed_hp)  # Index offset for ORC bars

    for i, value in enumerate(ed_orc):
        if labels_orc[i] == 'EVA2':
            bar = ax.patches[orc_index + i]  # Get the correct ORC bar
            ax.text(x=bar.get_x() + bar.get_width() / 2,  # Position text at the center of the bar's width
                    y=bar.get_y() + bar.get_height() / 2,  # Center text vertically in the bar
                    s=f"{value:.1f}",  # Use the corresponding value for the text
                    ha='center',  # Center text horizontally
                    va='center',  # Center text vertically
                    color='black')
        left_position_orc += value

    # Creating custom legend
    legend_patches = [mpatches.Patch(color=colors_hp[i], label=labels_hp[i]) for i in range(len(labels_hp))] + \
                     [mpatches.Patch(color=colors_orc[i], label=labels_orc[i]) for i in range(len(labels_orc))]

    # Displaying the legend
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 0.9), ncol=2, frameon=False)

    # Add text boxes for exergetic efficiency of HP and ORC
    eff_hp = round(df_hp_comps_base.loc['tot', 'epsilon'] * 100, 1)
    eff_orc = round(df_orc_comps_base.loc['tot', 'epsilon'] * 100, 1)
    ax.text(1.09, 0.92, "HP", transform=ax.transAxes, fontsize=18, ha='left', va='center')
    ax.text(1.30, 0.92, "ORC", transform=ax.transAxes, fontsize=18, ha='left', va='center')
    ax.text(1.04, 0.05, f"$\\varepsilon =$ {eff_hp} %", transform=ax.transAxes, fontsize=18, ha='left', va='center')
    ax.text(1.26, 0.05, f"$\\varepsilon =$ {eff_orc} %", transform=ax.transAxes, fontsize=18, ha='left', va='center')

    # Setting the x-axis label and limits
    label_text = "Exergetic efficiency of the overall system $\\varepsilon = 1 - y_\\mathrm{D} - y_\\mathrm{L}$ [%]"
    ax.set_xlabel(label_text)
    ax.set_xlim(left=0)

    # Adjust subplot parameters to give more space for the legend
    plt.subplots_adjust(left=0.07)
    plt.subplots_adjust(right=0.71)
    plt.subplots_adjust(top=0.95)
    plt.subplots_adjust(bottom=0.22)

    plt.savefig(f'exergy_conversion_{scenario}.pdf')
    plt.show()
