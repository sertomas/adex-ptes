import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

plt.rc('font', family='Arial', size=18)

# Define the HP and ORC scenarios with corresponding fluids
hp_scenarios = {
    "R1336MZZZ": ["140_70", "130_70", "120_70"],
    "R245fa": ["95_70", "95_60", "95_50"]
}

orc_scenarios = {
    "R152a": ["140_70", "130_70", "120_70"],
    "R134a": ["95_70", "95_60", "95_50"]
}

# Base output directory
output_dir = "../"

# Labels and colors
labels_hp = ['COMP', 'COND1', 'IHX1', 'VAL', 'EVA1']
labels_orc = ['PUMP', 'EVA2', 'IHX2', 'EXP', 'COND2']
colors_hp = ['#ffcccc', '#ff9999', '#ff6666', '#cc0000', '#990000']
colors_orc = ['#ccccff', '#9999ff', '#6666ff', '#3333ff', '#0000ff']

# Generate plots for each HP and ORC combination
for hp_fluid, hp_cases in hp_scenarios.items():
    for hp_case in hp_cases:
        # Match the ORC fluid based on the HP case
        orc_fluid = "R152a" if "140_70" in hp_case else "R134a"
        orc_case = hp_case

        # Paths for HP and ORC results
        hp_path = os.path.join(output_dir, f"hp_ihx_{hp_fluid}_{hp_case}", "comps", "hp_comps_all_real.csv")
        orc_path = os.path.join(output_dir, f"orc_ihx_{orc_fluid}_{orc_case}", "comps", "orc_comps_all_real.csv")

        # Check if files exist
        if not os.path.exists(hp_path) or not os.path.exists(orc_path):
            print(f"Files missing for HP: {hp_fluid}_{hp_case}, ORC: {orc_fluid}_{orc_case}. Skipping...")
            continue

        # Load data
        df_hp_comps_base = pd.read_csv(hp_path, index_col=0)
        df_orc_comps_base = pd.read_csv(orc_path, index_col=0)

        # Calculate EP and ED breakdowns
        ep = [
            100,
            df_hp_comps_base.loc['tot', 'epsilon'] * 100,
            df_hp_comps_base.loc['tot', 'epsilon'] * 100,
            df_hp_comps_base.loc['tot', 'epsilon'] * df_orc_comps_base.loc['tot', 'epsilon'] * 100,
            df_hp_comps_base.loc['tot', 'epsilon'] * df_orc_comps_base.loc['tot', 'epsilon'] * 100
        ]

        ed_hp = [
            df_hp_comps_base.loc['comp', 'ED [kW]'] / df_hp_comps_base.loc['tot', 'EF [kW]'] * 100,
            df_hp_comps_base.loc['cond', 'ED [kW]'] / df_hp_comps_base.loc['tot', 'EF [kW]'] * 100,
            df_hp_comps_base.loc['ihx', 'ED [kW]'] / df_hp_comps_base.loc['tot', 'EF [kW]'] * 100,
            df_hp_comps_base.loc['val', 'ED [kW]'] / df_hp_comps_base.loc['tot', 'EF [kW]'] * 100,
            df_hp_comps_base.loc['eva', 'ED [kW]'] / df_hp_comps_base.loc['tot', 'EF [kW]'] * 100
        ]

        ed_orc = [
            df_orc_comps_base.loc['pump', 'ED [kW]'] / df_orc_comps_base.loc['tot', 'EF [kW]'] * 100 * df_hp_comps_base.loc['tot', 'epsilon'],
            df_orc_comps_base.loc['eva', 'ED [kW]'] / df_orc_comps_base.loc['tot', 'EF [kW]'] * 100 * df_hp_comps_base.loc['tot', 'epsilon'],
            df_orc_comps_base.loc['ihx', 'ED [kW]'] / df_orc_comps_base.loc['tot', 'EF [kW]'] * 100 * df_hp_comps_base.loc['tot', 'epsilon'],
            df_orc_comps_base.loc['exp', 'ED [kW]'] / df_orc_comps_base.loc['tot', 'EF [kW]'] * 100 * df_hp_comps_base.loc['tot', 'epsilon'],
            df_orc_comps_base.loc['cond', 'ED [kW]'] / df_orc_comps_base.loc['tot', 'EF [kW]'] * 100 * df_hp_comps_base.loc['tot', 'epsilon']
        ]

        # Plot
        fig, ax = plt.subplots()
        fig.set_size_inches(14.5, 3.5)

        groups = ['Output', 'ORC', 'TES', 'HP', 'Input']
        ep_reversed = ep[::-1]
        left_for_ed_hp = ep_reversed.copy()
        left_for_ed_orc = ep_reversed.copy()
        left_for_ed_orc[1] += sum(ed_hp)

        # Plot EP
        ax.barh(groups, ep_reversed, color="lightgrey", edgecolor="black", linewidth=1, height=0.9)

        # Plot HP components
        for i, value in enumerate(ed_hp):
            ax.barh(groups[3], value, left=left_for_ed_hp[3], edgecolor="black", linewidth=1, color=colors_hp[i], height=0.9)
            left_for_ed_hp[3] += value

        # Plot ORC components
        for i, value in enumerate(ed_orc):
            ax.barh(groups[1], value, left=left_for_ed_orc[1], edgecolor="black", linewidth=1, color=colors_orc[i], height=0.9)
            left_for_ed_orc[1] += value

        # Add labels and legend
        ax.set_xlabel("Exergetic efficiency of the overall system $\\varepsilon = 1 - y_\\mathrm{D} - y_\\mathrm{L}$ [%]")
        ax.set_xlim(left=0)

        legend_patches = [mpatches.Patch(color=colors_hp[i], label=labels_hp[i]) for i in range(len(labels_hp))] + \
                         [mpatches.Patch(color=colors_orc[i], label=labels_orc[i]) for i in range(len(labels_orc))]
        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 0.9), ncol=2, frameon=False)

        plt.subplots_adjust(left=0.07, right=0.71, top=0.95, bottom=0.22)

        # Save the plot
        filename = f"exergy_conversion_{hp_fluid}_{hp_case}.pdf"
        plt.savefig(filename)
        print(f"Saved plot: {filename}")
        plt.close()
