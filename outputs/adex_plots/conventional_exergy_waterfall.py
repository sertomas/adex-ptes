import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Set global font to Arial, size 18
plt.rc('font', family='Arial', size=18)

###############################################################################
# 1) SPECIFY SCENARIOS AND LABELS
###############################################################################
# The order you want: top row => (130_70, 95_75), second => (120_70, 95_65),
# third => (110_70, 95_60)
scenario_order = [
    "130_70", "95_70",
    "120_70", "95_65",
    "110_70", "95_60"
]

# Maps each scenario to a nice title for the subplot
scenario_titles = {
    "130_70": "HT-PTES (130-70)",
    "120_70": "HT-PTES (120-70)",
    "110_70": "HT-PTES (110-70)",
    "95_70": "LT-PTES (95-70)",
    "95_65": "LT-PTES (95-65)",
    "95_60": "LT-PTES (95-60)"
}

# Colors and labels for HP and ORC
labels_hp = ['COMP', 'COND1', 'IHX1', 'VAL', 'EVA1']
colors_hp = ['#ffcccc', '#ff9999', '#ff6666', '#cc0000', '#990000']

labels_orc = ['PUMP', 'EVA2', 'IHX2', 'EXP', 'COND2']
colors_orc = ['#ccccff', '#9999ff', '#6666ff', '#3333ff', '#0000ff']

# Combined legend patches (one legend at the end)
legend_patches = [
    mpatches.Patch(color=colors_hp[i], label=labels_hp[i]) for i in range(len(labels_hp))
] + [
    mpatches.Patch(color=colors_orc[i], label=labels_orc[i]) for i in range(len(labels_orc))
]


###############################################################################
# 2) SETUP FIGURE WITH 3 ROWS × 2 COLUMNS
###############################################################################
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 11))
# axes is a 3×2 array, so axes[row, col] gives each subplot

###############################################################################
# 3) LOOP OVER SCENARIOS, ONE PER SUBPLOT
###############################################################################
for idx, scenario in enumerate(scenario_order):
    # Determine row, col for this scenario
    row = idx // 2  # integer division
    col = idx % 2
    ax = axes[row, col]

    # --------------------------------------------------
    # LOAD THE DATA
    # --------------------------------------------------
    hp_path = f"../hp_ihx_R1336MZZZ_{scenario}/comps/hp_comps_all_real.csv"
    orc_path = f"../orc_ihx_R152a_{scenario}/comps/orc_comps_all_real.csv"

    # Fallback if above paths don't exist
    if not os.path.exists(hp_path) and not os.path.exists(orc_path):
        hp_path = f"../hp_ihx_R245fa_{scenario}/comps/hp_comps_all_real.csv"
        orc_path = f"../orc_ihx_R134a_{scenario}/comps/orc_comps_all_real.csv"

    df_hp = pd.read_csv(hp_path, index_col=0)
    df_orc = pd.read_csv(orc_path, index_col=0)

    # --------------------------------------------------
    # PREPARE EP AND ED
    # --------------------------------------------------
    # 'ep' in order: Input, HP, TES, ORC, Output
    ep = [
        100,
        df_hp.loc['tot', 'epsilon'] * 100,  # HP
        df_hp.loc['tot', 'epsilon'] * 100,  # TES
        df_hp.loc['tot', 'epsilon'] * df_orc.loc['tot', 'epsilon'] * 100,  # ORC
        df_hp.loc['tot', 'epsilon'] * df_orc.loc['tot', 'epsilon'] * 100   # Output
    ]

    # ED breakdown for HP
    ed_hp = [
        df_hp.loc['comp', 'ED [kW]'] / df_hp.loc['tot', 'EF [kW]'] * 100,
        df_hp.loc['cond', 'ED [kW]'] / df_hp.loc['tot', 'EF [kW]'] * 100,
        df_hp.loc['ihx',  'ED [kW]'] / df_hp.loc['tot', 'EF [kW]'] * 100,
        df_hp.loc['val',  'ED [kW]'] / df_hp.loc['tot', 'EF [kW]'] * 100,
        df_hp.loc['eva',  'ED [kW]'] / df_hp.loc['tot', 'EF [kW]'] * 100
    ]

    # ED breakdown for ORC (multiplied by HP's epsilon so they share the same horizontal scale)
    ed_orc = [
        df_orc.loc['pump', 'ED [kW]'] / df_orc.loc['tot', 'EF [kW]'] * 100 * df_hp.loc['tot', 'epsilon'],
        df_orc.loc['eva',  'ED [kW]'] / df_orc.loc['tot', 'EF [kW]'] * 100 * df_hp.loc['tot', 'epsilon'],
        df_orc.loc['ihx',  'ED [kW]'] / df_orc.loc['tot', 'EF [kW]'] * 100 * df_hp.loc['tot', 'epsilon'],
        df_orc.loc['exp',  'ED [kW]'] / df_orc.loc['tot', 'EF [kW]'] * 100 * df_hp.loc['tot', 'epsilon'],
        df_orc.loc['cond', 'ED [kW]'] / df_orc.loc['tot', 'EF [kW]'] * 100 * df_hp.loc['tot', 'epsilon']
    ]

    # Reverse the group names for barh
    groups = ['Output', 'ORC', 'TES', 'HP', 'Input']
    ep_reversed = ep[::-1]

    # --------------------------------------------------
    # PLOT THE BASE GREY BARS
    # --------------------------------------------------
    bar_thickness = 0.85
    base_bars = ax.barh(
        groups,
        ep_reversed,
        color="lightgrey",
        edgecolor="black",
        linewidth=1,
        height=bar_thickness
    )

    # Keep track of where we left off for stacking
    left_for_hp = ep_reversed.copy()
    left_for_orc = ep_reversed.copy()

    # HP = index 3, ORC = index 1 in `groups`
    for i_hp, val_hp in enumerate(ed_hp):
        ax.barh(
            groups[3],
            val_hp,
            left=left_for_hp[3],
            color=colors_hp[i_hp],
            edgecolor="black",
            linewidth=1,
            height=bar_thickness
        )
        left_for_hp[3] += val_hp

    for i_orc, val_orc in enumerate(ed_orc):
        ax.barh(
            groups[1],
            val_orc,
            left=left_for_orc[1],
            color=colors_orc[i_orc],
            edgecolor="black",
            linewidth=1,
            height=bar_thickness
        )
        left_for_orc[1] += val_orc

    # --------------------------------------------------
    # ADD TEXT INSIDE THE BARS
    # --------------------------------------------------
    # (A) Base grey bars
    for i_bar, bar in enumerate(ax.patches[:len(ep_reversed)]):
        bar_x = bar.get_x() + bar.get_width()/2
        bar_y = bar.get_y() + bar.get_height()/2
        ax.text(bar_x, bar_y, f"{ep_reversed[i_bar]:.1f}",
                ha="center", va="center", color="black", fontsize=18)

    # (B) HP bars => next len(ed_hp) patches
    hp_bars = ax.patches[len(ep_reversed): len(ep_reversed)+len(ed_hp)]
    for i_hp, bar in enumerate(hp_bars):
        if labels_hp[i_hp] in ['COND1', 'COMP']:
            bar_x = bar.get_x() + bar.get_width()/2
            bar_y = bar.get_y() + bar.get_height()/2
            ax.text(bar_x, bar_y, f"{ed_hp[i_hp]:.1f}",
                    ha='center', va='center', color='black', fontsize=18)

    # (C) ORC bars => next len(ed_orc) patches
    orc_bars = ax.patches[len(ep_reversed)+len(ed_hp) : len(ep_reversed)+len(ed_hp)+len(ed_orc)]
    for i_orc, bar in enumerate(orc_bars):
        if labels_orc[i_orc] == 'EVA2':
            bar_x = bar.get_x() + bar.get_width()/2
            bar_y = bar.get_y() + bar.get_height()/2
            ax.text(bar_x, bar_y, f"{ed_orc[i_orc]:.1f}",
                    ha='center', va='center', color='black', fontsize=18)

    # --------------------------------------------------
    # ADD EXERGETIC EFFICIENCIES & SUBPLOT TITLE
    # --------------------------------------------------
    eff_hp = round(df_hp.loc['tot', 'epsilon']*100, 1)
    eff_orc = round(df_orc.loc['tot', 'epsilon']*100, 1)

    # Use the scenario_titles dictionary to get HT-PTES or LT-PTES naming
    ax.set_title(scenario_titles[scenario], fontsize=18)

    # HP/ORC eff at bottom-right, no background
    ax.text(
        0.98, 0.02,
        f"$\\varepsilon_{{HP}} = {eff_hp:.1f}\\%$\n"
        f"$\\varepsilon_{{ORC}} = {eff_orc:.1f}\\%$",
        transform=ax.transAxes,
        ha='right', va='bottom', fontsize=18
    )

    # X label only on last row (row==2) for clarity
    if row == 2:
        ax.set_xlabel("Exergetic Efficiency [%]", fontsize=18)

    # No y-label for cleanliness
    ax.set_ylabel("")

    ax.set_xlim(left=0, right=102)

###############################################################################
# 4) SINGLE LEGEND AT BOTTOM
###############################################################################
fig.legend(
    handles=legend_patches,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.01),  # adjust as needed
    ncol=5,
    frameon=False,
    fontsize=18
)

# Adjust spacing so subplots + legend don’t overlap
plt.subplots_adjust(
    left=0.07, right=0.97,
    top=0.93, bottom=0.15,
    wspace=0.25, hspace=0.35
)

# Save or display
plt.savefig("conventional_exergy_waterfall.pdf", dpi=300, bbox_inches='tight')
plt.show()
