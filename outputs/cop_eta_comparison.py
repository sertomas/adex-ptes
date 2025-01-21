import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set global font to Arial, size 18
plt.rc('font', family='Arial', size=18)

# Define color schemes for HP and ORC
colors_hp = ['#ffcccc', '#ff9999', '#ff6666', '#cc0000', '#990000']
colors_orc = ['#ccccff', '#9999ff', '#6666ff', '#3333ff', '#0000ff']

# Scenario titles mapping
scenario_titles = {
    "130_70": "HT-PTES (130-70)",
    "120_70": "HT-PTES (120-70)",
    "110_70": "HT-PTES (110-70)",
    "95_70": "LP-PTES (95-70)",
    "95_65": "LP-PTES (95-65)",
    "95_60": "LP-PTES (95-60)"
}

# Example scenarios
scenarios_hp = [
    "R1336MZZZ_130_70",
    "R1336MZZZ_120_70",
    "R1336MZZZ_110_70",
    "R245fa_95_70",
    "R245fa_95_65",
    "R245fa_95_60"
]

scenarios_orc = [
    "R152a_130_70",
    "R152a_120_70",
    "R152a_110_70",
    "R134a_95_70",
    "R134a_95_65",
    "R134a_95_60",
]


def collect_optimal_all_real_cops(root_outputs: str, scenarios: list) -> pd.Series:
    """
    Reads hp_cop.csv from each scenario's folder in `root_outputs`,
    extracts the 'cop' entry where label == 'optimal_all_real',
    and returns these values as a pandas Series.

    Parameters
    ----------
    root_outputs : str
        The directory under which each scenario's folder is located.
        E.g. if your structure is:
          outputs/
            hp_ihx_R1336MZZZ_120_70/
              hp_cop.csv
        then root_outputs = 'outputs'
    scenarios : list of str
        List of scenario folder suffixes, e.g. 'R1336MZZZ_120_70'.

    Returns
    -------
    pd.Series
        A Series whose index is each scenario name, and whose value is the
        COP from 'optimal_all_real'.
    """
    data = {}  # We'll build a dictionary {scenario -> cop}

    for scenario in scenarios:
        scenario_folder = os.path.join(root_outputs, f"hp_ihx_{scenario}")
        csv_path = os.path.join(scenario_folder, "hp_cop.csv")

        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} does not exist. Skipping.")
            continue

        df_cop = pd.read_csv(csv_path)

        # Filter for label == "optimal_all_real"
        mask = df_cop['label'] == "optimal_all_real"
        if not mask.any():
            print(f"Warning: 'optimal_all_real' not found in {csv_path}. Skipping.")
            continue

        # If multiple rows match, we'll just take the first
        cop_value = df_cop.loc[mask, 'cop'].iloc[0]
        data[scenario] = cop_value

    # Convert dictionary into a Series
    # The index is scenario, the value is COP
    s = pd.Series(data, name='optimal_all_real_COP')
    return s


def collect_optimal_all_real_eta(root_outputs: str, scenarios: list) -> pd.Series:
    """
    Reads orc_eta.csv from each scenario's folder under root_outputs,
    extracts the 'eta' entry where label == 'optimal_all_real',
    and returns these values as a pandas Series.
    """
    import os
    import pandas as pd

    data = {}  # dictionary to store scenario -> eta

    for scenario in scenarios:
        scenario_folder = os.path.join(root_outputs, f"orc_ihx_{scenario}")
        csv_path = os.path.join(scenario_folder, "orc_eta.csv")

        if not os.path.isfile(csv_path):
            print(f"Warning: {csv_path} does not exist. Skipping {scenario}.")
            continue

        df_eta = pd.read_csv(csv_path)

        # Filter rows for label == "optimal_all_real"
        mask = df_eta['label'] == "optimal_all_real"
        if not mask.any():
            print(f"Warning: 'optimal_all_real' not found in {csv_path}. Skipping {scenario}.")
            continue

        # Take the first matching eta value
        eta_value = df_eta.loc[mask, 'eta'].iloc[0]
        data[scenario] = eta_value

    # Convert dictionary to a pandas Series
    return pd.Series(data, name='optimal_all_real_eta')


def collect_exergy_efficiency_hp(root_outputs: str, scenarios: list) -> pd.Series:
    """
    Reads 'hp_comps_all_real.csv' from each HP scenario folder,
    extracts the 'epsilon' value from the 'tot' row,
    and returns these as a pandas Series.
    """
    data = {}
    for scenario in scenarios:
        comps_path = os.path.join(root_outputs, f"hp_ihx_{scenario}", "comps", "hp_comps_all_real.csv")
        if not os.path.isfile(comps_path):
            print(f"Warning: {comps_path} does not exist. Skipping.")
            continue
        df = pd.read_csv(comps_path, index_col=0)
        if "tot" not in df.index or "epsilon" not in df.columns:
            print(f"Warning: Missing 'tot' row or 'epsilon' column in {comps_path}. Skipping.")
            continue
        data[scenario] = df.loc["tot", "epsilon"]
    return pd.Series(data, name='exergy_efficiency_hp')


def collect_exergy_efficiency_orc(root_outputs: str, scenarios: list) -> pd.Series:
    """
    Reads 'orc_comps_all_real.csv' from each ORC scenario folder,
    extracts the 'epsilon' value from the 'tot' row,
    and returns these as a pandas Series.
    """
    data = {}
    for scenario in scenarios:
        comps_path = os.path.join(root_outputs, f"orc_ihx_{scenario}", "comps", "orc_comps_all_real.csv")
        if not os.path.isfile(comps_path):
            print(f"Warning: {comps_path} does not exist. Skipping.")
            continue
        df = pd.read_csv(comps_path, index_col=0)
        if "tot" not in df.index or "epsilon" not in df.columns:
            print(f"Warning: Missing 'tot' row or 'epsilon' column in {comps_path}. Skipping.")
            continue
        data[scenario] = df.loc["tot", "epsilon"]
    return pd.Series(data, name='exergy_efficiency_orc')


def extract_key(scenario):
    """Extracts the key (e.g., '130_70', '95_70') from a given scenario string."""
    parts = scenario.split('_')
    if len(parts) >= 3:
        # Return the last two segments as the key
        return f"{parts[-2]}_{parts[-1]}"
    return scenario

if __name__ == "__main__":
    root_dir = ""  # Set root directory if needed

    # Collect thermal and exergetic performance data
    optimal_cop_series = collect_optimal_all_real_cops(root_dir, scenarios_hp)
    optimal_eta_series = collect_optimal_all_real_eta(root_dir, scenarios_orc)
    exergy_eff_hp = collect_exergy_efficiency_hp(root_dir, scenarios_hp)
    exergy_eff_orc = collect_exergy_efficiency_orc(root_dir, scenarios_orc)

    # --- Combined Figure Setup: Two Subplots Side by Side ---
    width = 0.35  # width of the bars
    fig, (ax_hp, ax_orc) = plt.subplots(ncols=2, figsize=(15, 7))

    # ---------------------- HP Plot (Left Subplot) ----------------------
    ax2_hp = ax_hp.twinx()
    df_hp = pd.DataFrame({
        "COP": optimal_cop_series,
        "Exergy Efficiency": exergy_eff_hp
    })
    scenarios_hp_list = df_hp.index.tolist()
    x_hp = np.arange(len(scenarios_hp_list))

    bars1_hp = ax_hp.bar(x_hp - width/2, df_hp["COP"], width, color=colors_hp[0], label="COP")
    bars2_hp = ax2_hp.bar(x_hp + width/2, df_hp["Exergy Efficiency"], width, color=colors_hp[3], label="Exergy Efficiency")

    ax_hp.set_ylabel("COP", color='black')
    ax2_hp.set_ylabel("Exergy Efficiency (ε)", color='black')
    ax_hp.set_title("HP")

    transformed_labels_hp = [scenario_titles.get(extract_key(s), extract_key(s)) for s in scenarios_hp_list]
    ax_hp.set_xticks(x_hp)
    ax_hp.set_xticklabels(transformed_labels_hp, rotation=45, ha='right')

    ax_hp.grid(axis='y', linestyle='--', alpha=0.7)

    # Restore legend for HP plot
    lines_labels_hp = [ax.get_legend_handles_labels() for ax in [ax_hp, ax2_hp]]
    lines_hp, labels_hp = [sum(lol, []) for lol in zip(*lines_labels_hp)]
    ax_hp.legend(lines_hp, labels_hp, loc='upper left')

    # Increase vertical space by setting y-axis limits for both y-axes on HP plot
    ax_hp.set_ylim(0, 4)         # Set COP range from 0 to 4.5
    ax2_hp.set_ylim(0, 1)        # Adjust as needed for Exergy Efficiency

    # ---------------------- ORC Plot (Right Subplot) ----------------------
    ax2_orc = ax_orc.twinx()
    df_orc = pd.DataFrame({
        "Thermal Efficiency": optimal_eta_series,
        "Exergy Efficiency": exergy_eff_orc
    })
    scenarios_orc_list = df_orc.index.tolist()
    x_orc = np.arange(len(scenarios_orc_list))

    bars1_orc = ax_orc.bar(x_orc - width/2, df_orc["Thermal Efficiency"], width, color=colors_orc[0], label="Thermal Efficiency")
    bars2_orc = ax2_orc.bar(x_orc + width/2, df_orc["Exergy Efficiency"], width, color=colors_orc[3], label="Exergy Efficiency")

    ax_orc.set_ylabel("Thermal Efficiency (η)", color='black')
    ax2_orc.set_ylabel("Exergy Efficiency (ε)", color='black')
    ax_orc.set_title("ORC")

    transformed_labels_orc = [scenario_titles.get(extract_key(s), extract_key(s)) for s in scenarios_orc_list]
    ax_orc.set_xticks(x_orc)
    ax_orc.set_xticklabels(transformed_labels_orc, rotation=45, ha='right')

    ax_orc.grid(axis='y', linestyle='--', alpha=0.7)

    # Restore legend for ORC plot
    lines_labels_orc = [ax.get_legend_handles_labels() for ax in [ax_orc, ax2_orc]]
    lines_orc, labels_orc = [sum(lol, []) for lol in zip(*lines_labels_orc)]
    ax_orc.legend(lines_orc, labels_orc, loc='upper left')

    # Increase vertical space by setting y-axis limits for both y-axes on ORC plot
    ax_orc.set_ylim(0, 0.25)   # Set Thermal Efficiency range, adjust as necessary
    ax2_orc.set_ylim(0, 1)  # Adjust as needed for Exergy Efficiency

    # Increase space at the bottom to avoid overlap of x-axis labels
    plt.subplots_adjust(bottom=0.25)

    plt.tight_layout()
    plt.savefig("exergy-energy-comparison.pdf", dpi=300, bbox_inches='tight')
    plt.show()