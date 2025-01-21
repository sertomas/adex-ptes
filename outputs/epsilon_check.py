import os
import glob
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
root_dir = ""  # Look in the current working directory

scenarios_orc = [
    "R152a_130_70",
    "R152a_120_70",
    "R152a_110_70",
    "R134a_95_70",
    "R134a_95_65",
    "R134a_95_60",
]
scenarios_hp = [
    "R1336MZZZ_130_70",
    "R1336MZZZ_120_70",
    "R1336MZZZ_110_70",
    "R245fa_95_70",
    "R245fa_95_65",
    "R245fa_95_60",
]

ORC_PREFIX = "orc_ihx"
HP_PREFIX = "hp_ihx"

reference_token = "all_real"
low_bound, high_bound = 0.999, 1.001


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_comps_folder(cycle_prefix: str, scenario: str, root: str) -> str:
    return os.path.join(root, f"{cycle_prefix}_{scenario}", "comps")


def load_epsilon_series(csv_path: str) -> pd.Series:
    if not os.path.isfile(csv_path):
        return pd.Series(dtype=float)
    df = pd.read_csv(csv_path, index_col=0)
    if "epsilon" not in df.columns:
        print(f"[WARNING] 'epsilon' column not found in {csv_path}")
        return pd.Series(dtype=float)
    return df["epsilon"]


def compare_epsilon_ratios_and_diffs(cycle_prefix: str, scenario: str, root: str,
                                     aggregator_ratios: dict, aggregator_diffs: dict):
    comps_dir = get_comps_folder(cycle_prefix, scenario, root)
    if not os.path.isdir(comps_dir):
        print(f"[WARNING] comps folder not found for {cycle_prefix}_{scenario}")
        return

    pattern = os.path.join(comps_dir, "*.csv")
    all_csv_files = glob.glob(pattern)

    # Identify the reference CSV file.
    ref_csv = None
    for fpath in all_csv_files:
        if reference_token in os.path.basename(fpath):
            ref_csv = fpath
            break

    if ref_csv is None:
        print(f"[WARNING] No reference file containing '{reference_token}' found in {comps_dir}")
        return

    eps_ref = load_epsilon_series(ref_csv)
    if eps_ref.empty:
        print(f"[WARNING] Reference epsilon series is empty for {ref_csv}. Skipping.")
        return

    for csv_path in all_csv_files:
        basename = os.path.basename(csv_path).lower()
        # Skip the reference file and any file containing "unavoid"
        if csv_path == ref_csv or "unavoid" in basename:
            continue

        eps_case = load_epsilon_series(csv_path)
        if eps_case.empty:
            continue

        all_components = set(eps_case.index).union(set(eps_ref.index))

        for comp in sorted(all_components):
            # Skip the "tot" component
            if comp.lower() == "tot":
                continue
            if comp not in eps_ref or comp not in eps_case:
                continue

            eps_c = eps_case[comp]
            eps_r = eps_ref[comp]
            if pd.isna(eps_c) or pd.isna(eps_r) or eps_r == 0:
                continue

            # Skip if eps_case is close to 1
            if low_bound <= eps_c <= high_bound:
                continue

            ratio = eps_c / eps_r
            # Only record ratios outside tolerance
            if not (low_bound <= ratio <= high_bound):
                aggregator_ratios.setdefault(comp, []).append(ratio)
                aggregator_diffs.setdefault(comp, []).append(eps_c - eps_r)


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ratios_orc = {}
    diffs_orc = {}
    ratios_hp = {}
    diffs_hp = {}

    print("Processing ORC scenarios...")
    for scenario in scenarios_orc:
        compare_epsilon_ratios_and_diffs(ORC_PREFIX, scenario, root_dir, ratios_orc, diffs_orc)

    print("Processing HP scenarios...")
    for scenario in scenarios_hp:
        compare_epsilon_ratios_and_diffs(HP_PREFIX, scenario, root_dir, ratios_hp, diffs_hp)


    def print_component_statistics(ratios: dict, diffs: dict, cycle_name: str):
        print(f"\n=== {cycle_name} Component Statistics ===")
        components = set(ratios.keys()).union(diffs.keys())
        if not components:
            print("No data collected.")
            return
        for comp in sorted(components):
            comp_ratios = ratios.get(comp, [])
            comp_diffs = diffs.get(comp, [])
            if comp_ratios and comp_diffs:
                min_ratio = min(comp_ratios)
                max_ratio = max(comp_ratios)
                min_diff = min(comp_diffs)
                max_diff = max(comp_diffs)
                print(f"Component={comp:10s}  min_ratio={min_ratio:8.4f}  max_ratio={max_ratio:8.4f}  "
                      f"min_diff={min_diff:8.4f}  max_diff={max_diff:8.4f}")
            else:
                print(f"Component={comp:10s}  Insufficient data.")


    print_component_statistics(ratios_orc, diffs_orc, "ORC")
    print_component_statistics(ratios_hp, diffs_hp, "HP")
