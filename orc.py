import pandas as pd
import os
import json

from models.orc_ihx import multiprocess_orc, serial_orc

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Define scenarios
scenarios = [
    "R152a_130_70",  # works (08.01.)   -   higher (140) does NOT work
    "R152a_120_70",  # works (08.01.)
    "R152a_110_70",  # works (08.01.)
    "R134a_95_70",   # works (08.01.)   -   higher (75 and 80) does NOT work
    "R134a_95_65",   # works (19.12.)
    "R134a_95_60",   # works (08.01.)   -   lower (55) does NOT work
]


def get_config_paths(scenario):
    """3
    Generates configuration paths for the given scenario.

    Args:
        scenario (str): Scenario name, e.g., "R152a_140_70".

    Returns:
        dict: Dictionary with paths for "base", "unavoid", and "outputs".
    """
    base_path = f"inputs/orc_ihx_{scenario}.json"
    unavoid_path = f"inputs/orc_ihx_{scenario}_unavoid.json"
    output_path = f"outputs/orc_ihx_{scenario}/"
    return {
        'base': base_path,
        'unavoid': unavoid_path,
        'outputs': output_path
    }


def read_pressure_from_json(json_file):
    """
    Reads the pressure value 'p33' from the JSON file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        float: Pressure value 'p33' from the JSON file.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data['starting_values']['pressures']['p33']


multi = True  # True: multiprocess, False: sequential computation

if __name__ == '__main__':
    # Track results
    successful_scenarios = []
    failed_scenarios = []
    missing_files_scenarios = []

    # Process each scenario
    for scenario in scenarios:
        print(f"\nProcessing scenario: {scenario}")
        try:
            # Get config paths for the current scenario
            config_paths = get_config_paths(scenario)

            # Verify that the input files exist
            if not os.path.exists(config_paths['base']):
                print(f"Error: Base configuration file not found for {scenario}")
                missing_files_scenarios.append((scenario, "missing base configuration file"))
                continue
            if not os.path.exists(config_paths['unavoid']):
                print(f"Error: Unavoid configuration file not found for {scenario}")
                missing_files_scenarios.append((scenario, "missing unavoid configuration file"))
                continue

            # Read p33 from the "base" JSON file
            pressure_p33 = read_pressure_from_json(config_paths['base'])

            # Create output directories if not present
            os.makedirs(config_paths['outputs'], exist_ok=True)
            os.makedirs(os.path.join(config_paths['outputs'], 'streams'), exist_ok=True)
            os.makedirs(os.path.join(config_paths['outputs'], 'comps'), exist_ok=True)
            os.makedirs(os.path.join(config_paths['outputs'], 'diagrams'), exist_ok=True)

            # Run the simulation
            if multi:
                multiprocess_orc(config_paths, pressure_p33)
            else:
                serial_orc(config_paths, pressure_p33)

            successful_scenarios.append(scenario)

        except Exception as e:
            failed_scenarios.append((scenario, str(e)))

    # Print summary
    print("\n" + "=" * 50)
    print("EXECUTION SUMMARY")
    print("=" * 50)

    print(f"\nTotal scenarios: {len(scenarios)}")
    print(f"Successful: {len(successful_scenarios)}")
    print(f"Failed: {len(failed_scenarios)}")
    print(f"Missing files: {len(missing_files_scenarios)}")

    if successful_scenarios:
        print("\nSuccessful scenarios:")
        for scenario in successful_scenarios:
            print(f"✓ {scenario}")

    if failed_scenarios:
        print("\nFailed scenarios:")
        for scenario, error in failed_scenarios:
            print(f"✗ {scenario}: {error}")

    if missing_files_scenarios:
        print("\nScenarios with missing files:")
        for scenario, file_type in missing_files_scenarios:
            print(f"! {scenario}: {file_type}")