import pandas as pd
import matplotlib.pyplot as plt

# HEAT PUMP ------------------------------------------------------------------------------------------------------------

# Load the DataFrame
df_hp_mexo = pd.read_csv('../adex_hp/hp_adex_analysis.csv', index_col=[0, 1])

components = ["comp", "cond", "ihx", "val", "eva"]  # The first-level index values you're interested in
colors = ['#ccccff', '#ccffcc', '#ff6666']  # Colors for endo, mexo, ex_l
labels = [
    r'$\dot{E}^\mathrm{EN}_{\mathrm{D},k}$',  # For ED EN
    r'$\dot{E}^\mathrm{MEXO}_{\mathrm{D},k}$',  # For ED MEXO Positive
    r'$\dot{E}^{\mathrm{EX},kl}_{\mathrm{D},k}$',  # For ED MEXO Negative
    r'$\dot{E}^\mathrm{EXL}$',   # For ED EX l Positive
    r'$\dot{E}^\mathrm{EXL}$'    # For ED EX l Negative
]

ed_endo_hp = []
ed_mexo_hp_positive = []
ed_mexo_hp_negative = []
ed_ex_l_hp_positive = {}  # Initialize as an empty dictionary for storing ED EX l values
ed_ex_l_hp_negative = {}  # Initialize as an empty dictionary for storing ED EX l values

for component in components:
    values_endo = df_hp_mexo.loc[component, "ED EN [kW]"].sum()
    ed_endo_hp.append(values_endo)  # Sum and append directly

    values_mexo = df_hp_mexo.loc[component, "ED MEXO [kW]"].sum()  # Sum first, then check
    if values_mexo < 0:
        ed_mexo_hp_negative.append(values_mexo)  # Append the sum if negative
    elif values_mexo > 0:
        ed_mexo_hp_positive.append(values_mexo)  # Append the sum if positive

    for other_component in components:
        if other_component != component:
            key = (component, other_component)
            value = df_hp_mexo.loc[(component, other_component), "ED EX l [kW]"].sum()
            if value < 0:
                ed_ex_l_hp_negative[key] = value
            else:
                ed_ex_l_hp_positive[key] = value

# Before plotting, create handles for the legend
handles = [
    plt.Rectangle((0, 0), 1, 1, color=color, label=label) for color, label in zip(colors[:3], labels[:3])
]
# Assuming df_hp_mexo is already loaded and components list is defined

# Initialize arrays for plotting
totals_positive = []
totals_negative = []

# Calculate totals for each component
for component in components:
    # Start with endo value (assumed always positive for simplicity)
    total_positive = [df_hp_mexo.loc[component, "ED EN [kW]"].sum()]
    total_negative = [0]  # Placeholder for symmetry in plotting

    # Add mexo positive and negative values
    mexo_values = df_hp_mexo.loc[component, "ED MEXO [kW]"].sum()
    if mexo_values > 0:
        total_positive.append(mexo_values)
        total_negative.append(0)  # Placeholder
    else:
        total_positive.append(0)  # Placeholder
        total_negative.append(mexo_values)

    # Aggregate positive and negative ED EX l values for this component
    positive_ex_l = sum(
        value for (comp1, comp2), value in ed_ex_l_hp_positive.items() if comp1 == component)
    negative_ex_l = sum(
        value for (comp1, comp2), value in ed_ex_l_hp_negative.items() if comp1 == component)

    total_positive.append(positive_ex_l)
    total_negative.append(negative_ex_l)

    totals_positive.append(total_positive)
    totals_negative.append(total_negative)

# Change component names to uppercase for plotting
components_uppercase = [component.upper() for component in components]

# Plotting
fig, ax = plt.subplots()
fig.set_size_inches(12, 5.3)

# Plot positive values
for i, totals in enumerate(totals_positive):
    left = 0
    for j, total in enumerate(totals):
        if total > 0:  # Only plot positive totals
            ax.barh(components_uppercase[i], total, left=left, color=colors[j], edgecolor="black", linewidth=1, height=0.9)
            left += total

# Plot negative values
for i, totals in enumerate(totals_negative):
    left = 0
    for j, total in enumerate(totals):
        if total < 0:  # Only plot negative totals
            ax.barh(components_uppercase[i], total, left=left, color=colors[j], edgecolor="black", linewidth=1, height=0.9)
            left += total

# Add the legend to the plot
ax.legend(handles=handles, fontsize='large')
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Exergy destruction [kW]')
ax.set_xlim(-25, 175)
plt.tight_layout()
plt.savefig('adex_hp.png')
plt.show()


# ORC ------------------------------------------------------------------------------------------------------------------

# Load the DataFrame
df_orc_mexo = pd.read_csv('../adex_orc/orc_adex_analysis.csv', index_col=[0, 1])

components = ["pump", "eva", "ihx", "exp", "cond"]  # The first-level index values you're interested in
labels = [
    r'$\dot{E}^\mathrm{EN}_{\mathrm{D},k}$',  # For ED EN
    r'$\dot{E}^\mathrm{MEXO}_{\mathrm{D},k}$',  # For ED MEXO Positive
    r'$\dot{E}^{\mathrm{EX},kl}_{\mathrm{D},k}$',  # For ED MEXO Negative
    r'$\dot{E}^\mathrm{EXL}$',   # For ED EX l Positive
    r'$\dot{E}^\mathrm{EXL}$'    # For ED EX l Negative
]

ed_endo_orc = []
ed_mexo_orc_positive = []
ed_mexo_orc_negative = []
ed_ex_l_orc_positive = {}  # Initialize as an empty dictionary for storing ED EX l values
ed_ex_l_orc_negative = {}  # Initialize as an empty dictionary for storing ED EX l values

for component in components:
    values_endo = df_orc_mexo.loc[component, "ED EN [kW]"].sum()
    ed_endo_orc.append(values_endo)  # Sum and append directly

    values_mexo = df_orc_mexo.loc[component, "ED MEXO [kW]"].sum()  # Sum first, then check
    if values_mexo < 0:
        ed_mexo_orc_negative.append(values_mexo)  # Append the sum if negative
    elif values_mexo > 0:
        ed_mexo_orc_positive.append(values_mexo)  # Append the sum if positive

    for other_component in components:
        if other_component != component:
            key = (component, other_component)
            value = df_orc_mexo.loc[(component, other_component), "ED EX l [kW]"].sum()
            if value < 0:
                ed_ex_l_orc_negative[key] = value
            else:
                ed_ex_l_orc_positive[key] = value

# Before plotting, create handles for the legend
handles = [
    plt.Rectangle((0,0),1,1, color=color, label=label) for color, label in zip(colors[:3], labels[:3])
]
# Assuming df_orc_mexo is already loaded and components list is defined

# Initialize arrays for plotting
totals_positive = []
totals_negative = []

# Calculate totals for each component
for component in components:
    # Start with endo value (assumed always positive for simplicity)
    total_positive = [df_orc_mexo.loc[component, "ED EN [kW]"].sum()]
    total_negative = [0]  # Placeholder for symmetry in plotting

    # Add mexo positive and negative values
    mexo_values = df_orc_mexo.loc[component, "ED MEXO [kW]"].sum()
    if mexo_values > 0:
        total_positive.append(mexo_values)
        total_negative.append(0)  # Placeholder
    else:
        total_positive.append(0)  # Placeholder
        total_negative.append(mexo_values)

    # Aggregate positive and negative ED EX l values for this component
    positive_ex_l = sum(
        value for (comp1, comp2), value in ed_ex_l_orc_positive.items() if comp1 == component)
    negative_ex_l = sum(
        value for (comp1, comp2), value in ed_ex_l_orc_negative.items() if comp1 == component)

    total_positive.append(positive_ex_l)
    total_negative.append(negative_ex_l)

    totals_positive.append(total_positive)
    totals_negative.append(total_negative)

# Change component names to uppercase for plotting
components_uppercase = [component.upper() for component in components]

# Plotting
fig, ax = plt.subplots()
fig.set_size_inches(12, 5.3)

# Plot positive values
for i, totals in enumerate(totals_positive):
    left = 0
    for j, total in enumerate(totals):
        if total > 0:  # Only plot positive totals
            ax.barh(components_uppercase[i], total, left=left, color=colors[j], edgecolor="black", linewidth=1, height=0.9)
            left += total

# Plot negative values
for i, totals in enumerate(totals_negative):
    left = 0
    for j, total in enumerate(totals):
        if total < 0:  # Only plot negative totals
            ax.barh(components_uppercase[i], total, left=left, color=colors[j], edgecolor="black", linewidth=1, height=0.9)
            left += total

# Add the legend to the plot
ax.legend(handles=handles, fontsize='large')
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Exergy destruction [kW]')
ax.set_xlim(-25, 175)
plt.tight_layout()
plt.savefig('adex_orc.png')
plt.show()
