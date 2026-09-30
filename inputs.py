import pandas as pd

# ============================================================
# Historical model input
# ============================================================
# - Water Data -

# water_data: from Berlin water price
# Historical water series extended to 2025-S2. Berlin water and wastewater tariffs are held constant after the last published tariff change.
WATER_HISTORY = pd.DataFrame({
    "DATE" : pd.date_range(
        start="2008",
        end="2026-01-31", 
        freq="6ME",
        inclusive="left"
    ),
    "WF" : [0.002071, 0.002071, 0.002038, 0.002038, 0.0020325, 0.0020325, 0.002027, 0.002027, 0.002027, 0.002027, 0.002027, 0.002027, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694],
    "WWF" : [0.002567, 0.002567, 0.002543, 0.002543, 0.002504, 0.002504, 0.002464, 0.002464, 0.002464, 0.002464, 0.002464, 0.002464, 0.002464, 0.002464, 0.002307, 0.002307, 0.002303, 0.002303, 0.002303, 0.002303, 0.00221, 0.00221, 0.00221, 0.00221, 0.00221, 0.00221, 0.00221, 0.00221, 0.002155, 0.002155, 0.002155, 0.002155, 0.002155, 0.002155, 0.002155, 0.002155]
})


# - Electricity Data -

# Historical elc. data are from EUROSTAT
# data_elc: elc. data from EUROSTAT used elc. from 50 Mio kWh/a
# Historical electricity series extended through 2025-S2. Forecasts start from the last historical period for continuity.
ELECTRICITY_HISTORY= pd.DataFrame({
    "DATE"    : pd.date_range(
        start="2004",
        end="2026-01-31",
        freq="6ME",
        inclusive="left"
    ),
    "no_tax"  : [0.0764, 0.0793, 0.0840, 0.0877, 0.0949, 0.0963, 0.1003, 0.0832,
                    0.0900, 0.0899, 0.0942, 0.0901, 0.0878, 0.0957, 0.1002, 0.1091,
                    0.1040, 0.1049, 0.1123, 0.1118, 0.1158, 0.1117, 0.1111, 0.1124,
                    0.1021, 0.0965, 0.0972, 0.0887, 0.0860, 0.0879, 0.0904, 0.1093,
                    0.1206, 0.1271, 0.1267, 0.1349, 0.1768, 0.1939, 0.1905, 0.1767,
                    0.1575, 0.1629, 0.1570, 0.1511],
})

# data_nw_elc: elc. data from EUROSTAT used elc. from 50 Mio kWh/a
ELECTRICITY_HISTORY_NO_WAR= pd.DataFrame({
    "DATE"    : pd.date_range(
        start="2004",
        end="2018-07-31",
        freq="6ME",
        inclusive="left"
    ),
    "no_tax"  : [0.0764, 0.0793, 0.0840, 0.0877, 0.0949, 0.0963, 0.1003, 0.0832, \
                    0.0900, 0.0899, 0.0942, 0.0901, 0.0878, 0.0957, 0.1002, 0.1091, \
                    0.1040, 0.1049, 0.1123, 0.1118, 0.1158, 0.1117, 0.1111, 0.1124, \
                    0.1021, 0.0965, 0.0972, 0.0887, 0.0860],
})

# Source metadata for historical inputs
HISTORICAL_INPUT_SOURCES = {
    "electricity": "Eurostat electricity price statistics / nrg_pc_205; active no_tax series extended to 2025-S2.",
    "water": "Berliner Wasserbetriebe published water and wastewater tariffs; active water_data extended to 2025-S2 using current tariff levels.",
}

# Historical uncertainty assumptions for Monte Carlo inputs.
# For these inputs, the Excel sheets provide historical summary statistics.
# The triangular distribution uses min and max as empirical bounds and the mean as mode.
HISTORICAL_METAL_STATS = {
    # "metals": {
        "Steel": {
            "mean": 0.5831,
            "std": 0.1404676025581025,
            "variance": 0.01973114736842105,
            "min": 0.37,
            "max": 0.905,
            "range": 0.535,
            "c": 0.3983177570093457,
        },
        "Nickel": {
            "mean": 18.08900217153637,
            "std": 6.413925515723414,
            "variance": 41.13844052124786,
            "min": 9.595179080988459,
            "max": 37.13584189039519,
            "range": 27.54066280940673,
            "c": 0.30841026410035344,
        },
        "Copper": {
            "mean": 7.274755925478757,
            "std": 1.453860395124267,
            "variance": 2.113710048510888,
            "min": 4.867897429653678,
            "max": 9.946881947016749,
            "range": 5.078984517363071,
            "c": 0.4738857713775197,
        },
        "Titanium": {
            "mean": 11.925,
            "std": 1.018706610312673,
            "variance": 1.037763157894737,
            "min": 10.5,
            "max": 15.0,
            "range": 4.5,
            "c": 0.3166666666666668,
        },
        "Iridium": {
            "mean": 57171.74430158868,
            "std": 56402.09775194612,
            "variance": 3181196630.820086,
            "min": 11217.17283387429,
            "max": 163003.2337735307,
            "range": 151786.0609396564,
            "c": 0.30275883821758803,
        },
        "Platinum": {
            "mean": 31263.020700000005,
            "std": 4861.759954601031,
            "variance": 23636709.856162217,
            "min": 24079.692,
            "max": 39853.952,
            "range": 15774.259999999998,
            "c": 0.45538292763020,
        },
    # },
    # "resources": {
    #     "AWE_S1_resource_cost_20y": {
    #         "mean": 61316741.02431111,
    #         "std": 16435798.17947772,
    #         "variance": 270135461796523.16,
    #         "min": 44855543.53819999,
    #         "max": 98889531.23820001,
    #         "range": 54033987.70000002,
    #         "c": 0.3046452462014221,
    #     },
    #     "AWE_S2_resource_cost_20y": {
    #         "mean": 62305020.91764444,
    #         "std": 16509099.359253697,
    #         "variance": 272550361653710.84,
    #         "min": 45838077.6182,
    #         "max": 100647750.1182,
    #         "range": 54809672.50000001,
    #         "c": 0.3004386369840915,
    #     },
    #     "PEMWE_S1_resource_cost_20y": {
    #         "mean": 52747650.5617215,
    #         "std": 14140943.38226515,
    #         "variance": 199966279740428.53,
    #         "min": 38585459.726880506,
    #         "max": 85074161.12937808,
    #         "range": 46488701.402497575,
    #         "c": 0.3046372647027765,
    #     },
    #     "PEMWE_S2_resource_cost_20y": {
    #         "mean": 53597925.892899275,
    #         "std": 14204014.922320748,
    #         "variance": 201754039913510.47,
    #         "min": 39430791.59683051,
    #         "max": 86586860.26507808,
    #         "range": 47156068.66824757,
    #         "c": 0.30043077585066313,
    #     },
    # },
}


# ============================================================
# Common model input
# ============================================================

LIFETIME_YEARS = 20

INFLATION_RATE = 0.01

EQUITY = 0.25
EQUITY_RETURN = 0.07

DEBT = 0.75
DEBT_INTEREST_RATE = 0.045

OPEX_LABOUR_FRACTION = 0.025

# ============================================================
# Technology-specific inputs
# ============================================================

from input_awe import (
    AWE_CAPACITY_MW,
    # AWE_H2_PER_YEAR,
    AWE_H2_PER_HALF_YEAR,
    AWE_ELECTRICITY_KWH_PER_KG,
    AWE_WATER_KG_PER_KG,
    AWE_KOH_KG_PER_KG,
    AWE_KOH_PRICE_EUR_PER_KG,
    AWE_CONSTRUCTION_LABOUR_FRACTION,
    AWE_DECONSTRUCTION_FRACTION,
    # AWE_HISTORICAL_MATERIALS,
    # AWE_HISTORICAL_DISTRIBUTIONS,
    AWE_MATERIALS,
    AWE_EOL_ASSUMPTIONS,
    AWE_MAINTENANCE_COST
)

from input_pemwe import (
    PEMWE_CAPACITY_MW,
    # PEMWE_H2_PER_YEAR,
    PEMWE_H2_PER_HALF_YEAR,
    PEMWE_ELECTRICITY_KWH_PER_KG,
    PEMWE_WATER_KG_PER_KG,
    PEMWE_KOH_KG_PER_KG,
    PEMWE_KOH_PRICE_EUR_PER_KG,
    PEMWE_CONSTRUCTION_LABOUR_FRACTION,
    PEMWE_DECONSTRUCTION_FRACTION,
    PEMWE_MAINTENANCE_COST,
    # PEMWE_HISTORICAL_MATERIALS,
    # PEMWE_HISTORICAL_DISTRIBUTIONS,
    PEMWE_MATERIALS,
    PEMWE_EOL_ASSUMPTIONS,
)