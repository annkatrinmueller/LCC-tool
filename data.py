import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt
from IPython.display import clear_output


# - Water Data -

# water_data: from Berlin water price
# Historical water series extended to 2025-S2. Berlin water and wastewater tariffs are held constant after the last published tariff change.
water_data = {"DATE" : pd.date_range(start="2008", end="2026-01-31", freq="6ME", inclusive="left"),
              "WF" : [0.002071, 0.002071, 0.002038, 0.002038, 0.0020325, 0.0020325, 0.002027, 0.002027, 0.002027, 0.002027, 0.002027, 0.002027, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694, 0.001694],
              "WWF" : [0.002567, 0.002567, 0.002543, 0.002543, 0.002504, 0.002504, 0.002464, 0.002464, 0.002464, 0.002464, 0.002464, 0.002464, 0.002464, 0.002464, 0.002307, 0.002307, 0.002303, 0.002303, 0.002303, 0.002303, 0.00221, 0.00221, 0.00221, 0.00221, 0.00221, 0.00221, 0.00221, 0.00221, 0.002155, 0.002155, 0.002155, 0.002155, 0.002155, 0.002155, 0.002155, 0.002155]
             }



# - Electricity Data -

# Historical elc. data are from EUROSTAT
# data_elc: elc. data from EUROSTAT used elc. from 50 Mio kWh/a
# Historical electricity series extended through 2025-S2. Forecasts start from the last historical period for continuity.
data_elc = {
        "DATE"    : pd.date_range(start="2004", end="2026-01-31", freq="6ME", inclusive="left"),
        "no_tax"  : [0.0764, 0.0793, 0.0840, 0.0877, 0.0949, 0.0963, 0.1003, 0.0832,
                     0.0900, 0.0899, 0.0942, 0.0901, 0.0878, 0.0957, 0.1002, 0.1091,
                     0.1040, 0.1049, 0.1123, 0.1118, 0.1158, 0.1117, 0.1111, 0.1124,
                     0.1021, 0.0965, 0.0972, 0.0887, 0.0860, 0.0879, 0.0904, 0.1093,
                     0.1206, 0.1271, 0.1267, 0.1349, 0.1768, 0.1939, 0.1905, 0.1767,
                     0.1575, 0.1629, 0.1570, 0.1511],
        }

# data_nw_elc: elc. data from EUROSTAT used elc. from 50 Mio kWh/a
data_nw_elc = {
        "DATE"    : pd.date_range(start="2004", end="2018-07-31", freq="6ME", inclusive="left"),
        "no_tax"  : [0.0764, 0.0793, 0.0840, 0.0877, 0.0949, 0.0963, 0.1003, 0.0832, \
                     0.0900, 0.0899, 0.0942, 0.0901, 0.0878, 0.0957, 0.1002, 0.1091, \
                     0.1040, 0.1049, 0.1123, 0.1118, 0.1158, 0.1117, 0.1111, 0.1124, \
                     0.1021, 0.0965, 0.0972, 0.0887, 0.0860],
        }


# Backup (detailed historical data from BDEW)
# data_20: elc. data from BDEW, for used elc. from 160.000 - 20 Mio kWh/a
data_20 = {
        "DATE" : pd.date_range(start="1998", end="2023-07-31", freq="6ME", inclusive="left"),
        "BNV"  : [0.0915, 0.0915, 0.0851, 0.0851, 0.0546, 0.0546, 0.0561, 0.0561, 0.0599, 0.0599, 0.0617, 0.0617, 0.0702, 0.0702, 0.0765,\
                  0.0765, 0.0926, 0.0926, 0.0900, 0.0900, 0.1070, 0.1070, 0.0870, 0.0870, 0.0863, 0.0863, 0.0883, 0.0883, 0.0898, \
                  0.0898, 0.0785, 0.0785, 0.0695, 0.0695, 0.0719, 0.0719, 0.07, 0.07, 0.0802, 0.0802, 0.0897, 0.0897, 0.0948, 0.0948, \
                  0.0848, 0.0848, 0.1230, 0.1230, 0.2658, 0.5066, 0.3725],
        "KA"   : [0.0011]*51,
        "EEGU" : [0.0008, 0.0008, 0.0009, 0.0009, 0.002, 0.002, 0.0025, 0.0025, 0.0035, 0.0035, 0.0042, 0.0042, 0.0051, 0.0051, 0.0069, \
                  0.0069, 0.0088, 0.0088, 0.0102, 0.0102, 0.0116, 0.0116, 0.0131, 0.0131, 0.0205, 0.0205, 0.0353, 0.0353, 0.03592, \
                  0.03592, 0.05277, 0.05277, 0.06240, 0.06240, 0.06170, 0.06170, 0.06354, 0.06354, 0.06880, 0.06880, 0.06792, 0.06792,\
                  0.06405, 0.06405, 0.06756, 0.06756, 0.065, 0.065, 0.03723, 0, 0],
        "KWKG" : [0, 0, 0, 0, 0.0013, 0.0013, 0.0019, 0.0019, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, \
                  0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0003, 0.0003, 0.0004, 0.0004, 0.0007, \
                  0.0007, 0.0007, 0.0007, 0.0008, 0.0008, 0.0028, 0.0028, 0.0029, 0.0029, 0.0026, 0.0026, 0.0028, 0.0028, 0.00226,\
                  0.00226, 0.00254, 0.00254, 0.00378, 0.00378, 0.00357],
        "19U" : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0007, 0.0007, 0.0010, 0.0010, \
                 0.0023, 0.0023, 0.0015, 0.0015, 0.0024, 0.0024, 0.0025, 0.0025, 0.0024, 0.0024, 0.0020, 0.0020, 0.0023, 0.0023, \
                 0.0027, 0.0027, 0.0027, 0.0027, 0.0026],
        "ONU"  : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0017, 0.0017, 0.0017, \
                  0.0017, -0.0001, -0.0001, 0.0003, 0.0003, -0.00002, -0.00002, 0.00040, 0.00040, 0.00416, 0.00416, 0.00416, 0.00416, \
                  0.00395, 0.00395, 0.00419, 0.00419, 0.00591],
        "UAL"  : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00009, 0.00009, \
                  0.00006, 0.00006, 0, 0, 0.00006, 0.00006, 0.00011, 0.00011, 0.00005, 0.00005, 0.00007, 0.00007, 0.00009, 0.00009, \
                  0.00003, 0.00003, 0],
        "SST" : [0, 0, 0.0015, 0.0015, 0.0026, 0.0026, 0.0031, 0.0031, 0.0036, 0.0036, 0.0123, 0.0123, 0.0123, 0.0123, 0.0123, 0.0123, \
                 0.0123, 0.0123, 0.0123, 0.0123, 0.0123, 0.0123, 0.0123, 0.0123, 0.0123, 0.0123, 0.01537, 0.01537, 0.01537, 0.01537, \
                 0.01537, 0.01537, 0.01537, 0.01537, 0.01537, 0.01537, 0.01537, 0.01537, 0.01537, 0.01537, 0.01537, 0.01537, 0.01537, \
                 0.01537, 0.01537, 0.01537, 0.01537, 0.01537, 0.01537, 0.01537, 0.01537]
        }

# data_70_150: data from BDEW, for used elc. from 70-150 Mio kWh/a 
data_70_150 = {
        "DATE"    : pd.date_range(start="2007", end="2023-07-31", freq="6ME", inclusive="left"),
        "BNV"  : [0.0701, 0.0701, 0.0766, 0.0766, 0.0759, 0.0759, 0.0686, 0.0686, 0.0724, 0.0724, 0.0648, 0.0648, 0.0629, 0.0629,\
                  0.0597, 0.0597, 0.0555, 0.0555, 0.0429, 0.0429, 0.0445, 0.0445, 0.0468, 0.0468, 0.0497, 0.0497, 0.0462, 0.0462, 0.0626, 0.0626,\
                  0.1471, 0.1471, 0.1508],
        }



# Source metadata for historical inputs
HISTORICAL_INPUT_SOURCES = {
    "electricity": "Eurostat electricity price statistics / nrg_pc_205; active no_tax series extended to 2025-S2.",
    "water": "Berliner Wasserbetriebe published water and wastewater tariffs; active water_data extended to 2025-S2 using current tariff levels.",
}


# Historical distribution metadata for Monte Carlo uncertainty analysis
# The 20-year historical input sheets provide mean, standard deviation, variance,
# minimum, maximum and range. For simulation, historical cost inputs use a
# triangular distribution: min = observed historical minimum, max = observed
# historical maximum, mode = 20-year mean.
HISTORICAL_DISTRIBUTIONS = {
    'metals': {'Steel': {'mean': 0.5831, 'std': 0.1404676025581025, 'variance': 0.01973114736842105, 'min': 0.37, 'max': 0.905, 'range': 0.535, 'c': 0.3983177570093457}, 'Copper': {'mean': 7.274755925478757, 'std': 1.453860395124267, 'variance': 2.113710048510888, 'min': 4.867897429653678, 'max': 9.946881947016749, 'range': 5.078984517363071, 'c': 0.4738857713775197}, 'Titanium': {'mean': 11.925, 'std': 1.018706610312673, 'variance': 1.037763157894737, 'min': 10.5, 'max': 15.0, 'range': 4.5, 'c': 0.3166666666666668}, 'Iridium': {'mean': 57171.74430158868, 'std': 56402.09775194612, 'variance': 3181196630.820086, 'min': 11217.17283387429, 'max': 163003.2337735307, 'range': 151786.0609396564, 'c': 0.30275883821758803}, 'Nickel': {'mean': 18.08900217153637, 'std': 6.413925515723414, 'variance': 41.13844052124786, 'min': 9.595179080988459, 'max': 37.13584189039519, 'range': 27.54066280940673, 'c': 0.30841026410035344}, 'Platinum': {'mean': 57171.74430158882, 'std': 56402.09775194625, 'variance': 3181196630.820101, 'min': 11217.17283387431, 'max': 163003.2337735311, 'range': 151786.0609396567, 'c': 0.302758838217588}},
    'resources': {'AWE_S1_resource_cost_20y': {'mean': 61316741.02431111, 'std': 16435798.17947772, 'variance': 270135461796523.16, 'min': 44855543.53819999, 'max': 98889531.23820001, 'range': 54033987.70000002, 'c': 0.3046452462014221}, 'AWE_S2_resource_cost_20y': {'mean': 62305020.91764444, 'std': 16509099.359253697, 'variance': 272550361653710.84, 'min': 45838077.6182, 'max': 100647750.1182, 'range': 54809672.50000001, 'c': 0.3004386369840915}, 'PEMWE_S1_resource_cost_20y': {'mean': 52747650.5617215, 'std': 14140943.38226515, 'variance': 199966279740428.53, 'min': 38585459.726880506, 'max': 85074161.12937808, 'range': 46488701.402497575, 'c': 0.3046372647027765}, 'PEMWE_S2_resource_cost_20y': {'mean': 53597925.892899275, 'std': 14204014.922320748, 'variance': 201754039913510.47, 'min': 39430791.59683051, 'max': 86586860.26507808, 'range': 47156068.66824757, 'c': 0.30043077585066313}},
}

DISTRIBUTION_SELECTION_ARGUMENT = (
    'Historical metal, electricity and water inputs are modeled as triangular distributions '
    'using observed min, observed max and the historical mean as mode. This is preferred over '
    'a normal distribution for these short annual time series because all prices are positive, '
    'the observations can be skewed by commodity shocks, and triangular distributions avoid '
    'unrealistic negative tails while preserving the observed historical range.'
)


# Distribution policy used by the notebooks and Monte Carlo setup.
# Historical inputs use triangular distributions because the data provide empirical
# lower and upper bounds and a defensible most-likely value. Inputs without a time
# series use uniform distributions because there is no evidence for a single mode.
# The uniform range is selected by importance: high-cost components receive a wider
# uncertainty range than low-cost components.
DISTRIBUTION_POLICY = {
    "historical_time_series": "triangular distribution based on observed min, historical mean and observed max",
    "point_estimate_only": "uniform distribution around the point estimate",
}

UNCERTAINTY_RANGE_BY_IMPORTANCE = {
    "high": {"min_share": 0.10, "relative_half_width": 0.25},
    "medium": {"min_share": 0.01, "relative_half_width": 0.20},
    "low": {"min_share": 0.00, "relative_half_width": 0.10},
}
