import pandas as pd

# ============================================================
# PEMWE process inputs
# ============================================================

PEMWE_CAPACITY_MW = 5.0

# Legacy notebook cells 23 and 28 use 444,911.5105 kg H2 as the
# production value. In the clean model this is treated as annual
# production and split into two half-year operating periods.
PEMWE_H2_PER_YEAR = 444_911.5105
PEMWE_H2_PER_HALF_YEAR = PEMWE_H2_PER_YEAR / 2.0

# Legacy notebook cells 23 and 28.
PEMWE_ELECTRICITY_KWH_PER_KG = 50.0
PEMWE_WATER_KG_PER_KG = 9.3

# PEMWE does not use KOH in the operating-resource calculation.
PEMWE_KOH_KG_PER_KG = 0.0
PEMWE_KOH_PRICE_EUR_PER_KG = 0.0


# ============================================================
# PEMWE model parameters
# ============================================================

# Legacy notebook cell 5: per_m = 0.208.
PEMWE_CONSTRUCTION_LABOUR_FRACTION = 0.208

# Legacy notebook cells 24-31: per_cc = 0.025.
PEMWE_OPEX_LABOUR_FRACTION = 0.025

# Legacy notebook cells 11-12: percentage_of_cc = 0.06.
PEMWE_DECONSTRUCTION_FRACTION = 0.06

# Legacy notebook cells 24-31.
# Treatment remains an open model issue during the clean migration.
PEMWE_MAINTENANCE_COST = 3_750.0

# Legacy deterministic CAPEX call in cell 5 uses ictg=0.
PEMWE_INVESTMENT_GRANT = 0.0


# ============================================================
# Historical material-price mapping
# ============================================================

# Legacy PEMWE uncertainty logic treats these historical inputs
# as triangular distributions.
PEMWE_HISTORICAL_DISTRIBUTIONS = {
    "Steel": "triangular",
    "Copper": "triangular",
    "Titanium": "triangular",
    "Platinum": "triangular",
    "Iridium": "triangular",
}


# ============================================================
# PEMWE material/component inventory
# ============================================================
PEMWE_MATERIALS = pd.DataFrame({
    "Material": [
        "Steel",
        "Copper",
        "Titanium",
        "Platinum",
        "Iridium",
        "Carbon Paper / GDL",
        "Nafion N117",
        "FKM Gasket / Seal",
        "Stack Cooling Heat Exchanger",
        "Condenser",
        "Gas Water Separators",
        "Dry Cooler",
        "Power Cable",
        "Data Cable",
        "Pumps",
        "Rectifier / Power Electronics",
        "Control Unit",
        "Housing",
        "Foundation",
    ],
    "Amount": [
        625.0,
        68.0,
        3987.0,
        0.67,
        9.86,
        329.0,
        88.54,
        1.0,
        1.0,
        1.0,
        2.0,
        1.0,
        50.0,
        2000.0,
        6.0,
        1.0,
        1.0,
        2.0,
        4.5,
    ],
    "Unit": [
        "kg",
        "kg",
        "kg",
        "kg",
        "kg",
        "pcs",
        "m²",
        "pcs",
        "pcs",
        "pcs",
        "pcs",
        "pcs",
        "m",
        "m",
        "pcs",
        "pcs",
        "pcs",
        "pcs",
        "m³",
    ],
    "Unit Price": [
        0.5831,
        7.274755925478757,
        11.925,
        31263.020700000005,
        57171.74430158868,
        80.62,
        1800.0,
        796.36,
        48000.0,
        42000.0,
        2480.77,
        1594.78,
        92.0,
        0.78,
        9300.0,
        320699.0,
        6600.0,
        4950.0,
        181.0,
    ],
    "Price Unit": [
        "EUR/kg",
        "EUR/kg",
        "EUR/kg",
        "EUR/kg",
        "EUR/kg",
        "EUR/pcs",
        "EUR/m²",
        "EUR/pcs",
        "EUR/pcs",
        "EUR/pcs",
        "EUR/pcs",
        "EUR/pcs",
        "EUR/m",
        "EUR/m",
        "EUR/pcs",
        "EUR/pcs",
        "EUR/pcs",
        "EUR/pcs",
        "EUR/m³",
    ],
})


# def build_pemwe_materials(historical_metal_stats):
#     """
#     Build the active PEMWE material table.

#     Historical metal prices are injected from the central
#     HISTORICAL_METAL_STATS dictionary using their mean values.
#     Point-estimate component prices remain as defined above.
#     """

#     df = PEMWE_MATERIAL_SPECS.copy()

#     for idx, row in df.iterrows():
#         key = row["Historical Price Key"]

#         if key is None:
#             continue

#         if key not in historical_metal_stats:
#             raise KeyError(
#                 f"Missing historical metal statistics for '{key}'."
#             )

#         df.at[idx, "Unit Price"] = float(
#             historical_metal_stats[key]["mean"]
#         )

#     df["Unit Price"] = df["Unit Price"].astype(float)

#     return df.drop(
#         columns=["Historical Price Key"]
#     )


# ============================================================
# PEMWE end-of-life inventory
# ============================================================
#
# Legacy notebook cell 10.
#
# This is intentionally separate from the CAPEX material inventory:
# the legacy PEMWE EoL model uses its own EoL quantities, recycling
# fractions, selling/scrap prices, and transport/disposal prices.
#
# Legacy names "Carbon Paper" and "FKM" are standardized here to the
# same names used in the material/component inventory.

PEMWE_EOL_ASSUMPTIONS = pd.DataFrame({
    "Material": [
        "Steel",
        "Copper",
        "Titanium",
        "Platinum",
        "Iridium",
        "Carbon Paper / GDL",
        "Nafion N117",
        "FKM Gasket / Seal",
        "Stack Cooling Heat Exchanger",
        "Condenser",
        "Gas Water Separators",
        "Dry Cooler",
        "Power Cable",
        "Data Cable",
        "Pumps",
        "Rectifier / Power Electronics",
        "Control Unit",
        "Housing",
        "Foundation",
    ],
    "EoL Amount": [
        625.0,
        68.0,
        3987.0,
        0.75,
        9.86,
        35.0,
        121.0,
        322.24,
        1340.45,
        1157.73,
        1016.62,
        4495.0,
        292.0,
        103.44,
        1800.0,
        6000.0,
        400.0,
        4400.0,
        11250.0,
    ],
    "Recycle Ratio": [
        0.88,
        0.70,
        0.40,
        0.76,
        0.40,
        0.0,
        0.0,
        0.0,
        0.855,
        0.855,
        0.72,
        0.72,
        0.72,
        0.72,
        0.72,
        0.72,
        0.72,
        0.88,
        0.89,
    ],
    "Disposal Fraction": [
        0.12,
        0.30,
        0.60,
        0.24,
        0.60,
        1.0,
        1.0,
        1.0,
        0.145,
        0.145,
        0.28,
        0.28,
        0.28,
        0.28,
        0.28,
        0.28,
        0.28,
        0.12,
        0.11,
    ],
    "Salvage Unit Price": [
        0.23,
        7.13,
        0.0,
        16780.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.27,
        0.27,
        0.27,
        0.27,
        1.95,
        1.95,
        0.27,
        0.08,
        0.08,
        0.27,
        0.0,
    ],
    "Transport Disposal Unit Cost": [
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.23,
        0.10,
    ],
})


# ============================================================
# Source notes
# ============================================================

PEMWE_INPUT_SOURCE_NOTES = {
    "materials": (
        "Legacy notebook cell 4; point-estimate prices noted there "
        "as updated from PEMWE_2026.xlsx / Construction / Single price."
    ),
    "capex": (
        "Legacy notebook cell 5; construction labour per_m=0.208 "
        "and deterministic investment grant ictg=0."
    ),
    "eol_inventory": (
        "Legacy notebook cell 10; EoL amounts, recycling rates, "
        "disposal fractions, selling prices and transport costs."
    ),
    "eol_model": (
        "Legacy notebook cells 11-12; deconstruction fraction 0.06 "
        "and deterministic EoL setup."
    ),
    "operation": (
        "Legacy notebook cells 23-31; annual H2 value 444911.5105, "
        "50.0 kWh/kg H2, 9.3 kg water/kg H2, zero KOH, "
        "operating-labour factor 0.025 and maintenance EUR 3750."
    ),
}
