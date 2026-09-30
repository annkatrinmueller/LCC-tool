import pandas as pd
# ============================================================
# AWE process input
# ============================================================

AWE_CAPACITY_MW = 5.0

AWE_H2_PER_YEAR = 489_700.0  # kg 
AWE_H2_PER_HALF_YEAR = AWE_H2_PER_YEAR / 2.0 # kg 

AWE_ELECTRICITY_KWH_PER_KG = 52.8
AWE_WATER_KG_PER_KG = 10.0

AWE_KOH_KG_PER_KG = 0.0001
AWE_KOH_PRICE_EUR_PER_KG = 0.603


AWE_CONSTRUCTION_LABOUR_FRACTION = 0.208
AWE_MAINTENANCE_COST = 3750.0

# ============================================================
# AWE material input
# ============================================================

AWE_MATERIALS = pd.DataFrame(
    {
        "Material": [
            "Steel",
            "Nickel",
            "Copper",
            "Zirfon Membrane",
            "PTFE",
            "HDPE",
            "Power Cable",
            "Data Cable",
            "KOH Tank",
            "Gas Separator",
            "Heat Exchanger",
            "Water Pump",
            "Rectifier / Power Electronics",
            "Control Unit",
            "Housing",
        ],
        "Amount": [
            35511.808,
            6721.452,
            1855.36,
            464,
            3164.48,
            304,
            200,
            2000,
            1,
            2,
            1,
            4,
            22,
            1,
            1,
        ],
        "Unit": [
            "kg",
            "kg",
            "kg",
            "m²",
            "kg",
            "kg",
            "m",
            "m",
            "pcs",
            "pcs",
            "pcs",
            "pcs",
            "pcs",
            "pcs",
            "pcs",
        ],
        "Unit Price": [
            0.5831,
            18.08900217153637, 
            7.274755925478757,
            293.12,
            83.1818181818182,
            9.47368421052632,
            47.0,
            1.19,
            1195.82007741176,
            2496.38117647059,
            2735.0,
            497.0,
            15000.0,
            13350.4,
            4750.0,
        ],
        "Price Unit": [
            "EUR/kg",
            "EUR/kg",
            "EUR/kg",
            "EUR/m²",
            "EUR/kg",
            "EUR/kg",
            "EUR/m",
            "EUR/m",
            "EUR/pcs",
            "EUR/pcs",
            "EUR/pcs",
            "EUR/pcs",
            "EUR/pcs",
            "EUR/pcs",
            "EUR/pcs",
        ],
    }
)

AWE_EOL_ASSUMPTIONS = pd.DataFrame(
    {
        "Material": [
            "Steel",
            "Nickel",
            "Copper",
            "Zirfon Membrane",
            "PTFE",
            "HDPE",
            "Power Cable",
            "Data Cable",
            "KOH Tank",
            "Gas Separator",
            "Heat Exchanger",
            "Water Pump",
            "Rectifier / Power Electronics",
            "Control Unit",
            "Housing",
        ],
        "Recycle Ratio": [
            0.28,
            0.35,
            0.43,
            0,
            0,
            0,
            0.008,
            0.008,
            0.0002,
            0.0002,
            0.0002,
            0.0002,
            0.000013,
            0.000013,
            0.000043,
        ],
        "Transport Disposal Unit Cost": [0.23] * 15,
    }
)

AWE_DECONSTRUCTION_FRACTION = 0.06


AWE_EOL_UNCERTAINTY_HALF_WIDTHS = {
    "transport_disposal": 250.0,
    "salvage_value": 2500.0,
    "deconstruction_fraction": 0.0005,
    "inflation_rate": 0.0005,
    "equity": 0.005,
    "equity_return": 0.005,
    "debt_interest_rate": 0.0005,
}