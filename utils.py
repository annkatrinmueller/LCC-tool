import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt
from IPython.display import clear_output

# Prediction packages
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS
import pyaf.ForecastEngine as autof
from neuralprophet import NeuralProphet

# Prediction for water price in S1 and S2
def prophet_fit_predict_water(odf, var="BNV", years=50, plot=True):
    """
        fit a prophet model and predict for future dates.
    """
    # prepare expected column names
    pdf = odf.iloc[: , [0, odf.columns.get_loc(var)]].copy()
    pdf.rename(columns={"DATE": "ds", var: "y"}, inplace=True)
    pdf.columns = ['ds', 'y']

    # define the model
    model = Prophet(mcmc_samples=0)

    # fit the model
    model.fit(pdf, seed=1234)

    # define the period for which we want a prediction
    future = ['2025-07-31']  # forecast starts at last historical period after extending history to 2025-S2
    for i in range(26, 26+years):  # 2026 onward after 2025 history
            for j in [1, 7]:
                future.append(f"20{i}-0{j}-31")
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])

    # set random seed
    np.random.seed(1234)

    # use the model to make a forecast
    print(f"Predicting {var} for {years} Years...")
    prediction = model.predict(future)

    if plot:
        model.plot(prediction)
        plt.show()
    
    return prediction

# Elctricity cost S1 prediction: with war and epidemic
def prophet_fit_predict_elc(odf, var="BNV", years=50, plot=True):
    """
        fit a prophet model and predict for future dates.
    """
    # prepare expected column names
    pdf = odf.iloc[: , [0, odf.columns.get_loc(var)]].copy()
    pdf.rename(columns={"DATE": "ds", var: "y"}, inplace=True)
    pdf.columns = ['ds', 'y']

    # define the model
    model = Prophet(mcmc_samples=0)

    # fit the model
    model.fit(pdf, seed=1234)

    # define the period for which we want a prediction
    future = ['2025-07-31']  # forecast starts at last historical period after extending history to 2025-S2
    for i in range(26, 26+years):  # 2026 onward after 2025 history
        for j in [1, 7]:
            future.append(f"20{i}-0{j}-31")
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])

    # set random seed
    np.random.seed(1234)

    # use the model to make a prediction (forecast)
    print(f"Predicting {var} for {int(len(future)/2)} Years...")
    prediction = model.predict(future)

    if plot:
        model.plot(prediction)
        plt.show()
    
    return prediction

# Elctricity cost S2 prediction (the first 3 years): influence of war and epidemic 
# will only last for 3 years, after that, everything will back to normal.
def prophet_fit_predict_3_years_after_war(odf, var="BNV", years=50, plot=True):
    """
        fit a prophet model and predict for future dates.
    """
    # prepare expected column names
    pdf = odf.iloc[: , [0, odf.columns.get_loc(var)]].copy()
    pdf.rename(columns={"DATE": "ds", var: "y"}, inplace=True)
    pdf.columns = ['ds', 'y']

    # define the model
    model = Prophet(mcmc_samples=0)

    # fit the model
    model.fit(pdf, seed=1234)

    # define the period for which we want a prediction
    future = ['2025-07-31']  # forecast starts at last historical period after extending history to 2025-S2
    for i in range(26, 26+years):  # 2026 onward after 2025 history
        for j in [1, 7]:
            future.append(f"20{i}-0{j}-31")
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])

    # set random seed
    np.random.seed(1234)

    # use the model to make a forecast
    print(f"Predicting {var} for {int(len(future)/2)} Years...")
    prediction = model.predict(future)

    if plot:
        model.plot(prediction)
        plt.show()
    
    return prediction

# Elctricity cost S2 prediction (after 3 years): influence of war and epidemic 
# will only last for 3 years, after that, everything will back to normal.
def prophet_fit_predict_elc_rest(odf, var="BNV", years=50, plot=True):
    """
        fit a prophet model and predict for future dates.
    """
    # prepare expected column names
    pdf = odf.iloc[: , [0, odf.columns.get_loc(var)]].copy()
    pdf.rename(columns={"DATE": "ds", var: "y"}, inplace=True)
    pdf.columns = ['ds', 'y']

    # define the model
    model = Prophet(mcmc_samples=0)

    # fit the model
    model.fit(pdf, seed=1234)

    # define the period for which we want a prediction
    future = ['2025-07-31']  # forecast starts at last historical period after extending history to 2025-S2
    for i in range(26, 26+years):  # 2026 onward after 2025 history
        for j in [1, 7]:
            future.append(f"20{i}-0{j}-31")
    future = pd.DataFrame(future)
    future.columns = ['ds']
    future['ds']= pd.to_datetime(future['ds'])

    # set random seed
    np.random.seed(1234)

    # use the model to make a forecast
    print(f"Predicting {var} for {int(len(future)/2)} Years...")
    forecast = model.predict(future)

    if plot:
        model.plot(forecast)
        plt.show()
    
    return forecast


# - Equations -
# basics
def wacc_nominal(equity, eq_rate_return, debt, db_intrst_rate, **kwargs):
    return ((equity*eq_rate_return)/(equity+debt)) + ((debt*db_intrst_rate)/(equity+debt))

def wacc_real(inflation_rate, **kwargs):
    wacc_nom = wacc_nominal(**kwargs)
    return ((1+wacc_nom)/(1+inflation_rate)) - 1

def con_labour_cost(material_full, per_m, **kwargs):
    return(material_full*per_m)

def invest_cost(material, ictg, **kwargs):
    rwacc = wacc_real(**kwargs)
    labour = con_labour_cost(**kwargs)

    print(f"Construction Labour Cost: {labour}")
    return (material + labour - ictg)/((1 + rwacc)**0)

# for OPEX (with End-of-Life Cost)
def deconstruction_cost(construction_cost, percentage_of_cc, **kwargs):
    return (construction_cost*percentage_of_cc)

def end_of_life_cost(transport_cost_disposal, salvage_value, inflation_rate, **kwargs):
    decon_cost = deconstruction_cost(**kwargs)
    wacc_r=wacc_real(inflation_rate, **kwargs)
    return (decon_cost + transport_cost_disposal - salvage_value)/((1 + wacc_r)**20), wacc_r

def op_labour_cost(construction_cost, per_cc, **kwargs):
    return(construction_cost*per_cc)

def res_cost (water_price, electricity_price, **kwargs):
    op_labour = op_labour_cost(**kwargs)
    return (water_price + electricity_price + op_labour)

def operation_cost(resource_cost, maintenance_cost, full=False, **kwargs):
    op_labour = op_labour_cost(**kwargs)
    op_cost, wacc_r = end_of_life_cost(**kwargs)
    op_cost_t = op_cost
    if not full:
        op_cost = 0
    eol_val = op_cost

    if isinstance(resource_cost, list):
        n_resource_cost = []
        del resource_cost[0]

        for _ in range(20):
            n_resource_cost.append((resource_cost[0]+resource_cost[1]))
            del resource_cost[0]
            del resource_cost[0]

        for t, r in enumerate(n_resource_cost):
            op_cost += (r/((1+wacc_r)**t+1))
            op_cost_t += (((sum(n_resource_cost)+maintenance_cost)/20)/((1+wacc_r)**t+1))
    else:
        for t in range(20):
            op_cost += ((resource_cost+maintenance_cost/20)/((1+wacc_r)**t+1))

    print(f"Labour Cost: {op_labour}")
    print(f"End of life Cost: {eol_val}")

    return (op_labour + op_cost)


# for Tax impact
def tax_impact(resource_cost, depreciation, tax_rate, int_on_debt, **kwargs):
    wacc_r = wacc_real(**kwargs)
    n_resource_cost = []
    tax_impct = 0
    n_resource_cost.append(resource_cost[0])
    del resource_cost[0]

    for _ in range(20):
        n_resource_cost.append((resource_cost[0]+resource_cost[1])/2.0)
        del resource_cost[0]
        del resource_cost[0]


    for t, r in enumerate(n_resource_cost):
        tax_impct += ((r+int_on_debt+depreciation)/((1+wacc_r)**t+1))

    return tax_rate*tax_impct

# for TCO
def life_cycle_cost_of_h2(investment_cost, op_cost, tax_impact):
    return (investment_cost + op_cost - tax_impact)

# for LCOH
def levelized_cost_of_h2(life_cycle_cost_of_h2, h2_production, **kwargs):
    wacc_r = wacc_real(**kwargs)
    h2_p = 0
    for t in range(20):
        h2_p += ((h2_production/20)/((1+wacc_r)**t+1))
    # If:
    #   detailed data is needed
    # Then:
    #   print(f"---- {t} ----")
    #   print(f"Q: {((1+wacc_r)**t+1)}")

    return (life_cycle_cost_of_h2/h2_p)


# For Monaco
# focus: CAPEX-equation
def monte_capex(material, ictg, capex_labour, **kwargs):
    rwacc = wacc_real(**kwargs)
    return (material + capex_labour - ictg)/((1 + rwacc)**0)

# focus: CAPEX-material
def monte_pem_capex_5mw_soa(steel, copper, titanium, platin, iridium,
                            carbon_paper, nafion_N117, 
                            FKM, heat_exchanger_stack_cooling, 
                            heat_exchanger_condenser, 
                            gas_water_separators, dry_cooler, 
                            power_cable, data_cable, pumps, inverters, control_unit,
                            housing, foundation):
    
    return (6253*steel + 68*copper + 3987*titanium + 0.67*platin + 9.86*iridium + 329*carbon_paper + \
            88.54*nafion_N117 + 1*FKM + 1*heat_exchanger_stack_cooling + 1*heat_exchanger_condenser + \
            2*gas_water_separators + 1*dry_cooler + 50*power_cable + 2000*data_cable + 6*pumps + \
            1*inverters + 1*control_unit + 2*housing + 4.5*foundation)

# focus: OPEX-energy
def monte_res_cost(water_price, electricity_price, op_labour):
    return (water_price + electricity_price + op_labour)

# focus: OPEX-equation
def monte_opex(op_cost, op_labour, eol_cost):
    return (op_cost + op_labour + eol_cost)

# focus: End-of-Life cost-equation
def monte_eol_cost(transport_cost_disposal, salvage_value, decon_cost):
    return (decon_cost + transport_cost_disposal - salvage_value)

# focus: TCO-equation
def monte_lcc(investment_cost, opr_cost, tax_impact):
    return (investment_cost + opr_cost - tax_impact)


# additional AWE material-cost function, original PEMWE function above is kept unchanged.
def monte_awe_capex_5mw_current(steel, nickel, copper, zirfon_membrane, ptfe, hdpe,
                                power_cable, data_cable, koh_tank, gas_separator,
                                heat_exchanger, water_pump, rectifier_power_electronics,
                                control_unit, housing):
    return (35511.808*steel + 6721.452*nickel + 1855.36*copper + 464*zirfon_membrane +
            3164.48*ptfe + 304*hdpe + 200*power_cable + 2000*data_cable +
            1*koh_tank + 2*gas_separator + 1*heat_exchanger + 4*water_pump +
            22*rectifier_power_electronics + 1*control_unit + 1*housing)


# - Uncertainty distribution support -
# The functions below keep the Monte Carlo assumptions transparent and reproducible.
# They are intentionally small so that each distribution used in the notebooks can be
# traced back to one of two cases:
#   1. Historical time series available: triangular distribution.
#   2. No historical time series available: uniform distribution around the point estimate.
#
# A triangular distribution is used when a time series exists because the observed
# minimum and maximum provide empirical bounds and the historical mean gives a
# defensible most-likely value. This avoids negative values and avoids assuming a
# symmetric normal shape for commodity prices that may have short-lived shocks.
#
# A uniform distribution is used when there is only a catalogue value, supplier quote,
# engineering estimate or other point estimate. In that case no mode can be estimated
# from data, so every value inside the chosen range is treated as equally plausible.

DISTRIBUTION_POLICY = {
    "historical_time_series": "triangular: lower bound = observed minimum, mode = historical mean, upper bound = observed maximum",
    "point_estimate_only": "uniform: lower and upper bounds are built around the point estimate",
    "uniform_range_rule": "The relative range depends on material importance: high-impact items receive wider ranges than low-impact items."
}

# Cost-share thresholds used for point-estimate-only inputs.
# The uncertainty range is deliberately wider for components that dominate CAPEX,
# because an error in these inputs has a larger effect on the final LCC/LCOH result.
UNCERTAINTY_RANGE_BY_IMPORTANCE = {
    "high": {"min_share": 0.10, "relative_half_width": 0.25, "reason": "at least 10% of material cost"},
    "medium": {"min_share": 0.01, "relative_half_width": 0.20, "reason": "between 1% and 10% of material cost"},
    "low": {"min_share": 0.00, "relative_half_width": 0.10, "reason": "below 1% of material cost"},
}


def triangular_from_stats(stats):
    """Return scipy.stats.triang parameters from historical summary statistics.

    scipy's triangular distribution uses c=(mode-lower)/(upper-lower), loc=lower,
    and scale=upper-lower. Here the historical mean is used as the mode because it
    is the best single estimate of the most-likely price when only summary
    statistics are available. The value of c is clipped to [0, 1] as a safeguard
    against small rounding differences in manually entered statistics.
    """
    lower = float(stats["min"])
    upper = float(stats["max"])
    mode = float(stats.get("mean", lower + (upper - lower) / 2))
    if upper <= lower:
        return {"c": 0.5, "loc": lower, "scale": 0.0}
    c = (mode - lower) / (upper - lower)
    return {"c": float(max(0.0, min(1.0, c))), "loc": lower, "scale": upper - lower}


def uniform_from_bounds(lower, upper):
    """Return scipy.stats.uniform parameters from lower and upper bounds.

    scipy's uniform distribution uses loc as the lower bound and scale as the width
    of the interval. For example, lower=80 and upper=120 gives loc=80, scale=40.
    """
    lower = float(lower)
    upper = float(upper)
    if upper < lower:
        lower, upper = upper, lower
    return {"loc": lower, "scale": upper - lower}


def uniform_from_stats(stats):
    """Build a uniform distribution from a dictionary containing min and max."""
    return uniform_from_bounds(stats["min"], stats["max"])


def classify_cost_share(cost_share):
    """Classify a material by its contribution to total material cost."""
    if cost_share >= UNCERTAINTY_RANGE_BY_IMPORTANCE["high"]["min_share"]:
        return "high"
    if cost_share >= UNCERTAINTY_RANGE_BY_IMPORTANCE["medium"]["min_share"]:
        return "medium"
    return "low"


def uniform_around_value(value, cost_share=None, relative_half_width=None):
    """Create a uniform range around a point estimate.

    If relative_half_width is not provided, the function chooses it from the
    material cost share. This gives the largest uncertainty range to components
    that have the strongest effect on the final result.
    """
    value = float(value)
    if relative_half_width is None:
        importance = classify_cost_share(0.0 if cost_share is None else float(cost_share))
        relative_half_width = UNCERTAINTY_RANGE_BY_IMPORTANCE[importance]["relative_half_width"]
    lower = max(0.0, value * (1.0 - relative_half_width))
    upper = value * (1.0 + relative_half_width)
    return uniform_from_bounds(lower, upper)


def build_material_uncertainty_table(material_df, historical_materials=None):
    """Rank materials by cost contribution and assign distribution assumptions.

    The input table must contain 'Material and Component', 'Amount', 'Single Price'
    and 'Price'. The total row is ignored. The output is used as a documentation
    table in the notebook and as a basis for uniform ranges when no historical
    time series exists.
    """
    historical_materials = set(historical_materials or [])
    table = material_df.copy()
    table = table[table["Material and Component"].astype(str).str.lower() != "total"].copy()
    table["Price"] = pd.to_numeric(table["Price"], errors="coerce")
    total = table["Price"].sum()
    table["Cost share"] = table["Price"] / total if total else 0.0
    table["Importance"] = table["Cost share"].apply(classify_cost_share)
    table["Relative half-width"] = table["Importance"].map(lambda x: UNCERTAINTY_RANGE_BY_IMPORTANCE[x]["relative_half_width"])
    table["Distribution"] = table["Material and Component"].apply(lambda x: "triangular" if x in historical_materials else "uniform")
    table["Reason"] = table.apply(
        lambda row: "historical price series available" if row["Distribution"] == "triangular"
        else f"no time series; {row['Importance']} cost impact, uniform +/-{row['Relative half-width']:.0%}",
        axis=1,
    )
    return table.sort_values("Price", ascending=False).reset_index(drop=True)


def plot_price_series(df, date_col="DATE", value_cols=None, title="Historical price series"):
    """Plot one or more historical price series used to justify distribution choice."""
    value_cols = value_cols or [c for c in df.columns if c != date_col]
    ax = df.plot(x=date_col, y=value_cols, marker="o", figsize=(9, 4))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    plt.show()
    return ax
