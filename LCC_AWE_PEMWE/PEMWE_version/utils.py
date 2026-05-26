import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt
from IPython.display import clear_output

# Prediction packages
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
#from statsforecast import StatsForecast
#from statsforecast.models import AutoARIMA, AutoETS
#import pyaf.ForecastEngine as autof
#from neuralprophet import NeuralProphet

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
    # 🔴 GEÄNDERT: prediction starts after the extended historical series, i.e. after 2025-S2.
    future = pd.DataFrame({"ds": pd.date_range(start="2026-01-31", periods=years*2, freq="6ME")})

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
    # 🔴 GEÄNDERT: prediction starts after the extended historical series, i.e. after 2025-S2.
    future = pd.DataFrame({"ds": pd.date_range(start="2026-01-31", periods=years*2, freq="6ME")})

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
    # 🔴 GEÄNDERT: prediction starts after the extended historical series, i.e. after 2025-S2.
    future = pd.DataFrame({"ds": pd.date_range(start="2026-01-31", periods=years*2, freq="6ME")})

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
    # 🔴 GEÄNDERT: prediction starts after the extended historical series, i.e. after 2025-S2.
    future = pd.DataFrame({"ds": pd.date_range(start="2026-01-31", periods=years*2, freq="6ME")})

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
