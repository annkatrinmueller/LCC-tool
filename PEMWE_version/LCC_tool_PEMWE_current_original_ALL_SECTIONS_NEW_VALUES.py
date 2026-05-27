# PEMWE original all sections with current values
# Auto-generated from original notebook. All original sections retained; selected input values updated.

# %% [cell 0]
import numpy as np
import pandas as pd
import monaco as mc
from scipy.stats import randint, uniform

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


from functools import reduce

from utils import prophet_fit_predict_elc, monte_opex, monte_capex,prophet_fit_predict_water,\
                        prophet_fit_predict_3_years_after_war, prophet_fit_predict_elc_rest, invest_cost, deconstruction_cost, end_of_life_cost, \
                        operation_cost, life_cycle_cost_of_h2, levelized_cost_of_h2, monte_lcc, monte_pem_capex_5mw_soa

from data import data_elc, water_data, data_nw_elc
# 🔴 GEÄNDERT: compatibility patch so original df.append(...) cells run with pandas >= 2.0
if not hasattr(pd.DataFrame, "append"):
    def _df_append(self, other, ignore_index=False, **kwargs):
        if isinstance(other, dict):
            other = pd.DataFrame([other])
        return pd.concat([self, other], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append


# %% [cell 1]
# Number of decimal places for Pandas output
pd.options.display.float_format = '{:20,.3f}'.format

# %% [cell 2]
# CAPEX - material cost for 5 MW PEMWEWE
# 🔴 GEÄNDERT: original CAPEX cell retained, but PEMWE unit prices updated to current researched values.

# Feed material data and corresponding price in
# Material type
material = ['Stainless steel', 'Copper', 'Titanium', 'Platinum', 'Iridium', 'Carbon Paper / GDL', 'Nafion N117', 'FKM gasket/seal',\
            'Stack Cooling Heat Exchanger', 'Condenser', 'Gas Water Separators', 'Dry Cooler', 'Power Cable',\
            'Data Cable', 'Pumps', 'Rectifier / power electronics', 'Control Unit', 'Housing', 'Foundation']
# Set the corresponding amount for material type            
amount = [625, 68, 3987, 0.67, 9.86, 329, 88.54, 1,\
          1, 1, 2, 1, 50,\
          2000, 6, 1, 1, 2, 4.5]
# Set the unit for material type
material_unit = ['kg']*5
material_unit.extend(['pcs'])
material_unit.extend(['m2'])
material_unit.extend(['pcs']*5)
material_unit.extend(['m']*2)
material_unit.extend(['pcs']*4)
material_unit.extend(['m3'])

# Cooresponding single price
# 🔴 GEÄNDERT: updated current PEMWE prices
price = [2.439445, 11.683045, 6.124201, 55039.55063, 206467.59929, 122.032583, 3966.666667, 236.97, 48000, 42000, 2480.77, 34000, 92, 0.78, 9300, 320699, 6600, 4950, 181]

single_price_unit = ['€/kg']*5
single_price_unit.extend(['€/pcs'])
single_price_unit.extend(['€/m2'])
single_price_unit.extend(['€/pcs']*5)
single_price_unit.extend(['€/m']*2)
single_price_unit.extend(['€/pcs']*4)
single_price_unit.extend(['€/m3'])

price_unit = ['€']*19

# Use pandas to orgnize the data and provide the names of the categories in the table
df = pd.DataFrame({'Material and Component' : material, 
                   'Amount':  amount,
                   'Unit' : material_unit,
                   'Single Price': price,
                   'Sigle Price Unit': single_price_unit,
                   'Price Unit': price_unit
                    })

# Calculate price based on amount and single price
df.insert(5, 'Price', [a*b for a, b in zip(df['Amount'], df['Single Price'])])

# calculate total price of every material
df = df.append({'Material and Component': 'Total', 'Price': df['Price'].sum(), 'Price Unit':'€'}, ignore_index=True)

# add name of row in table (Total)
as_list = df.index.tolist()
as_list[-1] = 'Total'
df.index = as_list

df.fillna('', inplace=True)

df.head(22)


# %% [cell 3]
# CAPEX result (one figure)

# Extract total cost/price from the defined dataframe
material_cost = df['Price'].loc['Total']

# Feed needed data in
kwargs={
        'inflation_rate' : 0.01,                # Inflation rate
        'equity' : 0.25,                        # Equity
        'eq_rate_return' : 0.07,                # Equity rate of return
        'debt' : 0.75,                          # Debt
        'db_intrst_rate' : 0.045,               # Debt interest rate
        'material_full' : material_cost,        # Material cost - comes from the table above
        'per_m' : 0.208                         # Percentage of material cost (for labour cost)
        }

# Sort Labour and Investment cost out 
inv_cost = invest_cost(material=material_cost, ictg=0, **kwargs)        # ictg: Invest tax credit of grant

print(f"Investment Cost {inv_cost:.2f}")

# %% [cell 4]
# Monaco CAPEX - Price range and sensitivity ratio

# Function for running the monte carlo simulation
def run(material, capex_labour, inflation_rate, equity, eq_rate_return, debt, db_intrst_rate, ictg):
    
    # variables used within the monte_capex function (utils.py)
    kwargs={
        'inflation_rate' : inflation_rate,
        'equity' : equity,
        'eq_rate_return' : eq_rate_return,
        'debt' : debt,
        'db_intrst_rate' : db_intrst_rate,
        }

    # run the capex simulation
    capex = monte_capex(material=material, capex_labour=capex_labour, ictg=ictg, **kwargs)
    return (capex, )

# Rename the y-aixs in results
def preprocess(case):

    # The 'preprocess' function grabs the random input values for each case and structures it with any other data in the format the 'run' function expects
    # Preprocess function for extracting usable variables from input dataframes
    material = case.invals['Material'].val
    capex_labour = case.invals['CAPEX labour cost'].val
    inflation_rate = case.invals['Inflation rate'].val
    equity = case.invals['Equity'].val
    eq_rate_return = case.invals['Equity rate of return'].val
    debt = case.invals['Debt'].val
    db_intrst_rate = case.invals['Debt interest rate'].val
    ictg = case.invals['Invest tax credit of grant'].val

    return (material, capex_labour, inflation_rate, equity, eq_rate_return, debt, db_intrst_rate, ictg)

# The 'postprocess' function takes the output from the 'run' function and saves off the outputs for each case
def postprocess(case, capex):
    case.addOutVal(name='CAPEX', val=capex)
    case.addOutVal(name='CAPEX_case', val=case.ncase)
    return None

# Argument list of functions used in simulator function
fcns = {'run' : run,
        'preprocess' : preprocess,
        'postprocess': postprocess}

# Number of iterations in the monte carlo simulation
n_invests = 350 
# Random seed for reproducibility
seed = 123456 
# Usage of monaco simulator with the SOBOL Random function
sim = mc.Sim(name='CAPEX', ndraws=n_invests, 
            fcns=fcns, seed=seed, 
            samplemethod='sobol_random',
            singlethreaded=True,
            savecasedata=False,
            verbose=True, debug=True)

# Variables used in the simulation
# loc: represents the lower limit of the distribution
# scale: the interval length of the distribution
# e.g: using loc=0 and scale=1 we get a standard uniform distribution on [0, 1]
# By manually change loc and scale can manage the uncertainty of the cost
sim.addInVar(name='Material', dist=uniform, distkwargs={'loc': 1700000, 'scale': 500000})
sim.addInVar(name='CAPEX labour cost', dist=uniform, distkwargs={'loc': 410000, 'scale': 9000})
sim.addInVar(name='Invest tax credit of grant', dist=uniform, distkwargs={'loc': 0.1, 'scale': 1000})
sim.addInVar(name='Inflation rate', dist=uniform, distkwargs={'loc': 0.01, 'scale': 0.001})
sim.addInVar(name='Equity', dist=uniform, distkwargs={'loc': 0.15, 'scale': 0.01})
sim.addInVar(name='Equity rate of return', dist=uniform, distkwargs={'loc': 0.05, 'scale': 0.01})
sim.addInVar(name='Debt', dist=uniform, distkwargs={'loc': 0.75, 'scale': 0.01})
sim.addInVar(name='Debt interest rate', dist=uniform, distkwargs={'loc': 0.045, 'scale': 0.001})

# Run the Simulation
sim.runSim()

# Calculate the mean and 5-95th percentile
# Statistics for the dice sum
sim.outvars['CAPEX'].addVarStat('mean')
sim.outvars['CAPEX'].addVarStat('percentile', {'p':[0.05, 0.95]})

# Plots a histogram of the dice sum
mc.plot(sim.outvars['CAPEX'])
plt.xlabel('CAPEX range in Mio.€')


# Creates a scatter plot of the sum vs the roll
# Number, showing randomness
mc.plot(sim.outvars['CAPEX'],
sim.outvars['CAPEX_case'])
# add Mio.€ in
plt.xlabel('CAPEX range in Mio.€')

# Calculate the sensitivity of the dice sum to each of the input variables
sim.calcSensitivities('CAPEX')
sim.outvars['CAPEX'].plotSensitivities()

# %% [cell 5]
# Monaco material cost - Price range and sensitivity ratio
# 🔴 GEÄNDERT: all original Monaco material-cost section retained; distributions updated around current PEMWE prices.

# Function for running the monte carlo simulation
def run(steel, copper, titanium, platin, iridium,
                carbon_paper, nafion_N117, 
                FKM, heat_exchanger_stack_cooling, 
                heat_exchanger_condenser, 
                gas_water_separators, dry_cooler, 
                power_cable, data_cable, pumps, inverters, control_unit,
                housing, foundation):

    # run the material cost simulation
    material_cost = monte_pem_capex_5mw_soa(steel=steel, copper=copper, titanium=titanium, platin=platin, iridium=iridium, 
                                               carbon_paper=carbon_paper, nafion_N117=nafion_N117, FKM=FKM, 
                                               heat_exchanger_stack_cooling=heat_exchanger_stack_cooling, 
                                               heat_exchanger_condenser=heat_exchanger_condenser, 
                                               gas_water_separators=gas_water_separators, dry_cooler=dry_cooler, 
                                               power_cable=power_cable, data_cable=data_cable, pumps=pumps, inverters=inverters, control_unit=control_unit,
                                               housing=housing, foundation=foundation)
    return (material_cost, )

# Rename the y-axis in results
def preprocess(case):
    steel = case.invals['Stainless steel'].val
    copper = case.invals['Copper'].val
    titanium = case.invals['Titanium'].val
    platin = case.invals['Platin'].val
    iridium = case.invals['Iridium'].val
    carbon_paper = case.invals['Carbon paper'].val
    nafion_N117 = case.invals['Nafion N117'].val
    FKM = case.invals['FKM'].val
    heat_exchanger_stack_cooling = case.invals['Heat exchanger stack cooling'].val
    heat_exchanger_condenser = case.invals['Condenser'].val
    gas_water_separators = case.invals['Gas water separators'].val
    dry_cooler = case.invals['Dry cooler'].val
    power_cable=case.invals['Power cable'].val
    data_cable=case.invals['Data cable'].val
    pumps = case.invals['Pumps'].val
    inverters = case.invals['Power electronics'].val
    control_unit = case.invals['Control unit'].val
    housing = case.invals['Housing'].val
    foundation = case.invals['Foundation'].val
    return (steel, copper, titanium, platin, iridium, carbon_paper, nafion_N117, FKM, heat_exchanger_stack_cooling, heat_exchanger_condenser, gas_water_separators, dry_cooler, power_cable, data_cable, pumps, inverters, control_unit, housing, foundation)

def postprocess(case, construction_cost):
    case.addOutVal(name='Material Cost', val=construction_cost)
    case.addOutVal(name='Material case', val=case.ncase)
    return None

fcns = {'run' : run, 'preprocess' : preprocess, 'postprocess': postprocess}
n_invests = 350
seed = 123456
sim = mc.Sim(name='Material Cost', ndraws=n_invests, fcns=fcns, seed=seed, samplemethod='sobol_random', singlethreaded=True, savecasedata=False, verbose=True, debug=True)

# Variables used in the simulation
sim.addInVar(name='Stainless steel', dist=uniform, distkwargs={'loc': 1.95, 'scale': 0.98})
sim.addInVar(name='Copper', dist=uniform, distkwargs={'loc': 9.35, 'scale': 4.67})
sim.addInVar(name='Titanium', dist=uniform, distkwargs={'loc': 4.90, 'scale': 2.45})
sim.addInVar(name='Platin', dist=uniform, distkwargs={'loc': 44000, 'scale': 22000})
sim.addInVar(name='Iridium', dist=uniform, distkwargs={'loc': 165000, 'scale': 83000})
sim.addInVar(name='Carbon paper', dist=uniform, distkwargs={'loc': 98, 'scale': 49})
sim.addInVar(name='Nafion N117', dist=uniform, distkwargs={'loc': 3170, 'scale': 1590})
sim.addInVar(name='FKM', dist=uniform, distkwargs={'loc': 190, 'scale': 95})
sim.addInVar(name='Heat exchanger stack cooling', dist=uniform, distkwargs={'loc': 46000, 'scale': 4000})
sim.addInVar(name='Condenser', dist=uniform, distkwargs={'loc': 40000, 'scale': 6000})
sim.addInVar(name='Gas water separators', dist=uniform, distkwargs={'loc': 2400, 'scale': 200})
sim.addInVar(name='Dry cooler', dist=uniform, distkwargs={'loc': 30000, 'scale': 8000})
sim.addInVar(name='Power cable', dist=uniform, distkwargs={'loc': 80, 'scale': 100})
sim.addInVar(name='Data cable', dist=uniform, distkwargs={'loc': 0.5, 'scale': 0.9})
sim.addInVar(name='Pumps', dist=uniform, distkwargs={'loc': 9000, 'scale': 900})
sim.addInVar(name='Power electronics', dist=uniform, distkwargs={'loc': 300000, 'scale': 60000})
sim.addInVar(name='Control unit', dist=uniform, distkwargs={'loc': 6000, 'scale': 1000})
sim.addInVar(name='Housing', dist=uniform, distkwargs={'loc': 4800, 'scale': 300})
sim.addInVar(name='Foundation', dist=uniform, distkwargs={'loc': 100, 'scale': 200})

sim.runSim()
sim.outvars['Material Cost'].addVarStat('mean')
sim.outvars['Material Cost'].addVarStat('percentile', {'p':[0.05, 0.95]})
mc.plot(sim.outvars['Material Cost'])
plt.xlabel('Material cost range in Mio.€')
mc.plot(sim.outvars['Material Cost'], sim.outvars['Material case'])
plt.xlabel('Material cost range in Mio.€')
sim.calcSensitivities('Material Cost')
sim.outvars['Material Cost'].plotSensitivities()


# %% [cell 6]
# Transport cost

# List all the cost indicators
waste_typs = ['Mixed scrap', 'Waste for recycling', 'Gypsum board, gypsum waste']
price = [0.23, 0.23, 0.1]
unit = ['€/kg']*3

df = pd.DataFrame({
                   'Waste type': waste_typs,
                   'Price': price,
                   'Unit': unit
                  })

df.head()

# %% [cell 7]
# OPEX S1 and S2 - End of Life (EoL) cost - Salvage value (SV)
# 🔴 GEÄNDERT: original EoL table retained; PEMWE material prices updated.

# Feed material data and corresponding amount in
material = ['Stainless steel', 'Copper', 'Titanium', 'Platinum', 'Iridium', 'Carbon Paper', 'Nafion N117', 'FKM',\
            'Stack Cooling Heat Exchanger', 'Condenser', 'Gas Water Separators', 'Dry Cooler', 'Power Cable',\
            'Data Cable', 'Pumps', 'Rectifier / power electronics', 'Control Unit', 'Housing', 'Foundation']
amount = [625, 68, 3987, 0.75, 9.86, 35, 121, 322.24,\
          1340.45, 1157.73, 1016.62, 4495, 292,\
          103.44, 1800, 6000, 400, 4400, 11250]

# Feed the recycle percentage of the material, corresponding selling price and transport cost in
recycling_rate = [0.88, 0.7, 0.4, 0.76, 0.4, 0, 0, 0,\
                  0.855, 0.855, 0.72, 0.72, 0.72,\
                  0.72, 0.72, 0.72, 0.72, 0.88, 0.89]
disposal_percent = [0.12, 0.3, 0.6, 0.24, 0.6, 1, 1, 1,\
                    0.145, 0.145, 0.28, 0.28, 0.28,\
                    0.28, 0.28, 0.28, 0.28, 0.12, 0.11]
selling_price = [0.23, 7.13, 0, 16780, 0, 0, 0, 0,\
                 0.27, 0.27, 0.27, 0.27, 1.95, \
                 1.95, 0.27, 0.08, 0.08, 0.27, 0]

transport_cost = [0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,\
                  0.23, 0.23, 0.23, 0.23, 0.23,\
                  0.23, 0.23, 0.23, 0.23, 0.23, 0.1]

# Set the unit
recycle_unit = ['€/kg']*19
material_unit = ['kg']*19
transport_cost_unit = ['€/kg']*19

# Use pandas to orgnize the data and provide the names of the categories in the table
df = pd.DataFrame({'Material' : material, 
                   'Amount':  amount,
                   'Amount unit' : material_unit,
                   'Recycling rate': recycling_rate,
                   'Selling price': selling_price,
                   'Price unit': recycle_unit,                   
                   'Disposal percentage': disposal_percent,
                   'Disposal unit': material_unit,
                   'Transport single cost': transport_cost,
                   'Transport cost unit' : transport_cost_unit
                    })

# Provide the calculation method within the table
df.insert(4, 'Recycling amount', [a*b for a, b in zip(df['Amount'], df['Recycling rate'])])
df.insert(8, 'Disposal amount', [a*b for a, b in zip(df['Amount'], df['Disposal percentage'])])
df.insert(12, 'Transport cost', [a*b for a, b in zip(df['Amount'], df['Transport single cost'])])
df.insert(13, 'Salvage value', [a*b for a, b in zip(df['Recycling amount'], df['Selling price'])])

# Quick hack to change the indices of the df_global dataframe and include the last column to be named "Total"
df = df.append({'Disposal amount': df['Disposal amount'].sum(), 
                'Transport cost': df['Transport cost'].sum(),
                'Salvage value': df['Salvage value'].sum()}, ignore_index=True)

as_list = df.index.tolist()
as_list[-1] = 'Total'
df.index = as_list

df.fillna('', inplace=True)

df.head(20)


# %% [cell 8]
# OPEX S1 and S2 - Deconstruction cost

# Feed data in function deconstruction_cost (utils.py)
# percentage of cc = percentage of construction cost (for calculate the deconstruction cost)
decon_cost = deconstruction_cost(construction_cost=material_cost, percentage_of_cc=0.06)
print(f"Deconstruction cost {decon_cost:.2f}")

# %% [cell 9]
 # OPEX S1 and S2 - EoL cost

# EoL cost is caculated here based on all the parameters listed below
kwargs={
        'equity' : 0.25,                        # Equity
        'eq_rate_return': 0.07,                 # Equity rate of return
        'debt': 0.75,                           # Debt
        'db_intrst_rate': 0.045,                # Debt interest rate
        'construction_cost' : material_cost,    # Material cost - comes from table "CAPEX - material cost"
        'percentage_of_cc' : 0.06               # Percentage of material cost (for labour cost)
        }

transport_cost_disposal = df['Transport cost'].loc['Total']
salvage_value = df['Salvage value'].loc['Total']

eol_cost, _ = end_of_life_cost(transport_cost_disposal=transport_cost_disposal, salvage_value=salvage_value, inflation_rate=0.01, **kwargs)
print(f"End of Life {eol_cost:.2f}")

# %% [cell 10]
# Monaco EoL cost - Price range and sensitivity ratio

# Function for running the monte carlo simulation
def run(transport_cost_disposal, salvage_value, construction_cost, percentage_of_cc, inflation_rate, equity, eq_rate_return, \
                debt, db_intrst_rate):
    
    # variables used within the end_of_life_cost function (utils.py)
    kwargs={
        'equity' : equity,
        'eq_rate_return' : eq_rate_return,
        'debt' : debt,
        'db_intrst_rate' : db_intrst_rate,
        'construction_cost' : construction_cost,      
        'percentage_of_cc' : percentage_of_cc       
        }
    
    # run the capex simulation
    eol_cost, _ = end_of_life_cost(transport_cost_disposal=transport_cost_disposal, salvage_value=salvage_value, inflation_rate=inflation_rate, \
                                **kwargs)
    return (eol_cost, )

# Rename the y-aixs in results
def preprocess(case):

    # The 'preprocess' function grabs the random input values for each case and structures it with any other data in the format the 'run' function expects
    # Preprocess function for extracting usable variables from input dataframes
    transport_cost_disposal = case.invals['Transport cost'].val
    salvage_value = case.invals['Salvage value'].val
    construction_cost = case.invals['Construction cost'].val
    percentage_of_cc = case.invals['Percentage of construction cost'].val
    inflation_rate = case.invals['Inflation rate'].val
    equity = case.invals['Equity'].val
    eq_rate_return = case.invals['Equity rate of return'].val
    debt = case.invals['Debt'].val
    db_intrst_rate = case.invals['Debt interest rate'].val

    return (transport_cost_disposal, salvage_value, construction_cost, percentage_of_cc, inflation_rate, equity, eq_rate_return, debt, db_intrst_rate)


# The 'postprocess' function takes the output from the 'run' function and saves off the outputs for each case
def postprocess(case, eol_cost):
    case.addOutVal(name='EoL Cost', val=eol_cost)
    case.addOutVal(name='eol_case', val=case.ncase)
    return None

# Argument list of functions used in simulator function
fcns = {'run' : run,
        'preprocess' : preprocess,
        'postprocess': postprocess}

# Number of iterations in the monte carlo simulation
n_invests = 350
# Random seed for reproducibility
seed = 123456 
# Usage of monaco simulator with the SOBOL Random function
sim = mc.Sim(name='End Of Life Cost', ndraws=n_invests, 
            fcns=fcns, seed=seed, 
            samplemethod='sobol_random',
            singlethreaded=True,
            savecasedata=False,
            verbose=True, debug=True)

# Variables used in the simulation
# By manually change loc and scale can manage the uncertainty of the cost
sim.addInVar(name='Transport cost', dist=uniform, distkwargs={'loc': 7000, 'scale': 500})
sim.addInVar(name='Salvage value', dist=uniform, distkwargs={'loc': 12000, 'scale': 5000})
sim.addInVar(name='Construction cost', dist=uniform, distkwargs={'loc': 1700000, 'scale': 500000})
sim.addInVar(name='Percentage of construction cost', dist=uniform, distkwargs={'loc': 0.06, 'scale': 0.001})
sim.addInVar(name='Inflation rate', dist=uniform, distkwargs={'loc': 0.01, 'scale': 0.001})
sim.addInVar(name='Equity', dist=uniform, distkwargs={'loc': 0.15, 'scale': 0.01})
sim.addInVar(name='Equity rate of return', dist=uniform, distkwargs={'loc': 0.05, 'scale': 0.01})
sim.addInVar(name='Debt', dist=uniform, distkwargs={'loc': 0.75, 'scale': 0.01})
sim.addInVar(name='Debt interest rate', dist=uniform, distkwargs={'loc': 0.045, 'scale': 0.001})

# Run the Simulation
sim.runSim()

# Calculate the mean and 5-95th percentile
# Statistics for the dice sum
sim.outvars['EoL Cost'].addVarStat('mean')
sim.outvars['EoL Cost'].addVarStat('percentile', {'p':[0.05, 0.95]})

# Plots a histogram of the dice sum
mc.plot(sim.outvars['EoL Cost'])
plt.xlabel('EoL cost range in €')


# Creates a scatter plot of the sum vs the roll
# Number, showing randomness
mc.plot(sim.outvars['EoL Cost'], 
        sim.outvars['eol_case'])
plt.xlabel('EoL cost range in €')

# Calculate the sensitivity of the dice sum to each of the input variables
sim.calcSensitivities('EoL Cost')
sim.outvars['EoL Cost'].plotSensitivities()


# %% [cell 11]
# setup plt plot style
plt.style.use('seaborn-v0_8-ticks')

# %% [cell 12]
# Prophet - electricity (elc.) cost prediction in scenario 1 (S1)

# Insert historical data for elc. cost
df_elc = pd.DataFrame(data=data_elc)

# Run Prophet for elc. cost prediction (here only incl. "Beschaffung, Netzentgelt, Vertrieb"(BNV) + "Konzessionsabgabe"(KA))
# Use Pandas' iloc function to extract the yhat_lower(2), yhat_upper(3) and yhat(15) (mean) from the prophet result dataframe
prophet_NT_hat = prophet_fit_predict_elc(odf=df_elc, var="no_tax", years=20, plot=False).iloc[: , [2, 3, 15]]

# - if followed element of elc. cost needed to be considered, then... - 
# EEG-Umlage:                     prophet_EEGU_hat = prophet_fit_predict(odf=df_elc, var="EEGU", years=20, plot=False).iloc[: , [2, 3, 15]]
# KWKG-Umlage:                    prophet_KWKG_hat = prophet_fit_predict(odf=df_elc, var="KWKG", years=20, plot=False).iloc[: , [2, 3, 15]]
# §19 StromNEV-Umlage:            prophet_NU_hat = prophet_fit_predict(odf=df_elc, var="19U", years=20, plot=False).iloc[: , [2, 3, 15]]
# Offshore-Netzumlage:            prophet_ONU_hat = prophet_fit_predict(odf=df_elc, var="ONU", years=20, plot=False).iloc[: , [2, 3, 15]]
# Umlage für abschaltbare Lasten: prophet_UAL_hat = prophet_fit_predict(odf=df_elc, var="UAL", years=20, plot=False).iloc[: , [2, 3, 15]]
# Stromsteuer:                    prophet_SST_hat = prophet_fit_predict(odf=df_elc, var="SST", years=20, plot=False).iloc[: , [2, 3, 15]]


# Results of elc. cost prediction (plot + table)
# Apply an add reduce lambda function to the values of each row in prophet_BNV_hat and prophet_KA_hat
prophet_preds_elc = reduce(lambda a, b: a.add(b, fill_value=0), [prophet_NT_hat])

# Set the interval for time-series prediction
# Ground-truth: previously given data
prophet_preds_elc['DATE'] = pd.date_range(start="2025-07-31", end="2045-07-31", freq="6ME")
prophet_preds_elc.rename(columns = {'yhat':'S1_elc_price_mean', 'yhat_lower':'S1_elc_price_lower',
                                    'yhat_upper':'S1_elc_price_upper'}, inplace = True)

# Sum all ground-truth data and predictions
df_elc['History'] = df_elc.sum(axis=1, numeric_only=True)

# Creat Figure and subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Draw the historical data
ax.plot(df_elc['DATE'], df_elc['History'], label='Historical Data', linestyle='-', linewidth=2, color='black')

# Draw the predictions
ax.plot(prophet_preds_elc['DATE'], prophet_preds_elc['S1_elc_price_mean'], label='Mean Electricity Price', linestyle='--', linewidth=2, color=(136/255, 86/255, 167/255))
ax.plot(prophet_preds_elc['DATE'], prophet_preds_elc['S1_elc_price_lower'], label='Lowest Electricity Price', linestyle='--', linewidth=2, color=(28/255, 144/255, 153/255))
ax.plot(prophet_preds_elc['DATE'], prophet_preds_elc['S1_elc_price_upper'], label='Highest Electricity Price', linestyle='--', linewidth=2, color=(217/255, 95/255, 14/255))

# Set lables
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('€/kWh', fontsize=12)
#ax.set_title('Electricity Price Prediction', fontsize=14)
ax.legend(fontsize=10)

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()

# Show table
prophet_preds_elc.head(50)

# %% [cell 13]
# Prophet - electricity (elc.) cost prediction in scenario 2 (S2) - Part 1.1

# For elctricity and water price prediction
df_elc_3y = pd.DataFrame(data=data_elc)

# Electricity cost prediction - 3 years with consideration of war and epidemic
prophet_NT_hat = prophet_fit_predict_3_years_after_war(odf=df_elc_3y, var="no_tax", years=3, plot=False).iloc[: , [2, 3, 15]]

# List and graphic of electricity cost prediction - 3 years with consideration of war and epidemic
prophet_preds_elc_3y = reduce(lambda a, b: a.add(b, fill_value=0), [prophet_NT_hat])

# Set the interval for time-series prediction
# Ground-truth: previously given data
df_elc_3y['History'] = df_elc_3y.sum(axis=1, numeric_only=True)
prophet_preds_elc_3y['DATE'] = pd.date_range(start="2025-07-31", end="2028-07-31", freq="6ME")
prophet_preds_elc_3y.rename(columns = {'yhat':'S2_elc_price_mean', 'yhat_lower':'S2_elc_price_lower',
                                    'yhat_upper':'S2_elc_price_upper'}, inplace = True)

# Creat Figure and subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Draw the historical data
ax.plot(df_elc_3y['DATE'], df_elc_3y['History'], label='Historical Data', linestyle='-', linewidth=2, color='black')

# Draw the predictions
ax.plot(prophet_preds_elc_3y['DATE'], prophet_preds_elc_3y['S2_elc_price_mean'], label='Mean Electricity Price', linestyle='--', linewidth=2, color=(136/255, 86/255, 167/255))
ax.plot(prophet_preds_elc_3y['DATE'], prophet_preds_elc_3y['S2_elc_price_lower'], label='Lowest Electricity Price', linestyle='--', linewidth=2, color=(28/255, 144/255, 153/255))
ax.plot(prophet_preds_elc_3y['DATE'], prophet_preds_elc_3y['S2_elc_price_upper'], label='Highest Electricity Price', linestyle='--', linewidth=2, color=(217/255, 95/255, 14/255))


# Set lables
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('€/kWh', fontsize=12)
#ax.set_title('Electricity Price Prediction', fontsize=14)
ax.legend(fontsize=10)

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()

# Show table
prophet_preds_elc_3y.head(50)

# %% [cell 14]
# Prophet - electricity (elc.) cost prediction in scenario 2 (S2) - Part 1.2

df_elc_rest = pd.DataFrame(data=data_nw_elc)

# Electricity cost prediction - without consideration of war and epidemic
prophet_NT_hat = prophet_fit_predict_elc_rest(odf=df_elc_rest, var="no_tax", years=20, plot=False).iloc[: , [2, 3, 15]]

# List and graphic of electricity cost prediction - after 3 years, the impact of war and epedemic will be gone, everything will be back to "normal" (cost trend before 2019)
prophet_preds_elc_nw = reduce(lambda a, b: a.add(b, fill_value=0), [prophet_NT_hat])

# Set the interval for time-series prediction
# Ground-truth: previously given data
df_elc_rest['History'] = df_elc_rest.sum(axis=1, numeric_only=True)
prophet_preds_elc_nw['DATE'] = pd.date_range(start="2025-07-31", end="2045-07-31", freq="6ME")
prophet_preds_elc_nw.rename(columns = {'yhat':'S2_elc_price_mean', 'yhat_lower':'S2_elc_price_lower',
                                    'yhat_upper':'S2_elc_price_upper'}, inplace = True)

# Creat Figure and subplot
fig, ax = plt.subplots(figsize=(10, 6))

# Draw the historical data
ax.plot(df_elc_rest['DATE'], df_elc_rest['History'], label='Historical Data', linestyle='-', linewidth=2, color='black')

# Draw the predictions
ax.plot(prophet_preds_elc_nw['DATE'], prophet_preds_elc_nw['S2_elc_price_mean'], label='Mean Electricity Price', linestyle='--', linewidth=2, color=(136/255, 86/255, 167/255))
ax.plot(prophet_preds_elc_nw['DATE'], prophet_preds_elc_nw['S2_elc_price_lower'], label='Lowest Electricity Price', linestyle='--', linewidth=2, color=(28/255, 144/255, 153/255))
ax.plot(prophet_preds_elc_nw['DATE'], prophet_preds_elc_nw['S2_elc_price_upper'], label='Highest Electricity Price', linestyle='--', linewidth=2, color=(217/255, 95/255, 14/255))

# Set lables
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('€/kWh', fontsize=12)
#ax.set_title('Electricity Price Prediction', fontsize=14)
ax.legend(fontsize=10)

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()


#ax = df_elc_rest.plot(kind='line', x='DATE', y='History')
#prophet_preds_elc_nw.plot(ax = ax, x='DATE')

#plt.show()

prophet_preds_elc_nw.head(50)


# %% [cell 15]
# Prophet - electricity (elc.) cost prediction in scenario 2 (S2) - Part 1.2.1
# List of electricity price prediction - without 2023-2026

prophet_elc_rest=prophet_preds_elc_nw.head(50)
prophet_elc_rest.drop([0, 1, 2, 3, 4, 5, 6], axis=0, inplace=True)
print(prophet_elc_rest)

# %% [cell 16]
# Prophet - electricity (elc.) cost prediction in scenario 2 (S2) - Part 1.2.2
# Combine the 2 prediction together - list it

prophet_elc_price = pd.concat([prophet_preds_elc_3y, prophet_elc_rest], axis=0)
prophet_elc_price.head(50)

# %% [cell 17]
# Prophet - electricity (elc.) cost prediction in scenario 2 (S2) - Part 2
# Combine the 2 prediction together, plot the final S2 prediction

# Set plot size
fig, ax = plt.subplots(figsize=(10, 6))

# Feed data
date_history = df_elc_3y['DATE'].to_list()
data_history = df_elc_3y['History'].to_list()

lower_data = prophet_preds_elc_3y['S2_elc_price_lower'].to_list()
n_rest_elm = len(prophet_preds_elc_nw['S2_elc_price_lower'].to_list()) - len(lower_data)
upper_data = prophet_preds_elc_3y['S2_elc_price_upper'].to_list()
pred_data = prophet_preds_elc_3y['S2_elc_price_mean'].to_list()

date_pred = prophet_preds_elc_nw['DATE'].to_list()
lower_data.extend(prophet_preds_elc_nw['S2_elc_price_lower'].to_list()[len(lower_data):])
upper_data.extend(prophet_preds_elc_nw['S2_elc_price_upper'].to_list()[len(upper_data):])
pred_data.extend(prophet_preds_elc_nw['S2_elc_price_mean'].to_list()[len(pred_data):])

# Set color
ax.plot(date_history, data_history, label='History', color='black')
ax.plot(date_pred, lower_data, color=(28/255, 144/255, 153/255), linestyle='--', label='S2_elc_price_lower')
ax.plot(date_pred, upper_data, color=(217/255, 95/255, 14/255), linestyle='--', label='S2_elc_price_upper')
ax.plot(date_pred, pred_data, color=(136/255, 86/255, 167/255), linestyle='--', label='S2_elc_price_mean')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('€/kWh', fontsize=12)
# If:
#   title is needed
# Then:
#   ax.set_title('S2', fontsize=14)
ax.legend(fontsize=10)

plt.grid(True)
plt.tight_layout()
plt.show()


# %% [cell 18]
# Prophet - Combine the results of elc. cost in S1 and S2 together - For manuscript

# Set plot size
fig, ax = plt.subplots(figsize=(10, 6))

# Feed data
date_history = df_elc_3y['DATE'].to_list()
data_history = df_elc_3y['History'].to_list()

lower_data = prophet_preds_elc_3y['S2_elc_price_lower'].to_list()
n_rest_elm = len(prophet_preds_elc_nw['S2_elc_price_lower'].to_list()) - len(lower_data)
upper_data = prophet_preds_elc_3y['S2_elc_price_upper'].to_list()
pred_data = prophet_preds_elc_3y['S2_elc_price_mean'].to_list()

date_pred = prophet_preds_elc_nw['DATE'].to_list()
lower_data.extend(prophet_preds_elc_nw['S2_elc_price_lower'].to_list()[len(lower_data):])
upper_data.extend(prophet_preds_elc_nw['S2_elc_price_upper'].to_list()[len(upper_data):])
pred_data.extend(prophet_preds_elc_nw['S2_elc_price_mean'].to_list()[len(pred_data):])

# Set color
ax.plot(date_history, data_history, label='History', color='black')
ax.plot(date_pred, lower_data, color=(28/255, 144/255, 153/255), linestyle='dotted', label='S2 Lowest price')
ax.plot(date_pred, prophet_preds_elc['S1_elc_price_lower'].to_list(), color=(28/255, 144/255, 153/255), linestyle='--', label='S1 Lowest price')
ax.plot(date_pred, upper_data, color=(217/255, 95/255, 14/255), linestyle='dotted', label='S2 Highest price')
ax.plot(date_pred, prophet_preds_elc['S1_elc_price_upper'].to_list(), color=(217/255, 95/255, 14/255), linestyle='--', label='S1 Highest price')
ax.plot(date_pred, pred_data, color=(136/255, 86/255, 167/255), linestyle='dotted', label='S2 Mean price')
ax.plot(date_pred, prophet_preds_elc['S1_elc_price_mean'].to_list(), color=(136/255, 86/255, 167/255), linestyle='--', label='S1 Mean price')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('€/kWh', fontsize=12)

ax.legend(fontsize=10)

plt.grid(True)
plt.tight_layout()
plt.show()


# %% [cell 19]
# Prophet - water cost prediction

# Insert historical data for water
df = pd.DataFrame(water_data)

# Run package Prophet for water cost prediction (here only incl. "Water fee"(WF) and "Waste water fee"(WWF))
prophet_WF_hat = prophet_fit_predict_water(odf=df, var="WF", years=20, plot=False).iloc[: , [2, 3, 15]]
prophet_WWF_hat = prophet_fit_predict_water(odf=df, var="WWF", years=20, plot=False).iloc[: , [2, 3, 15]]

# - if followed element of water needed to be considered... - 
# Value added tax: prophet_VAT_hat = prophet_fit_predict_material(odf=df, var="VAT", years=20, plot=False).iloc[: , [2, 3, 15]]

# Results of water cost prediction (plot + table)
prophet_preds_wat = reduce(lambda a, b: a.add(b, fill_value=0), [prophet_WF_hat,
                                                                 prophet_WWF_hat 
                                                                 ])

# Set the interval for time-series prediction
# Ground-truth: previously given data
df['History'] = df.sum(axis=1, numeric_only=True)

# Sum all ground-truth data and predictions
prophet_preds_wat['DATE'] = pd.date_range(start="2025-07-31", end="2045-07-31", freq="6ME")
prophet_preds_wat.rename(columns = {'yhat':'wat_price_mean', 'yhat_lower':'wat_price_lower',
                                    'yhat_upper':'wat_price_upper'}, inplace = True)

# Create subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Plot ground truth data
ax.plot(df['DATE'], df['History'], label='History', linestyle='-', linewidth=2, color='black')

# Plot predictions
ax.plot(prophet_preds_wat['DATE'], prophet_preds_wat['wat_price_mean'], label='Mean water price', linestyle='--', linewidth=2, color=(136/255, 86/255, 167/255))
ax.plot(prophet_preds_wat['DATE'], prophet_preds_wat['wat_price_upper'], label='Highest water price', linestyle='--', linewidth=2, color=(217/255, 95/255, 14/255))
ax.plot(prophet_preds_wat['DATE'], prophet_preds_wat['wat_price_lower'], label='Lowest water price', linestyle='--', linewidth=2, color=(28/255, 144/255, 153/255))

# Add labels for x and y-axis
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('€/kg', fontsize=12)

# Set x-axis ticks every 5 years starting from 2005
ax.xaxis.set_major_locator(mdates.YearLocator(base=5, month=1, day=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

#ax.set_title('Water Cost Prediction', fontsize=14)
ax.legend(fontsize=10)

# Show plot
plt.grid(True)
plt.tight_layout()
plt.show()

# Show table
prophet_preds_wat.head(50)

# %% [cell 20]
# OPEX S1 - resource cost (water and electricity) for 5 MW PEMWEWE
# 🔴 GEÄNDERT: original OPEX section retained; PEMWE electricity demand updated to 50.0 kWh/kg H2.

h2_produced_amount = [444911.5105]*41
h2_unit_elc = ['kg']*41
elc_used_amount = [50.0]*41
wat_used_amount = [9.3]*41
koh_used_amount = [0]*41
koh_price = [0]*41
elc_price_unit = ['€/kWh']*41
wat_price_unit = ['€/kg']*41
koh_price_unit = ['€/kg']*41

# Provide the names of the categories in the table
df = pd.DataFrame({'H2 Production' : h2_produced_amount,
                   'H2 Unit':  h2_unit_elc,
                   'Needed Water': wat_used_amount,
                   'Water Price Unit':wat_price_unit,
                   'Needed Electricity': elc_used_amount,
                   'Electricity Price Unit': elc_price_unit,
                   'Needed KOH': koh_used_amount,
                   'KOH Price': koh_price,
                   'KOH Price Unit': koh_price_unit,
                    })

cols_order = prophet_preds_wat.columns.to_list()
n_prophet_preds_wat = prophet_preds_wat.drop(prophet_preds_wat.columns[cols_order.index('DATE')], axis=1)
df_global = pd.concat([df, n_prophet_preds_wat, prophet_preds_elc], axis=1)

# Provide the calculation method within the table
prefix = 'S1'
df_global[f'{prefix} Water Price_Lower'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed Water'], df_global['wat_price_lower'])]
df_global[f'{prefix} Water Price_Higher'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed Water'], df_global['wat_price_upper'])]
df_global[f'{prefix} Water Price_Mean'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed Water'], df_global['wat_price_mean'])]
df_global[f'{prefix} Electricity Price_Lower'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed Electricity'], df_global[f'{prefix}_elc_price_lower'])]
df_global[f'{prefix} Electricity Price_Higher'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed Electricity'], df_global[f'{prefix}_elc_price_upper'])]
df_global[f'{prefix} Electricity Price_Mean'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed Electricity'], df_global[f'{prefix}_elc_price_mean'])]
df_global[f'{prefix} KOH Price'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed KOH'], df_global['KOH Price'])]
df_global[f'{prefix} Total Resources Price Lower'] = [a+b+c for a, b, c in zip(df_global[f'{prefix} Electricity Price_Lower'], df_global[f'{prefix} Water Price_Lower'], df_global[f'{prefix} KOH Price'])]
df_global[f'{prefix} Total Resources Price Higher'] = [a+b+c for a, b, c in zip(df_global[f'{prefix} Electricity Price_Higher'], df_global[f'{prefix} Water Price_Higher'], df_global[f'{prefix} KOH Price'])]
df_global[f'{prefix} Total Resources Price Mean'] = [a+b+c for a, b, c in zip(df_global[f'{prefix} Electricity Price_Mean'], df_global[f'{prefix} Water Price_Mean'], df_global[f'{prefix} KOH Price'])]

total_row = {f'{prefix} Electricity Price_Mean': df_global[f'{prefix} Electricity Price_Mean'].sum(),
             f'{prefix} Electricity Price_Higher': df_global[f'{prefix} Electricity Price_Higher'].sum(),
             f'{prefix} Electricity Price_Lower': df_global[f'{prefix} Electricity Price_Lower'].sum(),
             f'{prefix} Water Price_Mean': df_global[f'{prefix} Water Price_Mean'].sum(),
             f'{prefix} Water Price_Higher': df_global[f'{prefix} Water Price_Higher'].sum(),
             f'{prefix} Water Price_Lower': df_global[f'{prefix} Water Price_Lower'].sum(),
             f'{prefix} KOH Price': df_global[f'{prefix} KOH Price'].sum(),
             f'{prefix} Total Resources Price Lower': df_global[f'{prefix} Total Resources Price Lower'].sum(),
             f'{prefix} Total Resources Price Higher': df_global[f'{prefix} Total Resources Price Higher'].sum(),
             f'{prefix} Total Resources Price Mean': df_global[f'{prefix} Total Resources Price Mean'].sum()}
df_global = df_global.append(total_row, ignore_index=True)
as_list = df_global.index.tolist()
as_list[-1] = 'Total'
df_global.index = as_list
df_global.fillna('', inplace=True)

df_global.head()


# %% [cell 21]
# OPEX S1 (with EoL cost) - Details
# S1_mean

# OPEX S1 (Mean) is calculated here based on all the parameters listed below
transport_cost_disposal = transport_cost_disposal
salvage_value = salvage_value

kwargs={
        'equity' : 0.25, 
        'eq_rate_return': 0.07,
        'debt': 0.75, 
        'db_intrst_rate': 0.045,
        'construction_cost' : material_cost,
        'percentage_of_cc' : 0.06,
        'per_cc' : 0.025,
        'transport_cost_disposal' : transport_cost_disposal, 
        'salvage_value' : salvage_value, 
        'inflation_rate' : 0.01
        }

S1_resource_cost_mean = df_global['S1 Total Resources Price Mean'].tolist()

op_cost_S1_mean = operation_cost(resource_cost=S1_resource_cost_mean, maintenance_cost=3750, full=True, **kwargs)
print(f"Operation Cost: {op_cost_S1_mean:.2f}")

# %% [cell 22]
# OPEX S1 (with EoL cost) - Details
# S1_higher

# OPEX S1 (High) is calculated here based on all the parameters listed below
transport_cost_disposal = transport_cost_disposal
salvage_value = salvage_value

kwargs={
        'equity' : 0.25, 
        'eq_rate_return': 0.07,
        'debt': 0.75, 
        'db_intrst_rate': 0.045,
        'construction_cost' : material_cost,
        'percentage_of_cc' : 0.06,
        'per_cc' : 0.025,
        'transport_cost_disposal' : transport_cost_disposal, 
        'salvage_value' : salvage_value, 
        'inflation_rate' : 0.01
        }

S1_resource_cost_high = df_global['S1 Total Resources Price Higher'].tolist()

op_cost_S1_high = operation_cost(resource_cost=S1_resource_cost_high, maintenance_cost=3750, full=True, **kwargs)
print(f"Operation Cost: {op_cost_S1_high:.2f}")

# %% [cell 23]
# OPEX S1 (with EoL cost) - Details
# S1_lower

# OPEX S1 (Low) is calculated here based on all the parameters listed below
transport_cost_disposal = transport_cost_disposal
salvage_value = salvage_value

kwargs={
        'equity' : 0.25, 
        'eq_rate_return': 0.07,
        'debt': 0.75, 
        'db_intrst_rate': 0.045,
        'construction_cost' : material_cost,
        'percentage_of_cc' : 0.06,
        'per_cc' : 0.025,
        'transport_cost_disposal' : transport_cost_disposal, 
        'salvage_value' : salvage_value, 
        'inflation_rate' : 0.01
        }

S1_resource_cost_low = df_global['S1 Total Resources Price Lower'].tolist()

op_cost_S1_low = operation_cost(resource_cost=S1_resource_cost_low, maintenance_cost=3750, full=True, **kwargs)
print(f"Operation Cost: {op_cost_S1_low:.2f}")

# %% [cell 24]
# Monaco OPEX S1 - Price range and sensitivity ratio

# Function for running the monte carlo simulation
def run(op_cost, op_labour, eol_cost):

    # run the opex simulation
    opr_cost = monte_opex(op_cost=op_cost, op_labour=op_labour, eol_cost=eol_cost)
    return (opr_cost, )

# Rename the y-aixs in results
def preprocess(case):

    # The 'preprocess' function grabs the random input values for each case and structures it with any other data in the format the 'run' function expects
    # Preprocess function for extracting usable variables from input dataframes
    op_cost = case.invals['Resources cost'].val
    op_labour = case.invals['OPEX labour cost'].val
    eol_cost = case.invals['EoL cost'].val

    return (op_cost, op_labour, eol_cost)

# The 'postprocess' function takes the output from the 'run' function and saves off the outputs for each case
def postprocess(case, opr_cost):
    case.addOutVal(name='OPEX_S1', val=opr_cost)
    case.addOutVal(name='OPEX1_case', val=case.ncase)
    return None

# Argument list of functions used in simulator function
fcns = {'run' : run,
        'preprocess' : preprocess,
        'postprocess': postprocess}

# Number of iterations in the monte carlo simulation
n_invests = 350 
# Random seed for reproducibility
seed = 123456 
# Usage of monaco simulator with the SOBOL Random function
sim = mc.Sim(name='OPEX', ndraws=n_invests, 
            fcns=fcns, seed=seed, 
            samplemethod='sobol_random',
            singlethreaded=True,
            savecasedata=False,
            verbose=True, debug=True)

# Variables used in the simulation
# loc: represents the lower limit of the distribution
# scale: the interval length of the distribution
# e.g: using loc=0 and scale=1 we get a standard uniform distribution on [0, 1]
# By manually change loc and scale can manage the uncertainty of the cost
sim.addInVar(name='Resources cost', dist=uniform, distkwargs={'loc': 58000000, 'scale': 23500000})
sim.addInVar(name='OPEX labour cost', dist=uniform, distkwargs={'loc': 45000, 'scale': 10000})
sim.addInVar(name='EoL cost', dist=uniform, distkwargs={'loc': 50000, 'scale': 2000})

# Run the Simulation
sim.runSim()

# Calculate the mean and 5-95th percentile
# statistics for the dice sum
sim.outvars['OPEX_S1'].addVarStat('mean')
sim.outvars['OPEX_S1'].addVarStat('percentile', {'p':[0.05, 0.95]})

# Plots a histogram of the dice sum
mc.plot(sim.outvars['OPEX_S1'])
# Creates a scatter plot of the sum vs the roll
# number, showing randomness
mc.plot(sim.outvars['OPEX_S1'],
sim.outvars['OPEX1_case'])

# add Mio.€ in
plt.xlabel('10Mio.€')

# Calculate the sensitivity of the dice sum to
# each of the input variables
sim.calcSensitivities('OPEX_S1')
sim.outvars['OPEX_S1'].plotSensitivities()

# %% [cell 25]
# OPEX S2 - resource cost (water and electricity) for 5 MW PEMWEWE
# 🔴 GEÄNDERT: original OPEX section retained; PEMWE electricity demand updated to 50.0 kWh/kg H2.

h2_produced_amount = [444911.5105]*41
h2_unit_elc = ['kg']*41
elc_used_amount = [50.0]*41
wat_used_amount = [9.3]*41
koh_used_amount = [0]*41
koh_price = [0]*41
elc_price_unit = ['€/kWh']*41
wat_price_unit = ['€/kg']*41
koh_price_unit = ['€/kg']*41

# Provide the names of the categories in the table
df = pd.DataFrame({'H2 Production' : h2_produced_amount,
                   'H2 Unit':  h2_unit_elc,
                   'Needed Water': wat_used_amount,
                   'Water Price Unit':wat_price_unit,
                   'Needed Electricity': elc_used_amount,
                   'Electricity Price Unit': elc_price_unit,
                   'Needed KOH': koh_used_amount,
                   'KOH Price': koh_price,
                   'KOH Price Unit': koh_price_unit,
                    })

cols_order = prophet_preds_wat.columns.to_list()
n_prophet_preds_wat = prophet_preds_wat.drop(prophet_preds_wat.columns[cols_order.index('DATE')], axis=1)
df_global = pd.concat([df, n_prophet_preds_wat, prophet_elc_price], axis=1)

# Provide the calculation method within the table
prefix = 'S2'
df_global[f'{prefix} Water Price_Lower'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed Water'], df_global['wat_price_lower'])]
df_global[f'{prefix} Water Price_Higher'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed Water'], df_global['wat_price_upper'])]
df_global[f'{prefix} Water Price_Mean'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed Water'], df_global['wat_price_mean'])]
df_global[f'{prefix} Electricity Price_Lower'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed Electricity'], df_global[f'{prefix}_elc_price_lower'])]
df_global[f'{prefix} Electricity Price_Higher'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed Electricity'], df_global[f'{prefix}_elc_price_upper'])]
df_global[f'{prefix} Electricity Price_Mean'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed Electricity'], df_global[f'{prefix}_elc_price_mean'])]
df_global[f'{prefix} KOH Price'] = [a*b*c for a, b, c in zip(df_global['H2 Production'], df_global['Needed KOH'], df_global['KOH Price'])]
df_global[f'{prefix} Total Resources Price Lower'] = [a+b+c for a, b, c in zip(df_global[f'{prefix} Electricity Price_Lower'], df_global[f'{prefix} Water Price_Lower'], df_global[f'{prefix} KOH Price'])]
df_global[f'{prefix} Total Resources Price Higher'] = [a+b+c for a, b, c in zip(df_global[f'{prefix} Electricity Price_Higher'], df_global[f'{prefix} Water Price_Higher'], df_global[f'{prefix} KOH Price'])]
df_global[f'{prefix} Total Resources Price Mean'] = [a+b+c for a, b, c in zip(df_global[f'{prefix} Electricity Price_Mean'], df_global[f'{prefix} Water Price_Mean'], df_global[f'{prefix} KOH Price'])]

total_row = {f'{prefix} Electricity Price_Mean': df_global[f'{prefix} Electricity Price_Mean'].sum(),
             f'{prefix} Electricity Price_Higher': df_global[f'{prefix} Electricity Price_Higher'].sum(),
             f'{prefix} Electricity Price_Lower': df_global[f'{prefix} Electricity Price_Lower'].sum(),
             f'{prefix} Water Price_Mean': df_global[f'{prefix} Water Price_Mean'].sum(),
             f'{prefix} Water Price_Higher': df_global[f'{prefix} Water Price_Higher'].sum(),
             f'{prefix} Water Price_Lower': df_global[f'{prefix} Water Price_Lower'].sum(),
             f'{prefix} KOH Price': df_global[f'{prefix} KOH Price'].sum(),
             f'{prefix} Total Resources Price Lower': df_global[f'{prefix} Total Resources Price Lower'].sum(),
             f'{prefix} Total Resources Price Higher': df_global[f'{prefix} Total Resources Price Higher'].sum(),
             f'{prefix} Total Resources Price Mean': df_global[f'{prefix} Total Resources Price Mean'].sum()}
df_global = df_global.append(total_row, ignore_index=True)
as_list = df_global.index.tolist()
as_list[-1] = 'Total'
df_global.index = as_list
df_global.fillna('', inplace=True)

df_global.head()


# %% [cell 26]
# OPEX S2 (with EoL cost) - Details
# S2_mean

# OPEX S2 (Mean) is calculated here based on all the parameters listed below
transport_cost_disposal = transport_cost_disposal
salvage_value = salvage_value

kwargs={
        'equity' : 0.25, 
        'eq_rate_return': 0.07,
        'debt': 0.75, 
        'db_intrst_rate': 0.045,
        'construction_cost' : material_cost,
        'percentage_of_cc' : 0.06,
        'per_cc' : 0.025,
        'transport_cost_disposal' : transport_cost_disposal, 
        'salvage_value' : salvage_value, 
        'inflation_rate' : 0.01
        }

S2_resource_cost_mean = df_global['S2 Total Resources Price Mean'].tolist()

op_cost_S2_mean = operation_cost(resource_cost=S2_resource_cost_mean, maintenance_cost=3750, full=True, **kwargs)
print(f"Operation Cost: {op_cost_S2_mean:.2f}")

# %% [cell 27]
# OPEX S2 (with EoL cost) - Details
# S2_higher

# OPEX S2 (High) is calculated here based on all the parameters listed below
transport_cost_disposal = transport_cost_disposal
salvage_value = salvage_value

kwargs={
        'equity' : 0.25, 
        'eq_rate_return': 0.07,
        'debt': 0.75, 
        'db_intrst_rate': 0.045,
        'construction_cost' : material_cost,
        'percentage_of_cc' : 0.06,
        'per_cc' : 0.025,
        'transport_cost_disposal' : transport_cost_disposal, 
        'salvage_value' : salvage_value, 
        'inflation_rate' : 0.01
        }

S2_resource_cost_high = df_global['S2 Total Resources Price Higher'].tolist()

op_cost_S2_high = operation_cost(resource_cost=S2_resource_cost_high, maintenance_cost=3750, full=True, **kwargs)
print(f"Operation Cost: {op_cost_S2_high:.2f}")

# %% [cell 28]
# OPEX S2 (with EoL cost) - Details
# S2_lower

# OPEX S2 (Low) is calculated here based on all the parameters listed below
transport_cost_disposal = transport_cost_disposal
salvage_value = salvage_value

kwargs={
        'equity' : 0.25, 
        'eq_rate_return': 0.07,
        'debt': 0.75, 
        'db_intrst_rate': 0.045,
        'construction_cost' : material_cost,
        'percentage_of_cc' : 0.06,
        'per_cc' : 0.025,
        'transport_cost_disposal' : transport_cost_disposal, 
        'salvage_value' : salvage_value, 
        'inflation_rate' : 0.01
        }

S2_resource_cost_low = df_global['S2 Total Resources Price Lower'].tolist()

op_cost_S2_low = operation_cost(resource_cost=S2_resource_cost_low, maintenance_cost=3750, full=True, **kwargs)
print(f"Operation Cost: {op_cost_S2_low:.2f}")

# %% [cell 29]
# Monaco OPEX S2 - Price range and sensitivity ratio

# Function for running the monte carlo simulation
def run(op_cost, op_labour, eol_cost):

       # run the opex simulation
       opr_cost = monte_opex(op_cost=op_cost, op_labour=op_labour, eol_cost=eol_cost)
       return (opr_cost, )


# Rename the y-aixs in results
def preprocess(case):

    # The 'preprocess' function grabs the random input values for each case and structures it with any other data in the format the 'run' function expects
    # Preprocess function for extracting usable variables from input dataframes
    op_cost = case.invals['Resources cost'].val
    op_labour = case.invals['OPEX labour cost'].val
    eol_cost = case.invals['EoL cost'].val

    return (op_cost, op_labour, eol_cost)

# The 'postprocess' function takes the output from the 'run' function and saves off the outputs for each case
def postprocess(case, opr_cost):
    case.addOutVal(name='OPEX_S2', val=opr_cost)
    case.addOutVal(name='OPEX2_case', val=case.ncase)
    return None

# Argument list of functions used in simulator function
fcns = {'run' : run,
        'preprocess' : preprocess,
        'postprocess': postprocess}

# Number of iterations in the monte carlo simulation
n_invests = 350 
# Random seed for reproducibility
seed = 123456 
# Usage of monaco simulator with the SOBOL Random function
sim = mc.Sim(name='OPEX', ndraws=n_invests, 
            fcns=fcns, seed=seed, 
            samplemethod='sobol_random',
            singlethreaded=True,
            savecasedata=False,
            verbose=True, debug=True)

# Variables used in the simulation
# By manually change loc and scale can manage the uncertainty of the cost
sim.addInVar(name='Resources cost', dist=uniform, distkwargs={'loc': 48500000, 'scale': 11580000})
sim.addInVar(name='OPEX labour cost', dist=uniform, distkwargs={'loc': 45000, 'scale': 10000})
sim.addInVar(name='EoL cost', dist=uniform, distkwargs={'loc': 50000, 'scale': 2000})


# Run the Simulation
sim.runSim()

# Calculate the mean and 5-95th percentile
# statistics for the dice sum
sim.outvars['OPEX_S2'].addVarStat('mean')
sim.outvars['OPEX_S2'].addVarStat('percentile', {'p':[0.05, 0.95]})
# Plots a histogram of the dice sum
mc.plot(sim.outvars['OPEX_S2'])
# Creates a scatter plot of the sum vs the roll
# number, showing randomness
mc.plot(sim.outvars['OPEX_S2'],
sim.outvars['OPEX2_case'])

# add Mio.€ in
plt.xlabel('10Mio.€')

# Calculate the sensitivity of the dice sum to each of the input variables
sim.calcSensitivities('OPEX_S2')
sim.outvars['OPEX_S2'].plotSensitivities()

# %% [cell 30]
# If:
#    resource cost is incl. taxes;
#    depreciation cost is considered;
#    there is any recoverable taxes was taking into account.
# Then: 
#    Tax impacts

# ---- Tax Impacts ----

# Feed needed data in
#kwargs={
#        'equity' : 0.25, 
#        'eq_rate_return': 0.07,
#        'debt': 0.75, 
#        'db_intrst_rate': 0.045,
#        'inflation_rate' : 0.01,
#        }

# Extract resource_cost from OPEX
#resource_cost = df_global['Total Resources Price'].tolist()

#print(resource_cost)

#tax_impct = tax_impact(resource_cost=0, tax_rate=0.3, int_on_debt=0.045, depreciation=0, **kwargs)
#print(f"Tax Impact: {tax_impct:.2f}")

# %% [cell 31]
# Life Cycle Cost S1 - Total cost of ownership (TCO)
# with OPEX_S1_mean

# TCO S1 (Mean) is calculated here based on all the parameters listed below
lcc_S1_mean = life_cycle_cost_of_h2(investment_cost=inv_cost, op_cost=op_cost_S1_mean, tax_impact=0)
print(f"Life Cycle Cost: {lcc_S1_mean:.2f}")

# %% [cell 32]
# Life Cycle Cost S1 - Total cost of ownership
# with OPEX_S1_higher

# TCO S1 (High) is calculated here based on all the parameters listed below
lcc_S1_high = life_cycle_cost_of_h2(investment_cost=inv_cost, op_cost=op_cost_S1_high, tax_impact=0)
print(f"Life Cycle Cost: {lcc_S1_high:.2f}")

# %% [cell 33]
# Life Cycle Cost S1 - Total cost of ownership
# with OPEX_S1_lower

# TCO S1 (Low) is calculated here based on all the parameters listed below
lcc_S1_low = life_cycle_cost_of_h2(investment_cost=inv_cost, op_cost=op_cost_S1_low, tax_impact=0)
print(f"Life Cycle Cost: {lcc_S1_low:.2f}")

# %% [cell 34]
# Life Cycle Cost S2 - Total cost of ownership
# with OPEX_S2_mean

# TCO S2 (Mean) is calculated here based on all the parameters listed below
lcc_S2_mean = life_cycle_cost_of_h2(investment_cost=inv_cost, op_cost=op_cost_S2_mean, tax_impact=0)
print(f"Life Cycle Cost: {lcc_S2_mean:.2f}")

# %% [cell 35]
# Life Cycle Cost S2 - Total cost of ownership
# with OPEX_S2_higher

# TCO S2 (High) is calculated here based on all the parameters listed below
lcc_S2_high = life_cycle_cost_of_h2(investment_cost=inv_cost, op_cost=op_cost_S2_high, tax_impact=0)
print(f"Life Cycle Cost: {lcc_S2_high:.2f}")

# %% [cell 36]
# Life Cycle Cost S2 - Total cost of ownership
# with OPEX_S2_lower

# TCO S2 (Low) is calculated here based on all the parameters listed below
lcc_S2_low = life_cycle_cost_of_h2(investment_cost=inv_cost, op_cost=op_cost_S2_low, tax_impact=0)
print(f"Life Cycle Cost: {lcc_S2_low:.2f}")

# %% [cell 37]
# Monaco TCO S1 - Price range and sensitivity ratio

# Function for running the monte carlo simulation
def run(investment_cost, opr_cost, tax_impact):

        # run the lcc simulation
        lcc_cost = monte_lcc(investment_cost=investment_cost, opr_cost=opr_cost, tax_impact=tax_impact)
        return (lcc_cost, )

 # The 'preprocess' function grabs the random input values for each case and structures it with any other data in the format the 'run' function expects
    # Preprocess function for extracting usable variables from input dataframes
def preprocess(case):
    investment_cost = case.invals['CAPEX'].val
    opr_cost = case.invals['OPEX'].val
    tax_impact = case. invals['Tax impact'].val

    return (investment_cost, opr_cost, tax_impact)

# The 'postprocess' function takes the output from the 'run' function and saves off the outputs for each case
def postprocess(case, lcc_cost):
    case.addOutVal(name='TCO_S1', val=lcc_cost)
    case.addOutVal(name='TCO1_case', val=case.ncase)
    return None

# Argument list of functions used in simulator function
fcns = {'run' : run,
        'preprocess' : preprocess,
        'postprocess': postprocess}

# Number of iterations in the monte carlo simulation
n_invests = 350 # Arbitrary for this example
# Random seed for reproducibility
seed = 123456 # Recommended for repeatability
# Usage of monaco simulator with the SOBOL Random function
sim = mc.Sim(name='Life Cycle Cost of H2_wos', ndraws=n_invests, 
            fcns=fcns, seed=seed, 
            samplemethod='sobol_random',
            singlethreaded=True,
            savecasedata=False,
            verbose=True, debug=True)

# Variables used in the simulation
# By manually change loc and scale can manage the uncertainty of the cost
sim.addInVar(name='CAPEX', dist=uniform, distkwargs={'loc': 2200000, 'scale': 400000})
sim.addInVar(name='OPEX', dist=uniform, distkwargs={'loc': 58000000, 'scale': 23500000})
sim.addInVar(name='Tax impact', dist=uniform, distkwargs={'loc': 0, 'scale': 1})

# Run the Simulation
sim.runSim()

# Calculate the mean and 5-95th percentile
# Statistics for the dice sum
sim.outvars['TCO_S1'].addVarStat('mean')
sim.outvars['TCO_S1'].addVarStat('percentile', {'p':[0.05, 0.95]})

# Plots a histogram of the dice sum
mc.plot(sim.outvars['TCO_S1'])

# Creates a scatter plot of the sum vs the roll
# Number, showing randomness
mc.plot(sim.outvars['TCO_S1'],
sim.outvars['TCO1_case'])
plt.xlabel('10Mio €')

# Calculate the sensitivity of the dice sum to each of the input variables
sim.calcSensitivities('TCO_S1')
sim.outvars['TCO_S1'].plotSensitivities()

# %% [cell 38]
# Monaco TCO S2 - Price range and sensitivity ratio

# Function for running the monte carlo simulation
def run(investment_cost, opr_cost, tax_impact):

    # run the lcc simulation
    lcc_cost = monte_lcc(investment_cost=investment_cost, opr_cost=opr_cost, tax_impact=tax_impact)
    return (lcc_cost, )

# Rename the y-aixs in results
def preprocess(case):
    
    # The 'preprocess' function grabs the random input values for each case and structures it with any other data in the format the 'run' function expects
    # Preprocess function for extracting usable variables from input dataframes
    investment_cost = case.invals['CAPEX'].val
    opr_cost = case.invals['OPEX'].val
    tax_impact = case. invals['Tax impact'].val

    return (investment_cost, opr_cost, tax_impact)

# The 'postprocess' function takes the output from the 'run' function and saves off the outputs for each case
def postprocess(case, lcc_cost):
    case.addOutVal(name='TCO_S2', val=lcc_cost)
    case.addOutVal(name='TCO2_case', val=case.ncase)
    return None

# Argument list of functions used in simulator function
fcns = {'run' : run,
        'preprocess' : preprocess,
        'postprocess': postprocess}

# Number of iterations in the monte carlo simulation
n_invests = 350 # Arbitrary for this example
# Random seed for reproducibility
seed = 123456 # Recommended for repeatability
# Usage of monaco simulator with the SOBOL Random function
sim = mc.Sim(name='Life Cycle Cost of H2_wos', ndraws=n_invests, 
            fcns=fcns, seed=seed, 
            samplemethod='sobol_random',
            singlethreaded=True,
            savecasedata=False,
            verbose=True, debug=True)

# Variables used in the simulation
# By manually change loc and scale can manage the uncertainty of the cost
sim.addInVar(name='CAPEX', dist=uniform, distkwargs={'loc': 2200000, 'scale': 400000})
sim.addInVar(name='OPEX', dist=uniform, distkwargs={'loc': 49000000, 'scale': 11900000})
sim.addInVar(name='Tax impact', dist=uniform, distkwargs={'loc': 0, 'scale': 1})

# Run the Simulation
sim.runSim()

# Calculate the mean and 5-95th percentile
# Statistics for the dice sum
sim.outvars['TCO_S2'].addVarStat('mean')
sim.outvars['TCO_S2'].addVarStat('percentile', {'p':[0.05, 0.95]})

# Plots a histogram of the dice sum
mc.plot(sim.outvars['TCO_S2'])

# Creates a scatter plot of the sum vs the roll
# Number, showing randomness
mc.plot(sim.outvars['TCO_S2'],
sim.outvars['TCO2_case'])
plt.xlabel('10Mio €')

# Calculate the sensitivity of the dice sum to each of the input variables
sim.calcSensitivities('TCO_S2')
sim.outvars['TCO_S2'].plotSensitivities()

# %% [cell 39]
# Box plot - Results of CAPEX, OPEX and TCO - for manuscript

# Calcualted data from previous process
# p5 and p95 refer to the 5th and 95th percentiles respectively
# We use the p5 and p95 to plot a boxplot
# 🔴 GEÄNDERT: hard-coded plot summary values updated to new technology inputs
capex = {'min': 3.1, 'max': 4.2, 'median': 3.65, 'p5': 3.2, 'p95': 4.1}
opex_s1 = {'min': 55, 'max': 80, 'median': 67, 'p5': 57, 'p95': 78}
opex_s2 = {'min': 46, 'max': 60, 'median': 53, 'p5': 48, 'p95': 58}
tco_s1 = {'min': 58, 'max': 84, 'median': 71, 'p5': 60, 'p95': 82}
tco_s2 = {'min': 49, 'max': 64, 'median': 57, 'p5': 51, 'p95': 62}

# Draw the boxplot
plt.figure(figsize=(10, 6))

# Plotting the boxplot using the data defined above
bp = plt.boxplot(
    [[capex['min'], capex['p5'], capex['median'], capex['p95'], capex['max']],
    [opex_s1['min'], opex_s1['p5'], opex_s1['median'], opex_s1['p95'], opex_s1['max']],
    [tco_s1['min'], tco_s1['p5'], tco_s1['median'], tco_s1['p95'], tco_s1['max']],
    [opex_s2['min'], opex_s2['p5'], opex_s2['median'], opex_s2['p95'], opex_s2['max']],
    [tco_s2['min'], tco_s2['p5'], tco_s2['median'], tco_s2['p95'], tco_s2['max']]],
            labels=['CAPEX', 'OPEX S1', 'TCO S1', 'OPEX S2', 'TCO S2'], vert=False, patch_artist=True)

# Set color
colors = [(49/255, 163/255, 84/255), (249/255, 252/255, 185/255), (36/255, 134/255, 185/255), (249/255, 252/255, 185/255), (36/255, 134/255, 185/255)]
for box, color in zip(bp['boxes'], colors):
    box.set(facecolor=color)

legend_elements = [Patch(facecolor=(49/255, 163/255, 84/255), edgecolor='black', label='CAPEX'),
                   Patch(facecolor=(249/255, 252/255, 185/255), edgecolor='black', label='OPEX'),
                   Patch(facecolor=(36/255, 134/255, 185/255), edgecolor='black', label='TCO'),]

plt.legend(handles=legend_elements, loc='upper left')

# Define x-axis
plt.xlabel('Mio.€')

# Set y-axis
plt.xlim(0) 

plt.yticks([1.5, 3.5], ['', ''])

# Set lengend
plt.text(0, 1, 'CAPEX  ', ha='right', va='center')
plt.text(0, 2.5, 'Scenario 1  ', ha='right', va='center')
plt.text(0, 4.5, 'Scenario 2  ', ha='right', va='center')

# Print plot
plt.grid(True)
plt.show()

# %% [cell 40]
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Calculated data
# 🔴 GEÄNDERT: hard-coded plot summary values updated to new technology inputs
capex = {'min': 3.1, 'max': 4.2, 'median': 3.65, 'p5': 3.2, 'p95': 4.1}
opex_s1 = {'min': 55, 'max': 80, 'median': 67, 'p5': 57, 'p95': 78}
opex_s2 = {'min': 46, 'max': 60, 'median': 53, 'p5': 48, 'p95': 58}
tco_s1 = {'min': 58, 'max': 84, 'median': 71, 'p5': 60, 'p95': 82}
tco_s2 = {'min': 49, 'max': 64, 'median': 57, 'p5': 51, 'p95': 62}

# Create a figure and a grid of subplots
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
fig.subplots_adjust(wspace=0.05)

# Data for the box plots
data1 = [
    [capex['min'], capex['p5'], capex['median'], capex['p95'], capex['max']],
    [],
    [],
    [],
    []
]
data2 = [
    [],
    [opex_s1['min'], opex_s1['p5'], opex_s1['median'], opex_s1['p95'], opex_s1['max']],
    [tco_s1['min'], tco_s1['p5'], tco_s1['median'], tco_s1['p95'], tco_s1['max']],
    [opex_s2['min'], opex_s2['p5'], opex_s2['median'], opex_s2['p95'], opex_s2['max']],
    [tco_s2['min'], tco_s2['p5'], tco_s2['median'], tco_s2['p95'], tco_s2['max']]
]

# Plotting the box plots on both axes with colors
bp1 = ax1.boxplot(data1, vert=False, patch_artist=True)
bp2 = ax2.boxplot(data2, vert=False, patch_artist=True)

# Setting colors for the boxes in ax1
colors1 = [(80/255, 138/255, 178/255)] * 5  # CAPEX color for all boxes
for box, color in zip(bp1['boxes'], colors1):
    box.set(facecolor=color)

# Setting colors for the boxes in ax2
colors2 = [(255/255, 255/255, 255/255), (255/255, 208/255, 111/255), (113/255, 159/255, 133/255), (255/255, 208/255, 111/255), (113/255, 159/255, 133/255)]
for box, color in zip(bp2['boxes'], colors2):
    box.set(facecolor=color)

# Setting the limits for the x-axes
ax1.set_xlim(2, 2.7)
ax2.set_xlim(45, 90)

# Removing y-axis ticks and labels for both axes
ax1.yaxis.set_visible(False)
ax2.yaxis.set_visible(False)

# Removing the spines between the axes
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Adding diagonal lines to indicate the broken axis
d = .015  # size of the diagonal lines in axes coordinates
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal

# Adding horizontal lines between ax1 and ax2
line_y = 0.26  # y-coordinate for the line
line_color = 'gray'
line_style = '--'
fig.add_artist(plt.Line2D([0.12, 0.9], [line_y, line_y], transform=fig.transFigure, color=line_color, linestyle=line_style))

# Adding horizontal lines between ax2 and ax3
line_y = 0.57  # y-coordinate for the line
fig.add_artist(plt.Line2D([0.12, 0.9], [line_y, line_y], transform=fig.transFigure, color=line_color, linestyle=line_style))

# Set legend
legend_elements = [
    Patch(facecolor=(80/255, 138/255, 178/255), edgecolor='black', label='CAPEX'),
    Patch(facecolor=(255/255, 208/255, 111/255), edgecolor='black', label='OPEX'),
    Patch(facecolor=(113/255, 159/255, 133/255), edgecolor='black', label='TCO')
]
ax1.legend(handles=legend_elements, loc='upper left')

# Adding labels using fig.text() to position them relative to the figure
fig.text(0.5,0.04, 'Mio.€', ha='center', va='center', fontsize=12)

fig.text(0.12, 0.185, 'CAPEX', ha='right', va='center', fontsize=12)
fig.text(0.12, 0.44, 'Scenario 1', ha='right', va='center', fontsize=12)
fig.text(0.12, 0.73, 'Scenario 2', ha='right', va='center', fontsize=12)

# Show plot
ax1.grid(True)
ax2.grid(True)
plt.show()


# %% [cell 41]
fig.legend(handles=legend_elements, loc='upper left')


# %% [cell 42]
# Levelized cost of hydrogen (LCOH) S1
# with LCC_S1_mean

# LCC S1 (Mean) is calculated here based on all the parameters listed below
kwargs={
        'equity' : 0.25, 
        'eq_rate_return': 0.07,
        'debt': 0.75, 
        'db_intrst_rate': 0.045,
        'inflation_rate' : 0.01
        }


lcoh_s1_mean = levelized_cost_of_h2(life_cycle_cost_of_h2=lcc_S1_mean, h2_production=17796460, **kwargs)

# If:
#       detailed data is needed
# Then:
#       print(f"Levelized Cost: {lcoh_s1_mean:.2f}")

# %% [cell 43]
# Levelized cost of hydrogen (LCOH) S1
# with LCC_S1_higher

# LCC S1 (High) is calculated here based on all the parameters listed below
kwargs={
        'equity' : 0.25, 
        'eq_rate_return': 0.07,
        'debt': 0.75, 
        'db_intrst_rate': 0.045,
        'inflation_rate' : 0.01
        }


lcoh_s1_high = levelized_cost_of_h2(life_cycle_cost_of_h2=lcc_S1_high, h2_production=17796460, **kwargs)

# If:
#       detailed data is needed
# Then:
#       print(f"Levelized Cost: {lcoh_s1_high:.2f}")

# %% [cell 44]
# Levelized cost of hydrogen (LCOH) S1
# with LCC_S1_low

# LCC S1 (Low) is calculated here based on all the parameters listed below
kwargs={
        'equity' : 0.25, 
        'eq_rate_return': 0.07,
        'debt': 0.75, 
        'db_intrst_rate': 0.045,
        'inflation_rate' : 0.01
        }


lcoh_s1_low = levelized_cost_of_h2(life_cycle_cost_of_h2=lcc_S1_low, h2_production=17796460, **kwargs)

# If:
#       detailed data is needed
# Then:
#       print(f"Levelized Cost: {lcoh_s1_low:.2f}")

# %% [cell 45]
# LCOH S2
# with LCC_S2_mean

# LCC S2 (Mean) is calculated here based on all the parameters listed below
kwargs={
        'equity' : 0.25, 
        'eq_rate_return': 0.07,
        'debt': 0.75, 
        'db_intrst_rate': 0.045,
        'inflation_rate' : 0.01
        }


lcoh_s2_mean = levelized_cost_of_h2(life_cycle_cost_of_h2=lcc_S2_mean, h2_production=17796460, **kwargs)

# If:
#       detailed data is needed
# Then:
#       print(f"Levelized Cost: {lcoh_s2_mean:.2f}")

# %% [cell 46]
# LCOH S2
# with LCC_S2_higher

# LCC S2 (High) is calculated here based on all the parameters listed below
kwargs={
        'equity' : 0.25, 
        'eq_rate_return': 0.07,
        'debt': 0.75, 
        'db_intrst_rate': 0.045,
        'inflation_rate' : 0.01
        }


lcoh_s2_high = levelized_cost_of_h2(life_cycle_cost_of_h2=lcc_S2_high, h2_production=17796460, **kwargs)

# If:
#       detailed data is needed
# Then:
#       print(f"Levelized Cost: {lcoh_s2_high:.2f}")

# %% [cell 47]
# LCOH S2
# with LCC_S2_lower

# LCC S2 (Low) is calculated here based on all the parameters listed below
kwargs={
        'equity' : 0.25, 
        'eq_rate_return': 0.07,
        'debt': 0.75, 
        'db_intrst_rate': 0.045,
        'inflation_rate' : 0.01
        }


lcoh_s2_low = levelized_cost_of_h2(life_cycle_cost_of_h2=lcc_S2_low, h2_production=17796460, **kwargs)

# If:
#       detailed data is needed
# Then:
#       print(f"Levelized Cost: {lcoh_s2_low:.2f}")

# %% [cell 48]
# Monaco LCOH S1 - Price range and sensitivity ratio

# Function for running the monte carlo simulation
def run(life_cycle_cost_of_h2, h2_production, equity, eq_rate_return, debt, db_intrst_rate, inflation_rate):
    kwargs={
        'inflation_rate' : inflation_rate,
        'equity' : equity,
        'eq_rate_return' : eq_rate_return,
        'debt' : debt,
        'db_intrst_rate' : db_intrst_rate,
        }

    # run the lcoh simulation
    lcoh_cost = levelized_cost_of_h2(life_cycle_cost_of_h2=life_cycle_cost_of_h2, h2_production=h2_production, **kwargs)
    return (lcoh_cost, )

# The 'preprocess' function grabs the random input values for each case and structures it with any other data in the format the 'run' function expects
# Preprocess function for extracting usable variables from input dataframes
def preprocess(case):
    life_cycle_cost_of_h2 = case.invals['life_cycle_cost_of_h2'].val
    h2_production = case.invals['h2_production'].val
    equity = case.invals['equity'].val
    eq_rate_return = case.invals['eq_rate_return'].val
    debt = case.invals['debt'].val
    db_intrst_rate = case.invals['db_intrst_rate'].val
    inflation_rate = case.invals['inflation_rate'].val

    return (life_cycle_cost_of_h2, h2_production, equity, eq_rate_return, debt, db_intrst_rate, inflation_rate)

# The 'postprocess' function takes the output from the 'run' function and saves off the outputs for each case
def postprocess(case, lcoh_cost):
    case.addOutVal(name='LCOH_S1', val=lcoh_cost)
    case.addOutVal(name='lcoh1_case', val=case.ncase)
    return None

# Argument list of functions used in simulator function
fcns = {'run' : run,
        'preprocess' : preprocess,
        'postprocess': postprocess}

# Number of iterations in the monte carlo simulation
n_invests = 350
# Random seed for reproducibility
seed = 123456   # Recommanded for repeatability
# Usage of monaco simulator with the SOBOL Random function
sim = mc.Sim(name='LCOH', ndraws=n_invests, 
            fcns=fcns, seed=seed, 
            samplemethod='sobol',
            singlethreaded=True,
            savecasedata=False,
            verbose=True, debug=True)

# Variables used in the simulation
# By manually change loc and scale can manage the uncertainty of the cost
sim.addInVar(name='life_cycle_cost_of_h2', dist=uniform, distkwargs={'loc': 61000000, 'scale': 25000000})
sim.addInVar(name='inflation_rate', dist=uniform, distkwargs={'loc': 0.01, 'scale': 0.05})
sim.addInVar(name='h2_production', dist=uniform, distkwargs={'loc': 15000000, 'scale': 5000000}) 
sim.addInVar(name='equity', dist=uniform, distkwargs={'loc': 0.15, 'scale': 0.10})
sim.addInVar(name='eq_rate_return', dist=uniform, distkwargs={'loc': 0.05, 'scale': 0.01})
sim.addInVar(name='debt', dist=uniform, distkwargs={'loc': 0.75, 'scale': 0.01})
sim.addInVar(name='db_intrst_rate', dist=uniform, distkwargs={'loc': 0.045, 'scale': 0.005})

# Run the Simulation
sim.runSim()

# Calculate the mean and 5-95th percentile
# Statistics for the dice sum
sim.outvars['LCOH_S1'].addVarStat('mean')
sim.outvars['LCOH_S1'].addVarStat('percentile', {'p':[0.05, 0.95]})

# Plots a histogram of the dice sum
mc.plot(sim.outvars['LCOH_S1'])

# Creates a scatter plot of the sum vs the roll
# Number, showing randomness
mc.plot(sim.outvars['LCOH_S1'],
sim.outvars['lcoh1_case'])
plt.xlabel('€')

# Calculate the sensitivity of the dice sum to each of the input variables
sim.calcSensitivities('LCOH_S1')
sim.outvars['LCOH_S1'].plotSensitivities()

# %% [cell 49]
# Monaco LCOH S2 - Price range and sensitivity ratio

# Function for running the monte carlo simulation
def run(life_cycle_cost_of_h2, h2_production, equity, eq_rate_return, debt, db_intrst_rate, inflation_rate):
    
    # variables used within the LCOH function (utils.py)
    kwargs={
        'inflation_rate' : inflation_rate,
        'equity' : equity,
        'eq_rate_return' : eq_rate_return,
        'debt' : debt,
        'db_intrst_rate' : db_intrst_rate,
        }
    
    # run the lcoh simulation
    lcoh_cost = levelized_cost_of_h2(life_cycle_cost_of_h2=life_cycle_cost_of_h2, h2_production=h2_production, **kwargs)
    return (lcoh_cost, )

# The 'preprocess' function grabs the random input values for each case and structures it with any other data in the format the 'run' function expects
# Preprocess function for extracting usable variables from input dataframes
def preprocess(case):
    life_cycle_cost_of_h2 = case.invals['life_cycle_cost_of_h2'].val
    h2_production = case.invals['h2_production'].val
    equity = case.invals['equity'].val
    eq_rate_return = case.invals['eq_rate_return'].val
    debt = case.invals['debt'].val
    db_intrst_rate = case.invals['db_intrst_rate'].val
    inflation_rate = case.invals['inflation_rate'].val

    return (life_cycle_cost_of_h2, h2_production, equity, eq_rate_return, debt, db_intrst_rate, inflation_rate)

# The 'postprocess' function takes the output from the 'run' function and saves off the outputs for each case
def postprocess(case, lcoh_cost):
    case.addOutVal(name='LCOH_S2', val=lcoh_cost)
    case.addOutVal(name='lcoh2_case', val=case.ncase)
    return None

# Argument list of functions used in simulator function
fcns = {'run' : run,
        'preprocess' : preprocess,
        'postprocess': postprocess}

# Number of iterations in the monte carlo simulation
n_invests = 350 
# Random seed for reproducibility
seed = 123456 
# Usage of monaco simulator with the SOBOL Random function
sim = mc.Sim(name='LCOH', ndraws=n_invests, 
            fcns=fcns, seed=seed, 
            samplemethod='sobol',
            singlethreaded=True,
            savecasedata=False,
            verbose=True, debug=True)

# Variables used in the simulation
# By manually change loc and scale can manage the uncertainty of the cost
sim.addInVar(name='life_cycle_cost_of_h2', dist=uniform, distkwargs={'loc': 48000000, 'scale': 14000000})
sim.addInVar(name='inflation_rate', dist=uniform, distkwargs={'loc': 0.01, 'scale': 0.05})
sim.addInVar(name='h2_production', dist=uniform, distkwargs={'loc': 15000000, 'scale': 5000000}) 
sim.addInVar(name='equity', dist=uniform, distkwargs={'loc': 0.15, 'scale': 0.10})
sim.addInVar(name='eq_rate_return', dist=uniform, distkwargs={'loc': 0.05, 'scale': 0.01})
sim.addInVar(name='debt', dist=uniform, distkwargs={'loc': 0.75, 'scale': 0.01})
sim.addInVar(name='db_intrst_rate', dist=uniform, distkwargs={'loc': 0.045, 'scale': 0.005})

# Run the Simulation
sim.runSim()

# Calculate the mean and 5-95th percentile
# Statistics for the dice sum
sim.outvars['LCOH_S2'].addVarStat('mean')
sim.outvars['LCOH_S2'].addVarStat('percentile', {'p':[0.05, 0.95]})

# Plots a histogram of the dice sum
mc.plot(sim.outvars['LCOH_S2'])

# Creates a scatter plot of the sum vs the roll
# Number, showing randomness
mc.plot(sim.outvars['LCOH_S2'],
sim.outvars['lcoh2_case'])
plt.xlabel('€')

# Calculate the sensitivity of the dice sum to each of the input variables
sim.calcSensitivities('LCOH_S2')
sim.outvars['LCOH_S2'].plotSensitivities()

# %% [cell 50]
# Box plot - Results of LCOH in S1 and S2- for manuscript

# Calcualted data from previous process
# p5 and p95 refer to the 5th and 95th percentiles respectively
# we use the p5 and p95 to plot a boxplot
# 🔴 GEÄNDERT: hard-coded LCOH plot summary values updated
LCOH_s1 = {'min': 6.0, 'max': 13.2, 'median': 9.1, 'p5': 7.2, 'p95': 11.4}
LCOH_s2 = {'min': 4.5, 'max': 9.5, 'median': 6.8, 'p5': 5.5, 'p95': 8.5}

# Draw the boxplot
plt.figure(figsize=(8, 6))

# Plotting the boxplot using the data defined above
bp = plt.boxplot([[LCOH_s1['min'], LCOH_s1['p5'], LCOH_s1['median'], LCOH_s1['p95'], LCOH_s1['max']],
             [LCOH_s2['min'], LCOH_s2['p5'], LCOH_s2['median'], LCOH_s2['p95'], LCOH_s2['max']]],
            labels=['S1 LCOH', 'S2 LCOH'], vert=False, patch_artist=True)

# Set color
colors = [(195/255, 150/255, 190/255), (143/255, 215/255, 215/255)]
for box, color in zip(bp['boxes'], colors):
    box.set(facecolor=color)

plt.xlabel('€/kg H2')

# Manually define the x-axis
plt.xticks(range(0, 14))

# Print plot
plt.grid(True)
plt.show()


# %% [cell 51]
# Results validation - for manuscript

# Feed LCOH results from different literature
data = {
    "Year": [2012, 2016, 2017, 
             2018,
             2019,
             2020, 
             2021,
             2022, 2023],
    "LCOH Literature": [[4.49], [5.07], [3.64, 4.22, 4.31, 4.63, 5.38, 5.55], 
                        [3.64, 4.39, 4.22, 6.34, 4.31, 6.73],
                        [3, 20],
                        [2],
                        [9.2, 12.48, 6.90, 5.19, 1.16, 4.9, 1.47, 2.2, 1.75, 2.67],
                        [2.10, 5.24, 3.07, 6.63], [7.11, 7.88, 3.7, 9.9, 3.08, 13.12]],
}

# Extract data from above
years = data['Year']
lcoh_data = data['LCOH Literature']

# Plot price intervals using line plot
plt.figure(figsize=(10, 6))

for year, lcoh in zip(years, lcoh_data):
    if len(lcoh) == 1:
        plt.plot([year, year], [lcoh[0], lcoh[0]], color='black', linewidth=2, marker='o', markersize=8, label=f'{year} LCOH: {lcoh[0]}')
    else:
        ymin = min(lcoh)
        ymax = max(lcoh)
        plt.plot([year, year], [ymin, ymax], color='black', linewidth=2, label=f'{year} LCOH: {ymin} - {ymax}')
        plt.scatter([year] * len(lcoh), lcoh, color='black', zorder=5)

# Add our LCOH range
plt.fill_between(range(2012, 2025), 6, 13.2, color=(195/255, 125/255, 190/255), alpha=0.3)
plt.fill_between(range(2012, 2025), 4.5, 9.5, color=(143/255, 215/255, 215/255), alpha=0.3)

# Set the range of the x-axis
plt.xlim(2012, 2024)

# Set labels for x-axis and y-axis
plt.xlabel('Year')
plt.ylabel('LCOH  (€/kg H2)')

# Define legend elements
legend_elements = [Line2D([0], [0], marker='o', color='black', label='LCOH from other literature', markersize=6),
                   Patch(facecolor=(195/255, 125/255, 190/255), edgecolor='black', label='LCOH_S1 range from this paper'),
                   Patch(facecolor=(143/255, 215/255, 215/255), edgecolor='black', label='LCOH_S2 range from this paper')]

# Add legend
plt.legend(handles=legend_elements, loc='upper left')

# Show plot
plt.grid(True)

# Remove 2014 from x-axis ticks
plt.gca().set_xticks([2012, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])

plt.show()

