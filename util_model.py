import pandas as pd
import numpy as np


def material_cost_table(materials):
    df = materials.copy()

    df["Cost"] = df["Amount"] * df["Unit Price"]

    df["Cost Share"] = df["Cost"] / df["Cost"].sum()

    return df


def calculate_material_cost(amounts, unit_prices):
    amounts = np.asarray(amounts, dtype=float)
    unit_prices = np.asarray(unit_prices, dtype=float)

    return unit_prices @ amounts


def investment_cost(
    material_cost,
    labour_fraction,
    investment_grant=0.0,
):
    labour_cost = material_cost * labour_fraction

    capex = material_cost + labour_cost - investment_grant

    return {
        "material_cost": material_cost,
        "labour_cost": labour_cost,
        "investment_grant": investment_grant,
        "capex": capex,
    }


def resource_cost_table(
    electricity_forecast,
    water_forecast,
    *,
    electricity_cols,
    h2_per_period,
    electricity_kwh_per_kg,
    water_kg_per_kg,
    koh_kg_per_kg=0.0,
    koh_price_eur_per_kg=0.0,
):
    electricity = (
        electricity_forecast[["DATE", *electricity_cols]]
        .copy()
        .rename(
            columns={
                electricity_cols[0]: "Electricity Price Lower",
                electricity_cols[1]: "Electricity Price Mean",
                electricity_cols[2]: "Electricity Price Upper",
            }
        )
    )

    water = (
        water_forecast[["DATE", "Water Lower", "Water Mean", "Water Upper"]]
        .copy()
        .rename(
            columns={
                "Water Lower": "Water Price Lower",
                "Water Mean": "Water Price Mean",
                "Water Upper": "Water Price Upper",
            }
        )
    )

    df = electricity.merge(
        water,
        on="DATE",
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(electricity) or len(df) != len(water):
        raise ValueError("Electricity and water forecast dates do not align")

    # Physical consumption per half-year operating period
    df["H2 Production"] = h2_per_period

    df["Electricity Consumption"] = h2_per_period * electricity_kwh_per_kg

    df["Water Consumption"] = h2_per_period * water_kg_per_kg

    df["KOH Consumption"] = h2_per_period * koh_kg_per_kg

    # Resource costs
    for bound in ["Lower", "Mean", "Upper"]:
        df[f"Electricity Cost {bound}"] = (
            df["Electricity Consumption"] * df[f"Electricity Price {bound}"]
        )

        df[f"Water Cost {bound}"] = df["Water Consumption"] * df[f"Water Price {bound}"]

    df["KOH Cost"] = df["KOH Consumption"] * koh_price_eur_per_kg

    for bound in ["Lower", "Mean", "Upper"]:
        df[f"Total Resource Cost {bound}"] = (
            df[f"Electricity Cost {bound}"] + df[f"Water Cost {bound}"] + df["KOH Cost"]
        )

    return df


def wacc_nominal(
    equity,
    equity_return,
    debt,
    debt_interest_rate,
):
    total_capital = equity + debt

    return (
        equity * equity_return
        + debt * debt_interest_rate
    ) / total_capital


def wacc_real(
    nominal_wacc,
    inflation_rate,
):
    return (
        (1.0 + nominal_wacc)
        / (1.0 + inflation_rate)
        - 1.0
    )

def end_of_life_cost(
    construction_cost,
    deconstruction_fraction,
    transport_disposal_cost,
    salvage_value,
    *,
    inflation_rate,
    equity,
    equity_return,
    debt,
    debt_interest_rate,
    lifetime_years,
):
    """
    Calculate nominal and present-value end-of-life cost.
    """

    # Financing cost before inflation adjustment
    nominal_wacc = wacc_nominal(
        equity=equity,
        equity_return=equity_return,
        debt=debt,
        debt_interest_rate=debt_interest_rate,
    )

    # Convert nominal WACC to a real discount rate
    real_wacc = wacc_real(
        nominal_wacc=nominal_wacc,
        inflation_rate=inflation_rate,
    )

    # Deconstruction is modeled as a fixed share of construction cost
    deconstruction_cost = (
        construction_cost
        * deconstruction_fraction
    )

    # Terminal EoL cost before discounting
    nominal_cost = (
        deconstruction_cost
        + transport_disposal_cost
        - salvage_value
    )

    # Present value of the terminal EoL cost
    present_cost = (
        nominal_cost
        / (1.0 + real_wacc) ** lifetime_years
    )

    return {
        "deconstruction": deconstruction_cost,
        "transport_disposal": transport_disposal_cost,
        "salvage_value": salvage_value,
        "nominal_wacc": nominal_wacc,
        "real_wacc": real_wacc,
        "nominal_cost": nominal_cost,
        "present_cost": present_cost,
    }


def eol_table(
    material_df,
    eol_assumptions,
):
    df = material_df.merge(
        eol_assumptions,
        on="Material",
        how="left",
        validate="one_to_one",
    )

    # Common EoL inputs required by both technologies.
    required_columns = [
        "Recycle Ratio",
        "Transport Disposal Unit Cost",
    ]

    if df[required_columns].isna().any().any():
        raise ValueError(
            "Missing EoL assumptions for one or more materials"
        )

    # --------------------------------------------------------
    # EoL amount
    # --------------------------------------------------------
    #
    # AWE:
    #     uses the construction material amount.
    #
    # PEMWE:
    #     provides an explicit EoL amount.
    #
    if "EoL Amount" in df.columns:

        if df["EoL Amount"].isna().any():
            raise ValueError(
                "Missing EoL Amount for one or more materials"
            )

        eol_amount = df["EoL Amount"].to_numpy(
            dtype=float,
        )

    else:
        eol_amount = df["Amount"].to_numpy(
            dtype=float,
        )

    # --------------------------------------------------------
    # Transport / disposal
    # --------------------------------------------------------

    df["Transport Disposal Cost"] = (
        eol_amount
        * df[
            "Transport Disposal Unit Cost"
        ].to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # Salvage value
    # --------------------------------------------------------
    #
    # AWE:
    #     cost * recycle ratio
    #
    #     which is equivalent to
    #
    #     amount * unit price * recycle ratio
    #
    # PEMWE:
    #     EoL Amount
    #     * recycle ratio
    #     * explicit salvage unit price
    #
    if "Salvage Unit Price" in df.columns:

        if df["Salvage Unit Price"].isna().any():
            raise ValueError(
                "Missing Salvage Unit Price "
                "for one or more materials"
            )

        df["Salvage Value"] = (
            eol_amount
            * df["Recycle Ratio"].to_numpy(
                dtype=float,
            )
            * df["Salvage Unit Price"].to_numpy(
                dtype=float,
            )
        )

    else:
        df["Salvage Value"] = (
            df["Cost"]
            * df["Recycle Ratio"]
        )

    return df

    
def operation_cost(
    resource_cost,
    *,
    material_cost,
    labour_fraction,
    eol_present_cost,
    discount_rate,
):
    resource_cost = np.asarray(
        resource_cost,
        dtype=float,
    )

    if len(resource_cost) != 40:
        raise ValueError(
            f"Expected 40 half-year resource costs, "
            f"got {len(resource_cost)}"
        )

    # Preserve the original model timing:
    # two half-year periods share one annual discount exponent.
    discount_exponents = np.repeat(
        np.arange(20),
        2,
    )

    discounted_resource_cost = np.sum(
        resource_cost
        / (1.0 + discount_rate) ** discount_exponents
    )

    labour_cost = (
        material_cost
        * labour_fraction
    )

    total_opex = (
        discounted_resource_cost
        + labour_cost
        + eol_present_cost
    )

    return {
        "resource_cost": discounted_resource_cost,
        "labour_cost": labour_cost,
        "eol_cost": eol_present_cost,
        "opex": total_opex,
    }

def discounted_h2_production(
    annual_h2_production,
    discount_rate,
    lifetime_years,
):
    """
    Calculate discounted lifetime hydrogen production.

    Annual hydrogen production is discounted using the same
    real discount rate that is used in the lifecycle-cost model.
    """

    years = np.arange(
        lifetime_years,
        dtype=float,
    )

    discounted_production = (
        annual_h2_production
        / (1.0 + discount_rate) ** years
    )

    return discounted_production.sum()

