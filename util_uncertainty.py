import numpy as np
import pandas as pd

import monaco as mc

from scipy.stats import triang, uniform, lognorm

from util_model import (
    calculate_material_cost,
    investment_cost,
    end_of_life_cost,
    operation_cost,
    discounted_h2_production,
)

UNCERTAINTY_RANGE_BY_IMPORTANCE = {
    "high": {"min_share": 0.10, "relative_half_width": 0.25},
    "medium": {"min_share": 0.01, "relative_half_width": 0.20},
    "low": {"min_share": 0.00, "relative_half_width": 0.10},
}


# old classify_cost_share func
def relative_uncertainty_from_cost_share(cost_share):
    if cost_share >= UNCERTAINTY_RANGE_BY_IMPORTANCE["high"]["min_share"]:
        return UNCERTAINTY_RANGE_BY_IMPORTANCE["high"]["relative_half_width"]

    if cost_share >= UNCERTAINTY_RANGE_BY_IMPORTANCE["medium"]["min_share"]:
        return UNCERTAINTY_RANGE_BY_IMPORTANCE["medium"]["relative_half_width"]

    return UNCERTAINTY_RANGE_BY_IMPORTANCE["low"]["relative_half_width"]


def material_uncertainty_table(
    material_df,
    historical_stats,
):
    result = material_df.copy()

    result["Distribution"] = "uniform"

    result["Relative Uncertainty"] = result["Cost Share"].apply(
        relative_uncertainty_from_cost_share
    )

    historical_mask = result["Material"].isin(historical_stats.keys())

    # Historical metals use fitted historical distributions.
    result.loc[
        historical_mask,
        "Distribution",
    ] = "triangular"

    # Nickel uses the legacy lognormal assumption.
    result.loc[
        result["Material"] == "Nickel",
        "Distribution",
    ] = "lognormal"

    result.loc[
        historical_mask,
        "Relative Uncertainty",
    ] = np.nan

    return result


def triangular_distkwargs(stats):
    lower = float(stats["min"])
    mode = float(stats["mean"])
    upper = float(stats["max"])

    if upper <= lower:
        raise ValueError("Triangular distribution requires max > min")

    c = (mode - lower) / (upper - lower)

    return {
        "c": c,
        "loc": lower,
        "scale": upper - lower,
    }


def uniform_distkwargs(value, relative_uncertainty):
    lower = value * (1.0 - relative_uncertainty)
    upper = value * (1.0 + relative_uncertainty)

    return {
        "loc": lower,
        "scale": upper - lower,
    }


def lognormal_distkwargs(stats):
    """
    Convert arithmetic mean and standard deviation into
    scipy.stats.lognorm parameters.
    """

    mean = float(stats["mean"])
    std = float(stats["std"])

    if mean <= 0:
        raise ValueError("Lognormal distribution requires mean > 0")

    if std <= 0:
        raise ValueError("Lognormal distribution requires std > 0")

    sigma_squared = np.log(1.0 + (std**2 / mean**2))

    sigma = np.sqrt(sigma_squared)

    mu = np.log(mean) - sigma_squared / 2.0

    return {
        "s": sigma,
        "loc": 0.0,
        "scale": np.exp(mu),
    }


def run_material_monte_carlo(
    material_uncertainty,
    historical_stats,
    n_draws=2048,
    seed=123456,
):
    material_names = material_uncertainty["Material"].tolist()

    amounts = material_uncertainty["Amount"].to_numpy(dtype=float)

    # Monaco model functions
    def preprocess(case):
        return tuple(case.invals[name].val for name in material_names)

    def run(*unit_prices):
        cost = calculate_material_cost(
            amounts,
            unit_prices,
        )

        return (cost,)

    def postprocess(case, cost):
        case.addOutVal(
            name="Material Cost",
            val=cost,
        )

        return None

    fcns = {
        "preprocess": preprocess,
        "run": run,
        "postprocess": postprocess,
    }

    # Simulation
    sim = mc.Sim(
        name="Material Cost",
        ndraws=n_draws,
        fcns=fcns,
        seed=seed,
        samplemethod="sobol_random",
        singlethreaded=True,
        savecasedata=False,
        verbose=False,
        debug=False,
    )

    # Input distributions
    for _, row in material_uncertainty.iterrows():
        name = row["Material"]

        if row["Distribution"] == "triangular":
            sim.addInVar(
                name=name,
                dist=triang,
                distkwargs=triangular_distkwargs(historical_stats[name]),
            )

        elif row["Distribution"] == "lognormal":

            sim.addInVar(
                name=name,
                dist=lognorm,
                distkwargs=lognormal_distkwargs(historical_stats[name]),
            )

        elif row["Distribution"] == "uniform":
            sim.addInVar(
                name=name,
                dist=uniform,
                distkwargs=uniform_distkwargs(
                    value=float(row["Unit Price"]),
                    relative_uncertainty=float(row["Relative Uncertainty"]),
                ),
            )

        else:
            raise ValueError(
                f"Unsupported distribution " f"'{row['Distribution']}' for {name}"
            )

    # Run
    sim.runSim()

    output = sim.outvars["Material Cost"]

    output.addVarStat("mean")
    output.addVarStat(
        "percentile",
        {"p": [0.05, 0.50, 0.95]},
    )

    return sim


def build_material_sobol_problem(
    material_uncertainty,
    historical_stats,
):
    names = []
    bounds = []
    dists = []

    for _, row in material_uncertainty.iterrows():
        name = row["Material"]

        names.append(name)

        if row["Distribution"] == "triangular":
            stats = historical_stats[name]

            lower = float(stats["min"])
            mode = float(stats["mean"])
            upper = float(stats["max"])

            c = (mode - lower) / (upper - lower)

            bounds.append(
                [
                    lower,
                    upper,
                    c,
                ]
            )

            dists.append("triang")

        elif row["Distribution"] == "uniform":
            value = float(row["Unit Price"])
            uncertainty = float(row["Relative Uncertainty"])

            lower = value * (1.0 - uncertainty)
            upper = value * (1.0 + uncertainty)

            bounds.append(
                [
                    lower,
                    upper,
                ]
            )

            dists.append("unif")

        else:
            raise ValueError(
                f"Unsupported distribution " f"'{row['Distribution']}' for {name}"
            )

    return {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
        "dists": dists,
    }


def resource_uncertainty_stats(resource_df):
    """
    Build a triangular uncertainty definition from the
    deterministic lower, mean, and upper resource-cost paths.
    """

    lower = float(resource_df["Total Resource Cost Lower"].sum())

    mean = float(resource_df["Total Resource Cost Mean"].sum())

    upper = float(resource_df["Total Resource Cost Upper"].sum())

    if not lower < mean < upper:
        raise ValueError("Expected lower < mean < upper resource cost.")

    # For a triangular distribution:
    #
    # mean = (lower + mode + upper) / 3
    #
    # Select the mode so that the distribution mean equals
    # the deterministic mean resource-cost trajectory.
    mode = 3.0 * mean - lower - upper

    if not lower <= mode <= upper:
        raise ValueError(
            "The deterministic resource-cost bounds cannot "
            "define a triangular distribution with this mean."
        )

    c = (mode - lower) / (upper - lower)

    return {
        "lower": lower,
        "mean": mean,
        "mode": mode,
        "upper": upper,
        "c": c,
    }


def resource_path_from_total(
    resource_total,
    resource_df,
    stats,
):
    """
    Reconstruct a 40-period resource-cost path from a sampled
    total resource cost.

    Values below the deterministic mean are interpolated between
    the lower and mean paths. Values above the mean are
    interpolated between the mean and upper paths.
    """

    lower_path = resource_df["Total Resource Cost Lower"].to_numpy(dtype=float)

    mean_path = resource_df["Total Resource Cost Mean"].to_numpy(dtype=float)

    upper_path = resource_df["Total Resource Cost Upper"].to_numpy(dtype=float)

    if len(lower_path) != 40:
        raise ValueError(
            f"Expected 40 resource-cost periods, " f"got {len(lower_path)}."
        )

    if resource_total <= stats["mean"]:
        weight = (resource_total - stats["lower"]) / (stats["mean"] - stats["lower"])

        return lower_path + weight * (mean_path - lower_path)

    weight = (resource_total - stats["mean"]) / (stats["upper"] - stats["mean"])

    return mean_path + weight * (upper_path - mean_path)


def sample_material_unit_prices(
    material_uncertainty,
    historical_stats,
    unit_draws,
):
    """
    Convert uniform random draws into material-price samples.

    Historical material prices use triangular distributions.
    Point estimates use the uniform ranges defined in the
    material uncertainty table.
    """

    n_draws, n_materials = unit_draws.shape

    if n_materials != len(material_uncertainty):
        raise ValueError(
            "Number of random-input columns does not match " "the number of materials."
        )

    samples = np.empty(
        (n_draws, n_materials),
        dtype=float,
    )

    for j, (_, row) in enumerate(material_uncertainty.iterrows()):
        name = row["Material"]

        if row["Distribution"] == "triangular":
            kwargs = triangular_distkwargs(historical_stats[name])

            samples[:, j] = triang.ppf(
                unit_draws[:, j],
                **kwargs,
            )

        elif row["Distribution"] == "lognormal":

            kwargs = lognormal_distkwargs(historical_stats[name])

            samples[:, j] = lognorm.ppf(
                unit_draws[:, j],
                **kwargs,
            )

        elif row["Distribution"] == "uniform":
            kwargs = uniform_distkwargs(
                value=float(row["Unit Price"]),
                relative_uncertainty=float(row["Relative Uncertainty"]),
            )

            samples[:, j] = uniform.ppf(
                unit_draws[:, j],
                **kwargs,
            )

        else:
            raise ValueError(
                f"Unsupported distribution " f"'{row['Distribution']}' for {name}"
            )

    return samples


def run_lcc_monte_carlo(
    *,
    scenario_name,
    material_uncertainty,
    historical_stats,
    eol_assumptions,
    resource_df,
    resource_stats,
    construction_labour_fraction,
    operation_labour_fraction,
    deconstruction_fraction,
    inflation_rate,
    equity,
    equity_return,
    debt,
    debt_interest_rate,
    lifetime_years,
    annual_h2_production,
    eol_mode="AWE",
    tax_impact=0.0,
    n_draws=2048,
    seed=123456,
):
    """
    Run the complete AWE lifecycle-cost Monte Carlo model.

    Each realization propagates sampled material prices and
    scenario-specific resource cost through:
        material cost -> CAPEX -> EoL -> OPEX -> TCO -> LCOH.

    Financial parameters and hydrogen production remain fixed at
    the deterministic values in this Monte Carlo implementation.
    """

    material_names = material_uncertainty["Material"].tolist()

    amounts = material_uncertainty["Amount"].to_numpy(dtype=float)

    n_materials = len(material_names)

    # --------------------------------------------------------
    # Common random draws
    # --------------------------------------------------------
    #
    # The final column is reserved for the resource-cost sample.
    # Using the same seed for S1 and S2 gives the two scenarios
    # the same material-price draws and resource quantiles.
    rng = np.random.default_rng(seed)

    unit_draws = rng.random((n_draws, n_materials + 1))

    # --------------------------------------------------------
    # Material-price samples
    # --------------------------------------------------------

    unit_price_samples = sample_material_unit_prices(
        material_uncertainty=material_uncertainty,
        historical_stats=historical_stats,
        unit_draws=unit_draws[:, :n_materials],
    )

    material_cost_samples = calculate_material_cost(
        amounts,
        unit_price_samples,
    )

    # --------------------------------------------------------
    # CAPEX
    # --------------------------------------------------------

    capex_samples = investment_cost(
        material_cost=material_cost_samples,
        labour_fraction=construction_labour_fraction,
    )

    # --------------------------------------------------------
    # EoL inputs
    # --------------------------------------------------------

    if eol_mode not in {
        "AWE",
        "PEMWE",
    }:
        raise ValueError(f"Unsupported EoL mode: {eol_mode}")

    eol_inputs = material_uncertainty[
        [
            "Material",
            "Amount",
        ]
    ].merge(
        eol_assumptions,
        on="Material",
        how="left",
        validate="one_to_one",
    )

    required_columns = [
        "Recycle Ratio",
        "Transport Disposal Unit Cost",
    ]

    if eol_inputs[required_columns].isna().any().any():
        raise ValueError("Missing EoL assumptions " "for one or more materials.")

    recycle_ratios = eol_inputs["Recycle Ratio"].to_numpy(dtype=float)

    transport_unit_costs = eol_inputs["Transport Disposal Unit Cost"].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # AWE EoL
    # --------------------------------------------------------

    if eol_mode == "AWE":

        # - eol quantity = construction material quantity
        # - salvage price = sampled material unit price
        eol_amounts = amounts

        transport_disposal_cost = np.sum(eol_amounts * transport_unit_costs)

        salvage_samples = np.sum(
            unit_price_samples
            * eol_amounts[np.newaxis, :]
            * recycle_ratios[np.newaxis, :],
            axis=1,
        )

    # --------------------------------------------------------
    # PEMWE EoL
    # --------------------------------------------------------

    else:

        required_pemwe_columns = [
            "EoL Amount",
            "Salvage Unit Price",
        ]

        if eol_inputs[required_pemwe_columns].isna().any().any():
            raise ValueError(
                "PEMWE EoL requires "
                "'EoL Amount' and "
                "'Salvage Unit Price' "
                "for every material."
            )

        eol_amounts = eol_inputs["EoL Amount"].to_numpy(dtype=float)

        salvage_unit_prices = eol_inputs["Salvage Unit Price"].to_numpy(dtype=float)

        transport_disposal_cost = np.sum(eol_amounts * transport_unit_costs)

        # pemwe uses explicit end-of-life selling prices.
        # therefore salvage does not depend on the sampled
        # construction material price
        salvage_value = np.sum(eol_amounts * recycle_ratios * salvage_unit_prices)

        salvage_samples = np.full(
            n_draws,
            salvage_value,
            dtype=float,
        )


    # --------------------------------------------------------
    # EoL cost
    # --------------------------------------------------------

    eol_samples = end_of_life_cost(
        construction_cost=material_cost_samples,
        deconstruction_fraction=deconstruction_fraction,
        transport_disposal_cost=transport_disposal_cost,
        salvage_value=salvage_samples,
        inflation_rate=inflation_rate,
        equity=equity,
        equity_return=equity_return,
        debt=debt,
        debt_interest_rate=debt_interest_rate,
        lifetime_years=lifetime_years,
    )

    real_wacc = float(np.asarray(eol_samples["real_wacc"]))

    eol_present_samples = np.asarray(
        eol_samples["present_cost"],
        dtype=float,
    )

    # --------------------------------------------------------
    # Resource-cost samples
    # --------------------------------------------------------

    resource_total_samples = triang.ppf(
        unit_draws[:, -1],
        c=resource_stats["c"],
        loc=resource_stats["lower"],
        scale=(resource_stats["upper"] - resource_stats["lower"]),
    )

    # --------------------------------------------------------
    # Discounted hydrogen production
    # --------------------------------------------------------

    discounted_h2 = discounted_h2_production(
        annual_h2_production=annual_h2_production,
        discount_rate=real_wacc,
        lifetime_years=lifetime_years,
    )

    # --------------------------------------------------------
    # Complete lifecycle calculation
    # --------------------------------------------------------

    rows = []

    for i in range(n_draws):

        resource_path = resource_path_from_total(
            resource_total=resource_total_samples[i],
            resource_df=resource_df,
            stats=resource_stats,
        )

        opex = operation_cost(
            resource_cost=resource_path,
            material_cost=material_cost_samples[i],
            labour_fraction=operation_labour_fraction,
            eol_present_cost=eol_present_samples[i],
            discount_rate=real_wacc,
        )

        tco = capex_samples["capex"][i] + opex["opex"] - tax_impact

        lcoh = tco / discounted_h2

        rows.append(
            {
                "case": i,
                "scenario": scenario_name,
                "material_cost": material_cost_samples[i],
                "construction_labour_cost": capex_samples["labour_cost"][i],
                "capex": capex_samples["capex"][i],
                "salvage_value": salvage_samples[i],
                "eol_present_cost": eol_present_samples[i],
                "resource_cost_total": resource_total_samples[i],
                "resource_cost_pv": opex["resource_cost"],
                "operating_labour_cost": opex["labour_cost"],
                "opex": opex["opex"],
                "tco": tco,
                "lcoh": lcoh,
            }
        )

    return pd.DataFrame(rows)
