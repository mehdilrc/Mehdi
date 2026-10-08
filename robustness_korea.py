# %% [markdown]
# # South Korea VAR - robustness checks
# 1) COVID dummies  2) VAR(2)  3) pre-2020 sample  4) alternative Cholesky orderings
# Run after var_korea.py.   pip install pandas numpy matplotlib statsmodels

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# %% [markdown]
# ## 0. Same quarterly dataset as var_korea.py

# %%
m = pd.read_csv("data/korea_macro_monthly.csv", parse_dates=["date"], index_col="date")
m = m[["inflation_yoy", "unemployment", "policy_rate", "real_gdp"]].dropna()

q = m.resample("QS").mean()
q["n_months"] = m["inflation_yoy"].resample("QS").count()
q = q[q["n_months"] == 3].drop(columns="n_months")

BASE_ORDER = ["gdp_qoq", "d_unemployment", "d_inflation", "d_policy_rate"]
data = pd.DataFrame({
    "gdp_qoq": q["real_gdp"].pct_change() * 100,
    "d_unemployment": q["unemployment"].diff(),
    "d_inflation": q["inflation_yoy"].diff(),
    "d_policy_rate": q["policy_rate"].diff(),
}).dropna()[BASE_ORDER]

# COVID dummies: a judgment call. Check the printed gdp_qoq and adjust the dates if needed.
covid = pd.DataFrame(0.0, index=data.index, columns=["covid_down", "covid_up"])
covid.loc["2020-01-01":"2020-06-30", "covid_down"] = 1.0   # contraction quarters
covid.loc["2020-07-01":"2020-12-31", "covid_up"] = 1.0     # rebound quarters
print("GDP growth around COVID (check the dummy quarters):")
print(data.loc["2019-10-01":"2021-04-01", "gdp_qoq"].round(2), "\n")

# %% [markdown]
# ## 1. Do the main findings survive different specifications?

# %%
KEY_PAIRS = [
    ("d_inflation", "gdp_qoq"),          # inflation -> GDP growth
    ("d_inflation", "d_policy_rate"),    # inflation -> policy rate
    ("d_policy_rate", "d_inflation"),    # policy rate -> inflation
]

def run_spec(name, p, use_exog=False, end=None):
    d = data if end is None else data.loc[:end]
    ex = covid.loc[d.index] if use_exog else None
    res = VAR(d, exog=ex).fit(p)
    row = {"specification": name, "lags": p, "n_obs": int(res.nobs), "stable": res.is_stable()}
    try:
        row["whiteness_p"] = round(res.test_whiteness(nlags=max(8, p + 2)).pvalue, 3)
    except Exception:
        row["whiteness_p"] = np.nan
    row["normality_p"] = round(res.test_normality().pvalue, 3)
    for cause, effect in KEY_PAIRS:
        row[f"{cause} -> {effect}"] = round(res.test_causality(effect, [cause], kind="f").pvalue, 3)
    return row

specs = [
    run_spec("Baseline VAR(1)", 1),
    run_spec("VAR(1) + COVID dummies", 1, use_exog=True),
    run_spec("VAR(2)", 2),
    run_spec("VAR(2) + COVID dummies", 2, use_exog=True),
    run_spec("Pre-2020 sample, VAR(1)", 1, end="2019-10-01"),
]
spec_table = pd.DataFrame(specs)
print("Granger-causality p-values across specifications (p < 0.05 = significant):")
print(spec_table.T.to_string(header=False))
spec_table.to_csv("results/robustness_specs.csv", index=False)

# %% [markdown]
# ## 2. Does the ordering of the Cholesky decomposition matter?

# %%
ORDERS = {
    "baseline (GDP first)": BASE_ORDER,
    "policy rate first": ["d_policy_rate", "gdp_qoq", "d_unemployment", "d_inflation"],
    "reversed": BASE_ORDER[::-1],
}

def ordering_check(order):
    res = VAR(data[order]).fit(1)
    i_inf, i_pol = order.index("d_inflation"), order.index("d_policy_rate")
    shares = res.fevd(8).decomp[i_inf, -1, :]  # variance shares of d_inflation at horizon 8
    out = {f"share from {c}": shares[order.index(c)] for c in BASE_ORDER}
    # cumulative response of d_inflation to a 1-s.d. policy-rate shock at horizon 8
    out["cum. inflation response to policy shock (h=8)"] = res.irf(8).orth_cum_effects[8, i_inf, i_pol]
    return out

order_table = pd.DataFrame({name: ordering_check(o) for name, o in ORDERS.items()}).T.round(3)
print("\nInflation variance decomposition and policy-shock response by Cholesky ordering:")
print(order_table.to_string())
order_table.to_csv("results/robustness_orderings.csv")

ax = order_table[[f"share from {c}" for c in BASE_ORDER]].plot(kind="bar", figsize=(9, 5), rot=0)
ax.set_title("Share of inflation forecast-error variance (h=8) by Cholesky ordering")
ax.set_ylabel("Share")
plt.tight_layout()
plt.savefig("figures/05_fevd_orderings.png", dpi=150)
plt.show()
