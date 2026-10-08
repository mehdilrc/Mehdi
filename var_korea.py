# %% [markdown]
# # South Korea: quarterly VAR (inflation, unemployment, policy rate, GDP growth)
# Run after fetch_fred_korea.py.   pip install pandas numpy matplotlib statsmodels

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")
Path("figures").mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Build the quarterly dataset (complete quarters only)

# %%
m = pd.read_csv("data/korea_macro_monthly.csv", parse_dates=["date"], index_col="date")
m = m[["inflation_yoy", "unemployment", "policy_rate", "real_gdp"]].dropna()

q = m.resample("QS").mean()
q["n_months"] = m["inflation_yoy"].resample("QS").count()
q = q[q["n_months"] == 3].drop(columns="n_months")  # drop the last partial quarter

# Stationary transformations (see ADF results). Order = Cholesky ordering for IRFs:
# real activity -> labour market -> prices -> policy rate  (an identifying ASSUMPTION)
data = pd.DataFrame({
    "gdp_qoq": q["real_gdp"].pct_change() * 100,
    "d_unemployment": q["unemployment"].diff(),
    "d_inflation": q["inflation_yoy"].diff(),
    "d_policy_rate": q["policy_rate"].diff(),
}).dropna()
level_infl = q["inflation_yoy"].loc[data.index]

print(f"Sample: {data.index.min().date()} -> {data.index.max().date()} ({len(data)} quarters)\n")

# %% [markdown]
# ## 2. Lag selection, estimation, stability

# %%
sel = VAR(data).select_order(maxlags=4)
print(sel.summary())
p = max(1, sel.selected_orders["bic"])
print(f"\nChosen lag order (BIC): {p}")

res = VAR(data).fit(p)
print(res.summary())
print("Stable (all roots outside unit circle):", res.is_stable())

# %% [markdown]
# ## 3. Residual diagnostics

# %%
print("Whiteness (Portmanteau) p-value:", round(res.test_whiteness(nlags=max(8, p + 2)).pvalue, 3))
print("Normality (Jarque-Bera) p-value:", round(res.test_normality().pvalue, 3))

# %% [markdown]
# ## 4. Granger causality (pairwise, F-test)

# %%
rows = []
for caused in data.columns:
    for causing in data.columns:
        if causing != caused:
            r = res.test_causality(caused, [causing], kind="f")
            rows.append({"cause": causing, "effect": caused, "p-value": round(r.pvalue, 3),
                         "significant (5%)": r.pvalue < 0.05})
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# ## 5. Impulse responses and variance decomposition

# %%
irf = res.irf(8)
for shock in ["d_policy_rate", "gdp_qoq"]:
    fig = irf.plot_cum_effects(orth=True, impulse=shock, figsize=(9, 8))
    fig.suptitle(f"Cumulative response to a {shock} shock (Cholesky)")
    fig.savefig(f"figures/03_irf_{shock}.png", dpi=150)
plt.show()

print("\nForecast error variance decomposition (horizon 8):")
res.fevd(8).summary()

# %% [markdown]
# ## 6. Out-of-sample inflation forecasts (one-step ahead, expanding window)
# Compare against a naive benchmark: a model is only useful if it beats "next = last".

# %%
n_test = 12
idx_infl = list(data.columns).index("d_inflation")
actual, naive, arima, var_fc, dates = [], [], [], [], []

for t in range(len(data) - n_test, len(data)):
    last = level_infl.iloc[t - 1]
    actual.append(level_infl.iloc[t])
    naive.append(last)
    arima.append(ARIMA(level_infl.iloc[:t].values, order=(1, 1, 0)).fit().forecast(1)[0])
    r = VAR(data.iloc[:t].values).fit(p)
    var_fc.append(last + r.forecast(data.iloc[t - p:t].values, 1)[0][idx_infl])
    dates.append(level_infl.index[t])

fc = pd.DataFrame({"actual": actual, "naive": naive, "arima(1,1,0)": arima, "var": var_fc}, index=dates)
errors = fc.drop(columns="actual").sub(fc["actual"], axis=0)
score = pd.DataFrame({"RMSE": np.sqrt((errors ** 2).mean()), "MAE": errors.abs().mean()}).round(3)
print(score)

ax = fc.plot(figsize=(10, 5), marker="o")
ax.set_title(f"One-step-ahead inflation forecasts, last {n_test} quarters")
ax.set_ylabel("Inflation (YoY, %)")
plt.tight_layout()
plt.savefig("figures/04_forecast_comparison.png", dpi=150)
plt.show()
