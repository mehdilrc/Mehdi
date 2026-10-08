# %% [markdown]
# # South Korea: Inflation, Unemployment & Growth - Exploratory Analysis
# Run after fetch_fred_korea.py. Works in VS Code / Jupyter (cells) or as a plain script.
# pip install pandas matplotlib statsmodels

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

Path("figures").mkdir(exist_ok=True)

df = pd.read_csv("data/korea_macro_monthly.csv", parse_dates=["date"], index_col="date")
cols = ["inflation_yoy", "unemployment", "policy_rate", "gdp_growth_yoy"]
d = df[cols].dropna()  # common window: where all four series exist

print(f"Common window: {d.index.min().date()} -> {d.index.max().date()} ({len(d)} months)\n")
print(d.describe().round(2))

# %% [markdown]
# ## 1. Time series with key episodes (approximate shading)

# %%
episodes = {
    "COVID-19": ("2020-02-01", "2020-12-31"),
    "Inflation surge": ("2021-06-01", "2023-06-30"),
}
labels = {
    "inflation_yoy": "Inflation (YoY, %)",
    "unemployment": "Unemployment rate (%)",
    "policy_rate": "Short-term interest rate (%)",
    "gdp_growth_yoy": "Real GDP growth (YoY, %)",
}

fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
for ax, c in zip(axes, cols):
    ax.plot(d.index, d[c], lw=1.6)
    ax.set_ylabel(labels[c])
    ax.axhline(0, color="grey", lw=0.5)
    for name, (a, b) in episodes.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), alpha=0.15, color="red" if name == "COVID-19" else "orange")
axes[0].set_title("South Korea: key macro indicators (shaded: COVID-19, inflation surge)")
plt.tight_layout()
plt.savefig("figures/01_timeseries.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. Correlations

# %%
corr = d.corr().round(2)
print(corr)

# %% [markdown]
# ## 3. Stationarity (ADF test): levels vs first differences

# %%
def adf(s, name):
    stat, p, *_ = adfuller(s.dropna(), autolag="AIC")
    return {"series": name, "ADF stat": round(stat, 2), "p-value": round(p, 3), "stationary (5%)": p < 0.05}

print("LEVELS")
print(pd.DataFrame([adf(d[c], c) for c in cols]).to_string(index=False))
print("\nFIRST DIFFERENCES")
print(pd.DataFrame([adf(d[c].diff(), f"d({c})") for c in cols]).to_string(index=False))

# %% [markdown]
# ## 4. Phillips curve: inflation vs unemployment, by sub-period

# %%
def phillips(sub, label):
    X = sm.add_constant(sub["unemployment"])
    m = sm.OLS(sub["inflation_yoy"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    print(f"{label:<12} slope={m.params['unemployment']:.3f}  "
          f"p={m.pvalues['unemployment']:.3f}  R2={m.rsquared:.3f}  n={int(m.nobs)}")

phillips(d, "Full sample")
phillips(d[:"2019-12"], "2010-2019")
phillips(d["2020-01":], "2020-2025")

fig, ax = plt.subplots(figsize=(7, 5))
pre, post = d[:"2019-12"], d["2020-01":]
ax.scatter(pre["unemployment"], pre["inflation_yoy"], alpha=0.6, label="2010-2019")
ax.scatter(post["unemployment"], post["inflation_yoy"], alpha=0.6, label="2020-2025")
ax.set_xlabel("Unemployment rate (%)")
ax.set_ylabel("Inflation (YoY, %)")
ax.set_title("Phillips curve: South Korea")
ax.legend()
plt.tight_layout()
plt.savefig("figures/02_phillips.png", dpi=150)
plt.show()
