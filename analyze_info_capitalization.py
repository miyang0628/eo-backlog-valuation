"""Information capitalization of contracted data revenue in listed EO firms.

This is the main analysis for the study
    "Does the Market Price Contracted Data Revenue? Backlog Depth as an
     Information Signal in the Valuation of Listed Earth-Observation Firms."

It reads the constructed four-firm panel (data/eo_panel_refined.csv) and
produces every table and figure in results/. The panel itself is built from
SEC XBRL company-concept facts joined to public quarter-end prices by
build_eo_panel.py (unchanged from the public replication package); this
script performs the valuation analysis on top of it.

Design choices, stated plainly:
  - Dependent variable is log(EV/Sales), a scale-free forward valuation
    multiple. Logs because multiples are right-skewed and the elasticity
    interpretation is the object of interest.
  - Backlog depth = remaining performance obligations / TTM revenue (RPO
    scaled by revenue). It proxies the DEPTH of contracted, recurring data
    revenue an EO firm has already booked but not yet recognized.
  - We estimate four nested specifications (pooled, firm FE, year FE,
    firm+year FE) with firm-clustered standard errors, and we report the
    fragility of four-cluster inference openly rather than hide it.
  - No imputation. Firm-quarters lacking any component are dropped, never
    filled. SATL drops out of the regression entirely (it never discloses
    RPO); we say so.

All inputs are public. Nothing here claims a causal effect.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ----------------------------------------------------------------------------
# Presentation constants (grayscale, print-safe, 600 dpi)
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})
GRAYS = ["#1a1a1a", "#4d4d4d", "#808080", "#b3b3b3"]
FIRM_ORDER = ["PL", "BKSY", "SPIR", "SATL"]
FIRM_LABEL = {
    "PL": "Planet Labs",
    "BKSY": "BlackSky",
    "SPIR": "Spire Global",
    "SATL": "Satellogic",
}

TAB = "results/tables"
FIG = "results/figures"


# ----------------------------------------------------------------------------
# Load & construct analysis variables
# ----------------------------------------------------------------------------
def load_panel(path="data/eo_panel_refined.csv"):
    df = pd.read_csv(path, parse_dates=["end"])
    df = df.sort_values(["ticker", "end"]).reset_index(drop=True)
    # Analysis variables. Backlog depth is the paper's information signal.
    df["backlog_depth"] = df["rpo_to_rev"]
    df["growth"] = df["rev_yoy"]
    df["ev_sales"] = df["ev_sales"]
    with np.errstate(invalid="ignore", divide="ignore"):
        df["log_ev_sales"] = np.log(df["ev_sales"])
    return df


def regression_sample(df):
    """Firm-quarters with all three regression variables present. No fills."""
    cols = ["log_ev_sales", "backlog_depth", "growth"]
    d = df.dropna(subset=cols).copy()
    d = d[np.isfinite(d["log_ev_sales"])]
    return d.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Table 1  — panel descriptives
# ----------------------------------------------------------------------------
def table_descriptives(d):
    rows = []
    for tk in [t for t in FIRM_ORDER if t in d.ticker.unique()]:
        s = d[d.ticker == tk]
        rows.append({
            "ticker": tk,
            "firm": FIRM_LABEL[tk],
            "n": len(s),
            "ev_sales_mean": round(s.ev_sales.mean(), 2),
            "ev_sales_sd": round(s.ev_sales.std(), 2),
            "backlog_depth_mean": round(s.backlog_depth.mean(), 2),
            "growth_mean": round(s.growth.mean(), 2),
        })
    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/t1_panel_descriptives.csv", index=False)
    return out


# ----------------------------------------------------------------------------
# Table 2  — nested panel regressions, firm-clustered SE
# ----------------------------------------------------------------------------
def _fit(formula, d):
    return smf.ols(formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["ticker"]}
    )


def table_regressions(d):
    specs = {
        "Pooled":        "log_ev_sales ~ backlog_depth + growth",
        "Firm FE":       "log_ev_sales ~ backlog_depth + growth + C(ticker)",
        "Year FE":       "log_ev_sales ~ backlog_depth + growth + C(year)",
        "Firm+Year FE":  "log_ev_sales ~ backlog_depth + growth + C(ticker) + C(year)",
    }
    rows = []
    fits = {}
    for name, f in specs.items():
        m = _fit(f, d)
        fits[name] = m
        rows.append({
            "Model": name,
            "n": int(m.nobs),
            "R2": round(m.rsquared, 3),
            "beta_backlog": round(m.params["backlog_depth"], 3),
            "se_backlog": round(m.bse["backlog_depth"], 3),
            "p_backlog": round(m.pvalues["backlog_depth"], 4),
            "beta_growth": round(m.params["growth"], 3),
            "se_growth": round(m.bse["growth"], 3),
            "p_growth": round(m.pvalues["growth"], 4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/t2_panel_regressions.csv", index=False)
    return out, fits


# ----------------------------------------------------------------------------
# Table 3  — robustness / fragility of small-cluster inference
# ----------------------------------------------------------------------------
def table_robustness(d):
    """Same headline spec under alternative SE and a leave-one-firm-out check."""
    base = "log_ev_sales ~ backlog_depth + growth + C(ticker) + C(year)"
    rows = []

    # (a) alternative standard errors on the full sample
    m_hc = smf.ols(base, data=d).fit(cov_type="HC1")
    m_cl = _fit(base, d)
    for label, m in [("HC1 (heterosk.)", m_hc), ("Cluster by firm", m_cl)]:
        rows.append({
            "variant": label,
            "n": int(m.nobs),
            "beta_backlog": round(m.params["backlog_depth"], 3),
            "se_backlog": round(m.bse["backlog_depth"], 3),
            "p_backlog": round(m.pvalues["backlog_depth"], 4),
        })

    # (b) leave-one-firm-out: does any single firm drive the result?
    for tk in sorted(d.ticker.unique()):
        sub = d[d.ticker != tk]
        m = smf.ols(base, data=sub).fit(cov_type="HC1")
        rows.append({
            "variant": f"drop {tk}",
            "n": int(m.nobs),
            "beta_backlog": round(m.params["backlog_depth"], 3),
            "se_backlog": round(m.bse["backlog_depth"], 3),
            "p_backlog": round(m.pvalues["backlog_depth"], 4),
        })

    # (c) wild-cluster-style sanity: within-transform then plain OLS
    dd = d.copy()
    for v in ["log_ev_sales", "backlog_depth", "growth"]:
        dd[v + "_w"] = dd.groupby("ticker")[v].transform(lambda x: x - x.mean())
    m_w = smf.ols("log_ev_sales_w ~ backlog_depth_w + growth_w - 1", data=dd).fit(cov_type="HC1")
    rows.append({
        "variant": "within-firm demeaned",
        "n": int(m_w.nobs),
        "beta_backlog": round(m_w.params["backlog_depth_w"], 3),
        "se_backlog": round(m_w.bse["backlog_depth_w"], 3),
        "p_backlog": round(m_w.pvalues["backlog_depth_w"], 4),
    })

    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/t3_robustness.csv", index=False)
    return out


# ----------------------------------------------------------------------------
# Table 4  — univariate associations with the raw multiple (transparency)
# ----------------------------------------------------------------------------
def table_associations(d):
    rows = []
    for v, lab in [("backlog_depth", "Backlog depth (RPO/rev)"),
                   ("growth", "Revenue growth (YoY)")]:
        r, p = stats.pearsonr(d[v], d["log_ev_sales"])
        rs, ps = stats.spearmanr(d[v], d["log_ev_sales"])
        rows.append({
            "signal": lab,
            "n": len(d),
            "pearson_r": round(r, 3), "pearson_p": round(p, 4),
            "spearman_rho": round(rs, 3), "spearman_p": round(ps, 4),
        })
    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/t4_associations.csv", index=False)
    return out


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_backlog_vs_multiple(d):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for i, tk in enumerate([t for t in FIRM_ORDER if t in d.ticker.unique()]):
        s = d[d.ticker == tk]
        ax.scatter(s.backlog_depth, s.log_ev_sales, s=42,
                   facecolor=GRAYS[i % 4], edgecolor="black", linewidth=0.5,
                   label=FIRM_LABEL[tk], zorder=3)
    # pooled fit line
    b = np.polyfit(d.backlog_depth, d.log_ev_sales, 1)
    xs = np.linspace(d.backlog_depth.min(), d.backlog_depth.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color="black", lw=1.2, ls="--",
            zorder=2, label="Pooled fit")
    ax.set_xlabel("Backlog depth  (RPO / TTM revenue)")
    ax.set_ylabel("log(EV / Sales)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(True, lw=0.3, alpha=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/fig1_backlog_vs_multiple.{ext}")
    plt.close(fig)


def fig_coef_across_specs(reg_tbl):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    names = reg_tbl.Model.tolist()
    beta = reg_tbl.beta_backlog.values
    se = reg_tbl.se_backlog.values
    y = np.arange(len(names))[::-1]
    ax.errorbar(beta, y, xerr=1.96 * se, fmt="o", color="black",
                capsize=3, lw=1, markersize=5)
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(r"$\beta$ on backlog depth  (95% CI, firm-clustered)")
    ax.grid(True, axis="x", lw=0.3, alpha=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/fig2_coef_across_specs.{ext}")
    plt.close(fig)


def fig_backlog_timeline(df):
    """Backlog depth over time by firm — the 'deepening recurring revenue' story."""
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, tk in enumerate([t for t in FIRM_ORDER if t in df.ticker.unique()]):
        s = df[df.ticker == tk].dropna(subset=["backlog_depth"])
        if len(s) < 2:
            continue
        ax.plot(s.end, s.backlog_depth, marker="o", ms=3.5, lw=1.1,
                color=GRAYS[i % 4], label=FIRM_LABEL[tk])
    ax.set_ylabel("Backlog depth  (RPO / TTM revenue)")
    ax.set_xlabel("Fiscal quarter-end")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(True, lw=0.3, alpha=0.5)
    fig.autofmt_xdate()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/fig3_backlog_timeline.{ext}")
    plt.close(fig)


def fig_per_firm_slopes(pfs):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    y = np.arange(len(pfs))[::-1]
    labels = [FIRM_LABEL.get(t, t) for t in pfs.ticker]
    ax.scatter(pfs.slope_backlog, y, s=60, color="black", zorder=3)
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    for yi, v in zip(y, pfs.slope_backlog):
        ax.annotate(f"{v:.2f}", (v, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Within-firm slope on backlog depth")
    ax.set_xlim(-0.3, 1.7)
    ax.grid(True, axis="x", lw=0.3, alpha=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/fig5_per_firm_slopes.{ext}")
    plt.close(fig)


def fig_leave_one_out(rob_tbl):
    sub = rob_tbl[rob_tbl.variant.str.startswith("drop")].copy()
    full = rob_tbl[rob_tbl.variant == "HC1 (heterosk.)"].iloc[0]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    y = np.arange(len(sub))[::-1]
    ax.errorbar(sub.beta_backlog, y, xerr=1.96 * sub.se_backlog, fmt="s",
                color="black", capsize=3, lw=1, markersize=5)
    ax.axvline(full.beta_backlog, color="gray", lw=1.0, ls="--",
               label="Full sample")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(sub.variant)
    ax.set_xlabel(r"$\beta$ on backlog depth (95% CI)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(True, axis="x", lw=0.3, alpha=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/fig4_leave_one_out.{ext}")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Table 5 — shared-denominator defense (added in revision)
# ----------------------------------------------------------------------------
def table_denominator_defense(d):
    """Address the mechanical concern that EV/Sales and RPO/Sales share the
    revenue denominator. If backlog depth survives (indeed strengthens) once
    log revenue is controlled, the association is not a pure deflator artifact.
    The per-share formulation is reported too, including its null, for honesty.
    """
    dd = d.copy()
    dd["log_ttm_rev"] = np.log(dd["ttm_rev"])
    dd["log_ev"] = np.log(dd["ev"])
    dd["rpo_ps"] = dd["rpo"] / dd["shares"]
    dd["log_price"] = np.log(dd["price"])

    rows = []

    def hc1(formula, data):
        return smf.ols(formula, data=data).fit(cov_type="HC1")

    m1 = hc1("log_ev_sales ~ backlog_depth + growth + C(ticker) + C(year)", dd)
    rows.append(("(1) Baseline (firm+year FE)", "backlog_depth",
                 m1.params["backlog_depth"], m1.bse["backlog_depth"],
                 m1.pvalues["backlog_depth"], int(m1.nobs)))

    m2 = hc1("log_ev_sales ~ backlog_depth + growth + log_ttm_rev + C(ticker) + C(year)", dd)
    rows.append(("(2) + control log(TTM revenue)", "backlog_depth",
                 m2.params["backlog_depth"], m2.bse["backlog_depth"],
                 m2.pvalues["backlog_depth"], int(m2.nobs)))

    m3 = hc1("log_ev ~ backlog_depth + growth + log_ttm_rev + C(ticker) + C(year)", dd)
    rows.append(("(3) LHS=log(EV), revenue on RHS", "backlog_depth",
                 m3.params["backlog_depth"], m3.bse["backlog_depth"],
                 m3.pvalues["backlog_depth"], int(m3.nobs)))

    m4 = hc1("log_price ~ rpo_ps + growth + C(ticker) + C(year)", dd)
    rows.append(("(4) Per-share: log(price)~RPO/share", "rpo_ps",
                 m4.params["rpo_ps"], m4.bse["rpo_ps"],
                 m4.pvalues["rpo_ps"], int(m4.nobs)))

    out = pd.DataFrame(rows, columns=["spec", "regressor", "beta", "se", "p_HC1", "n"])
    out[["beta", "se", "p_HC1"]] = out[["beta", "se", "p_HC1"]].round(4)
    out.to_csv(f"{TAB}/t5_denominator_defense.csv", index=False)
    return out


# ----------------------------------------------------------------------------
# Table 6 — liquidity control + changes/event-time test (added in revision)
# ----------------------------------------------------------------------------
def table_controls_and_changes(d):
    dd = d.sort_values(["ticker", "end"]).copy()
    dd["cash_to_rev"] = dd["cash"] / dd["ttm_rev"]
    rows = []

    def hc1(formula, data):
        return smf.ols(formula, data=data).fit(cov_type="HC1")

    # liquidity/balance-sheet control (proxy for quality/runway)
    m = hc1("log_ev_sales ~ backlog_depth + growth + cash_to_rev + C(ticker) + C(year)", dd)
    rows.append({"test": "Levels + cash/revenue control", "n": int(m.nobs),
                 "beta_backlog": round(m.params["backlog_depth"], 3),
                 "se": round(m.bse["backlog_depth"], 3),
                 "p_HC1": round(m.pvalues["backlog_depth"], 4)})

    # changes-on-changes (contemporaneous)
    dd["d_backlog"] = dd.groupby("ticker")["backlog_depth"].diff()
    dd["d_logmult"] = dd.groupby("ticker")["log_ev_sales"].diff()
    dd["d_growth"] = dd.groupby("ticker")["growth"].diff()
    cc = dd.dropna(subset=["d_backlog", "d_logmult", "d_growth"])
    mc = hc1("d_logmult ~ d_backlog + d_growth", cc)
    rows.append({"test": "Changes: d(mult) ~ d(backlog), contemp.", "n": int(mc.nobs),
                 "beta_backlog": round(mc.params["d_backlog"], 3),
                 "se": round(mc.bse["d_backlog"], 3),
                 "p_HC1": round(mc.pvalues["d_backlog"], 4)})

    # lead: d(backlog)_t -> d(mult)_{t+1}
    dd["d_logmult_next"] = dd.groupby("ticker")["d_logmult"].shift(-1)
    cl = dd.dropna(subset=["d_backlog", "d_logmult_next", "d_growth"])
    ml = hc1("d_logmult_next ~ d_backlog + d_growth", cl)
    rows.append({"test": "Lead: d(backlog)_t -> d(mult)_{t+1}", "n": int(ml.nobs),
                 "beta_backlog": round(ml.params["d_backlog"], 3),
                 "se": round(ml.bse["d_backlog"], 3),
                 "p_HC1": round(ml.pvalues["d_backlog"], 4)})

    out = pd.DataFrame(rows)
    out.to_csv(f"{TAB}/t6_controls_and_changes.csv", index=False)
    return out


def per_firm_slopes(d):
    """Ibragimov-Muller-style: estimate the backlog slope firm by firm."""
    rows = []
    for tk in sorted(d.ticker.unique()):
        s = d[d.ticker == tk]
        m = smf.ols("log_ev_sales ~ backlog_depth + growth", data=s).fit()
        rows.append({"ticker": tk, "n": len(s),
                     "slope_backlog": round(m.params["backlog_depth"], 3)})
    out = pd.DataFrame(rows)
    slopes = out["slope_backlog"].values
    t, p = stats.ttest_1samp(slopes, 0)
    out.to_csv(f"{TAB}/t7_per_firm_slopes.csv", index=False)
    return out, (round(float(np.mean(slopes)), 3), round(float(t), 2), round(float(p), 3),
                 bool(np.all(slopes > 0)))


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    df = load_panel()
    d = regression_sample(df)

    print(f"Full panel: {len(df)} firm-quarters, {df.ticker.nunique()} firms")
    print(f"Regression sample: {len(d)} firm-quarters, "
          f"{d.ticker.nunique()} firms "
          f"({', '.join(sorted(d.ticker.unique()))}); "
          f"years {int(d.year.min())}-{int(d.year.max())}")
    print("  (SATL drops out: never discloses RPO -> no backlog depth)\n")

    desc = table_descriptives(d)
    reg, fits = table_regressions(d)
    rob = table_robustness(d)
    assoc = table_associations(d)
    denom = table_denominator_defense(d)
    ctrl = table_controls_and_changes(d)
    pfs, im = per_firm_slopes(d)

    # one-SD economic magnitude of the preferred estimate
    beta_pref = reg.loc[reg.Model == "Firm+Year FE", "beta_backlog"].iloc[0]
    sd_bd = d["backlog_depth"].std()
    one_sd_effect = np.exp(beta_pref * sd_bd) - 1

    print("== T1 descriptives =="); print(desc.to_string(index=False)); print()
    print("== T2 regressions =="); print(reg.to_string(index=False)); print()
    print("== T3 robustness =="); print(rob.to_string(index=False)); print()
    print("== T4 associations =="); print(assoc.to_string(index=False)); print()
    print("== T5 denominator defense =="); print(denom.to_string(index=False)); print()
    print("== T6 controls + changes =="); print(ctrl.to_string(index=False)); print()
    print("== T7 per-firm slopes =="); print(pfs.to_string(index=False))
    print(f"   Ibragimov-Muller: mean={im[0]}, t={im[1]}, p={im[2]}, all_positive={im[3]}")
    print(f"   One-SD effect (SD={sd_bd:.3f}): {one_sd_effect:.1%} higher multiple\n")

    fig_backlog_vs_multiple(d)
    fig_coef_across_specs(reg)
    fig_backlog_timeline(df)
    fig_leave_one_out(rob)
    fig_per_firm_slopes(pfs)
    print("Figures written to results/figures/ (png + pdf, 600 dpi)")


if __name__ == "__main__":
    main()
