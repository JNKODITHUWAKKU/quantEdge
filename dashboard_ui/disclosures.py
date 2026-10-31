"""Research labels, mandate, provenance and funding limitations."""

from dataclasses import dataclass
import numpy as np
import pandas as pd

SELECTED_STRATEGY = "sample_gmv"
SELECTED_MONITOR = "garch_t"
GUARDED_EVT_METHOD = "garch_pot_guarded"
PRIMARY_COST_BPS = 5.0
PRIMARY_CONFIDENCE = 0.99
INITIAL_CAPITAL_USD = 10_000_000.0
INCEPTION_DATE = "2019-12-31"
DEFAULT_REVIEW_SHARE = 0.50

CONFIDENCE_LEVELS = (0.9, 0.95, 0.99)
COST_SCENARIOS_BPS = (0.0, 5.0, 10.0)
SPENDING_RATES = (0.0, 0.02, 0.03, 0.04)
LOSS_TOLERANCES = (0.15, 0.20, 0.25)

ASSET_NAMES = {
    "SPY": "US equities",
    "EFA": "Developed equities",
    "EEM": "Emerging equities",
    "AGG": "US bonds",
    "GLD": "Gold",
    "XLE": "Energy",
    "XLK": "Technology",
}
EXPECTED_ASSETS = frozenset(ASSET_NAMES)

METHOD_LABELS = {
    "historical": "Historical simulation",
    "gaussian": "Gaussian parametric",
    "monte_carlo": "Gaussian Monte Carlo",
}

CONDITIONAL_LABELS = {
    "ewma_normal": "EWMA · normal",
    "ewma_fhs": "EWMA · filtered historical",
    "garch_normal": "GARCH · normal",
    "garch_t": "GARCH · standardized Student-t",
    "garch_fhs": "GARCH · filtered historical",
    "raw_pot_guarded": "Raw EVT · guarded policy",
    "garch_pot_guarded": "Filtered EVT · guarded policy",
}

MANDATE = """
The illustrative USD10m university endowment supports scholarships, research,
faculty development and capital projects. The permitted asset universe is SPY,
EFA, EEM, AGG, GLD, XLE and XLK. The selected research portfolio is long-only,
fully invested Sample GMV. There is no leverage or additional concentration cap.

US CPI and USD are provisional liability proxies. Actual payout, liability
currency/basket, institutional loss limits, fees and liquidity needs require
human ownership. Spending of 0/2/3/4% of initial real capital and loss tolerances
of 15/20/25% are post-evaluation managerial illustrations, not adopted terms.
The permitted instrument universe is not expanded to manufacture a cash reserve.

The research decision is HOLD / NO TRADE. Correct numerical optimization does
not establish endowment suitability or investment approval.
"""

RECOVERY = """
Development ends on 31 December 2014. The original selection used 2015–2019
validation and selected Sample GMV with standardized Student-t GARCH monitoring.
The original selection is fixed in the Python settings; detailed selection records remain in the separate full research release. This application does not select
or tune a replacement using final outcomes.

Final retrospective outcomes run from 2 January 2020 to 23 October 2026.
Real purchasing power ends on 30 September 2026. Original final results had
already been examined before the disclosed Step 6 recovery. Original lost
pre-recovery bytes cannot be certified. Re-executing the recovered procedure
does not create a pristine holdout, real-time issuance record or live track record.

Signals use trailing 1,260 daily simple returns and execute at the following
session close; the next return uses the new holdings. Holdings drift between
trades. Each risk origin uses its known exposure to recast the trailing asset
returns into one centered portfolio series. Forecasts are persisted before
outcomes are joined. Final-day targets are indicative and unexecuted.

Saved provider history can be revised. Fifty-one large moves lack independent
human sign-off; twelve adjustment events reconcile against the same provider
only. No new data is fetched. ADF rejection does not establish iid returns.
Sparse tail events and dependent observations limit asymptotic tests. EVT
bootstrap intervals omit threshold-selection and model uncertainty. Basel
traffic-light references concern a 250-observation window and banking scope;
they do not approve an endowment allocation.
"""

FUNDING = """
Prices are adjusted for reinvested distributions. Self-financing accounting
solves proportional trading costs on both purchases and sales; dividends and
embedded fund expenses are not counted twice. Cost scenarios are 0/5/10 bp
assumptions, not observed execution costs. Taxes, advisory fees, market impact
and distress costs are not established.

Real wealth divides nominal wealth by CPI relative to December 2019. Missing
October 2025 CPI is retained as a gap. Illustrative annual real withdrawals are
paid through pro-rata December sales, including proportional sale costs. The
saved terminal withdrawal coefficient describes this particular historical
path; it is not a forecast of safe spending. Negative real-principal capacity
is reported honestly rather than clipped into a feasible positive payout.

Nominal variance, real purchasing power, observed drawdown, counterfactual
stress replay and risk calibration answer different questions. Passing a
loss-tolerance screen does not cure a failed real-capital floor or failed
calibration. No passing combination grants investment approval.
"""

METHODS = {**METHOD_LABELS, **CONDITIONAL_LABELS}

PORTFOLIOS = {
    "sample_gmv": "Selected research rule · sample GMV",
    "policy_40_60": "Comparator · 40% SPY / 60% AGG",
    "equal_weight": "Comparator · equal weight",
    "agg_only": "Comparator · AGG only",
    "shrinkage_gmv": "Comparator · shrinkage GMV",
}


@dataclass(frozen=True)
class MandateItem:
    """Single row of the display-only managerial mandate proposal."""

    item: str
    base_case: str


MANDATE_PROPOSAL = (
    MandateItem("Status", "Post-evaluation proposal; institutional adoption pending"),
    MandateItem("Capital / liability proxy", "USD10m / USD and US CPI"),
    MandateItem(
        "Horizon / review",
        "20-year planning horizon; annual committee review; quarterly research rebalancing",
    ),
    MandateItem(
        "Return objective",
        "At least 3% annualized net real total return before withdrawals over the planning horizon",
    ),
    MandateItem(
        "Spending",
        "USD300,000 annual fixed-real December withdrawal (3% of initial capital)",
    ),
    MandateItem(
        "Real-capital floor",
        "At least USD10m in base-month purchasing power after withdrawals",
    ),
    MandateItem(
        "Loss screens",
        "20% maximum nominal drawdown, largest replay loss and joint-shock loss",
    ),
    MandateItem(
        "Liquidity planning",
        "One year of payout = USD300,000; no cash asset added; ETF liquidation feasibility unverified",
    ),
    MandateItem(
        "Calibration",
        "Retain declared 95% / 99% coverage and independence checks; current monitor fails",
    ),
    MandateItem(
        "Evidence boundary",
        "Historical 2020–2026 screen; no 20-year forecast or assurance of future compliance",
    ),
    MandateItem(
        "Decision",
        "DO NOT DEPLOY as the complete endowment portfolio under the assessed assumptions",
    ),
)


def proposed_mandate() -> pd.DataFrame:
    """Display-only managerial proposal; never used by model selection or fitting."""
    return pd.DataFrame(
        [(row.item, row.base_case) for row in MANDATE_PROPOSAL],
        columns=["Proposed mandate item", "Proposed base case"],
    )


def risk_budget_review(
    components: pd.DataFrame, review_share: float = DEFAULT_REVIEW_SHARE
) -> pd.DataFrame:
    """Expose signed Gaussian ES allocations and a proposed review trigger."""
    frame = components[
        ["asset", "weight", "marginal_var", "var_component", "es_component"]
    ].copy()
    positive_total = frame.es_component.clip(lower=0).sum()
    frame["positive_es_share"] = (
        frame.es_component.clip(lower=0) / positive_total
        if positive_total > 0
        else np.nan
    )
    frame["proposed_review_trigger"] = review_share
    frame["sizing_review"] = np.where(
        frame.positive_es_share.isna(),
        "NOT ASSESSABLE",
        np.where(
            frame.positive_es_share > review_share,
            "REVIEW BEFORE ADDING EXPOSURE",
            "NO TRIGGER; NOT A SIZING APPROVAL",
        ),
    )
    return frame
