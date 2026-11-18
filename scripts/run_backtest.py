"""Phase 2: odds-only backtest over every catalogue match, and its report.

    uv run python scripts/run_backtest.py --config configs/backtest.yaml \
        --interim data/interim --catalogue data/catalogue.csv \
        --atp data/raw/tennis_atp/atp_matches_202[2-4].csv --report reports/phase2.md

Synthetic demo:  just demo   (or see the justfile)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from courtedge.data import atp
from courtedge.data.catalogue import load_catalogue
from courtedge.market.backtest import DEFAULT_COMMISSION, BacktestConfig, run_backtest
from courtedge.market.inputs import build_inputs
from courtedge.report import plots
from courtedge.report.metrics import (
    bootstrap_by_group,
    brier,
    calibration_table,
    expected_calibration_error,
    log_loss,
)


def score_table(preds: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Brier and log loss for model and market with match-level bootstrap CIs."""
    rows = []
    for name, col in [
        ("Market (de-vigged LTP)", "market_p"),
        ("Model (Markov, delayed)", "model_match"),
    ]:
        for metric, fn in [("Brier", brier), ("Log loss", log_loss)]:
            est, lo, hi = bootstrap_by_group(
                preds, "match_id", lambda d, c=col, f=fn: f(d[c], d["outcome"]), n_boot, seed
            )
            rows.append(dict(forecast=name, metric=metric, value=est, lo=lo, hi=hi))
    diff = bootstrap_by_group(
        preds,
        "match_id",
        lambda d: brier(d.model_match, d.outcome) - brier(d.market_p, d.outcome),
        n_boot,
        seed,
    )
    rows.append(
        dict(forecast="Model - market", metric="Brier", value=diff[0], lo=diff[1], hi=diff[2])
    )
    return pd.DataFrame(rows)


def pnl_summary(results: pd.DataFrame, n_boot: int, seed: int) -> dict:
    staked = results["staked"].sum()
    roi = lambda d: d["net"].sum() / d["staked"].sum() if d["staked"].sum() > 0 else np.nan  # noqa: E731
    est, lo, hi = bootstrap_by_group(results, "match_id", roi, n_boot, seed)
    return dict(
        n_bets=int(results["n_bets"].sum()),
        matches_with_bets=int((results.n_bets > 0).sum()),
        staked=float(staked),
        net=float(results["net"].sum()),
        commission=float(results["commission"].sum()),
        roi=est,
        roi_lo=lo,
        roi_hi=hi,
    )


def calibration_figure(preds: pd.DataFrame, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=plots.GRID, linewidth=1.5)
    for i, (label, col) in enumerate([("Market", "market_p"), ("Model", "model_match")]):
        t = calibration_table(preds[col], preds["outcome"], bins=10)
        ax.plot(t.mean_p, t.freq, color=plots.SERIES[i], marker="o", markersize=5, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast P(A wins match)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return plots.save(fig, path)


def pnl_figure(
    bets: pd.DataFrame, results: pd.DataFrame, path: Path, title: str, n_boot: int, seed: int
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(bets):
        b = bets.sort_values("time_utc").reset_index(drop=True)
        # Commission is charged per market, so attribute each market's commission to its last bet.
        comm = results.set_index("match_id")["commission"]
        b["pnl"] = np.where(
            b.side == b.match_id.map(_winners(results)), b.stake * (b.odds - 1), -b.stake
        )
        last = b.groupby("match_id").tail(1).index
        b.loc[last, "pnl"] -= b.loc[last, "match_id"].map(comm).to_numpy()
        cum = b["pnl"].cumsum().to_numpy()
        x = np.arange(1, len(b) + 1)
        # Bootstrap band: resample matches, rebuild the cumulative curve on the same x grid.
        rng = np.random.default_rng(seed)
        per_match = {m: d["pnl"].to_numpy() for m, d in b.groupby("match_id", sort=False)}
        ids = list(per_match)
        curves = np.full((n_boot, len(b)), np.nan)
        for k in range(n_boot):
            seq = np.concatenate([per_match[ids[j]] for j in rng.integers(len(ids), size=len(ids))])
            c = np.cumsum(seq)[: len(b)]
            curves[k, : len(c)] = c
        lo, hi = np.nanquantile(curves, [0.025, 0.975], axis=0)
        ax.fill_between(
            x, lo, hi, color=plots.BLUE_FILL, alpha=0.6, linewidth=0, label="95% bootstrap band"
        )
        ax.plot(x, cum, color=plots.SERIES[0], label="Cumulative net P&L")
        ax.axhline(0, color=plots.INK_2, linewidth=0.8)
        ax.legend(loc="upper left")
    ax.set_xlabel("Bet number (time order)")
    ax.set_ylabel("Net P&L (stake units)")
    ax.set_title(title)
    return plots.save(fig, path)


def _winners(results: pd.DataFrame) -> pd.Series:
    return results.set_index("match_id")["winner"]


def fmt_ci(v, lo, hi, digits=4):
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/backtest.yaml")
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--catalogue", default="data/catalogue.csv")
    ap.add_argument("--atp", nargs="+", required=True, help="atp_matches_YYYY.csv files")
    ap.add_argument("--report", default="reports/phase2.md")
    ap.add_argument("--synthetic", action="store_true", help="label every output SYNTHETIC")
    ap.add_argument("--truth", help="demo truth.csv, adds a serve-estimate diagnostic")
    args = ap.parse_args(argv)

    conf = yaml.safe_load(Path(args.config).read_text())
    cfg = BacktestConfig.from_dict(conf["backtest"])
    serve_cfg = atp.ServeModelConfig(**conf.get("serve_model", {}))
    n_boot, seed = conf["bootstrap"]["n"], conf["bootstrap"]["seed"]
    tag = "SYNTHETIC: " if args.synthetic else ""
    report = Path(args.report)
    fig_dir = report.parent / "figures"
    stem = report.stem

    interim = Path(args.interim)
    catalogue = load_catalogue(args.catalogue)
    matches = pd.read_parquet(interim / "matches.parquet")
    points = pd.read_parquet(interim / "points.parquet")
    bf = pd.read_parquet(interim / "betfair.parquet")

    long = atp.to_long(atp.load_matches(args.atp))
    serve = atp.estimate_catalogue(long, catalogue, serve_cfg)
    serve.to_parquet(interim / "serve_probs.parquet", index=False)

    inputs, info = build_inputs(catalogue, matches, points, bf, serve)
    preds, bets, results = run_backtest(inputs, cfg)
    results["winner"] = [m.winner for m in inputs]
    for name, df in [("predictions", preds), ("bets", bets), ("results", results)]:
        df.to_parquet(interim / f"{stem}_{name}.parquet", index=False)

    scores = score_table(preds, n_boot, seed)
    pnl = pnl_summary(results, n_boot, seed)
    ece = {
        c: expected_calibration_error(preds[c], preds.outcome) for c in ("market_p", "model_match")
    }
    pre = preds.groupby("match_id").head(1)  # first in-play tick: closest to pre-match
    calib = calibration_figure(
        preds, fig_dir / f"{stem}_calibration.png", f"{tag}In-play calibration"
    )
    pnl_fig = pnl_figure(
        bets, results, fig_dir / f"{stem}_pnl.png", f"{tag}Simulated P&L", n_boot, seed
    )

    rates = [m.commission(cfg) for m in inputs]
    timing = info.loc[info.used, "timing"].value_counts().to_dict()
    lines = [
        f"# {tag}Phase 2: odds-only backtest",
        "",
    ]
    if args.synthetic:
        lines += [
            "> **SYNTHETIC DATA.** Every number below comes from the generated demo world",
            "> (`scripts/make_demo_data.py`), not from real Betfair prices. It exercises the",
            "> pipeline end to end. Regenerate this report from real data before quoting it.",
            "",
        ]
    lines += [
        "## Setup",
        "",
        f"- Matches: **{int(info.used.sum())}** used of {len(info)} in the catalogue; "
        f"{len(preds):,} in-play decision ticks; point timing: {timing}.",
        f"- Model: Markov engine on pre-match serve probabilities (52-week, surface-shrunk, "
        f"opponent-adjusted) with the score known at **t - {cfg.delay_s:.0f} s**.",
        f"- Market: last traded prices de-vigged ({cfg.devig_method}).",
        f"- Strategy: back when |model - market| > {cfg.threshold}, {cfg.staking.rule} staking, "
        f"{cfg.slippage_ticks} tick slippage, at most {cfg.max_bets} bets per market, "
        f"{cfg.min_gap_s:.0f} s apart.",
        f"- Commission: per-market `marketBaseRate` from the Betfair files "
        f"(range {min(rates):.1%}-{max(rates):.1%}; fallback {DEFAULT_COMMISSION:.0%}). It is "
        "charged on net market winnings.",
        f"- Bootstrap: {n_boot} resamples of whole matches.",
        "",
        "## Forecast accuracy (in-play, P(A wins match))",
        "",
        "| Forecast | Metric | Value [95% CI] |",
        "|---|---|---|",
    ]
    for r in scores.itertuples():
        lines.append(f"| {r.forecast} | {r.metric} | {fmt_ci(r.value, r.lo, r.hi)} |")
    lines += [
        "",
        f"ECE: market {ece['market_p']:.4f}, model {ece['model_match']:.4f}. "
        f"At the first in-play tick: Brier market {brier(pre.market_p, pre.outcome):.4f}, "
        f"model {brier(pre.model_match, pre.outcome):.4f} (n = {len(pre)}).",
        "",
        f"![calibration](figures/{calib.name})",
        "",
        "## Simulated P&L",
        "",
        "| Bets | Matches with bets | Staked | Commission | Net | ROI [95% CI] |",
        "|---|---|---|---|---|---|",
        f"| {pnl['n_bets']} | {pnl['matches_with_bets']} | {pnl['staked']:.2f} | "
        f"{pnl['commission']:.2f} | {pnl['net']:.2f} | "
        f"{fmt_ci(pnl['roi'], pnl['roi_lo'], pnl['roi_hi'], 3)} |",
        "",
        f"![pnl](figures/{pnl_fig.name})",
        "",
    ]
    if args.truth:
        truth = pd.read_csv(args.truth).merge(serve, on="match_id")
        err = np.r_[truth.pa - truth.true_pa, truth.pb - truth.true_pb]
        mkt = np.r_[truth.market_pa - truth.true_pa, truth.market_pb - truth.true_pb]
        lines += [
            "## Serve-estimate diagnostic (synthetic only)",
            "",
            f"Estimated vs true serve-point probability: RMSE **{np.sqrt(np.mean(err**2)):.4f}**, "
            f"bias {err.mean():+.4f}. The demo market's skill error has RMSE "
            f"{np.sqrt(np.mean(mkt**2)):.4f}. The market knows each match's form and the stats "
            "model does not, so the model should lose to the market here.",
            "",
        ]
    lines += [
        "## Limitations",
        "",
        "- BASIC data: one-minute last traded prices, no volume or order book. Fills at LTP minus "
        "a slippage tick are optimistic for larger stakes.",
        "- Point timing is `elapsed` (Sackmann ElapsedTime anchored at the in-play minute, so "
        "up to ±60 s of anchor error) or `estimated` (spread between in-play start and suspension, "
        "which uses the match end time). Phase 4 replaces both with video alignment.",
        "- Thresholds and staking were fixed in `configs/backtest.yaml` before running. Any "
        "other configuration tried must be added to this report.",
        "",
        f"Config: `{args.config}`. Full parameters: `{asdict(cfg)}`",
    ]
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
