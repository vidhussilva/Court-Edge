"""Stake sizing for back bets on a two-runner exchange market.

Commission is charged on net market winnings, so the effective net odds of a winning
back bet are b = (odds - 1) * (1 - commission). Using the net odds in Kelly stops
the strategy from sizing up bets whose only edge is the commission it ignored.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StakingConfig:
    rule: str = "flat"  # "flat" or "kelly"
    stake: float = 1.0  # flat stake per bet
    bankroll: float = 100.0  # notional bankroll for Kelly (fixed per match, no compounding)
    kelly_fraction: float = 0.25  # fractional Kelly to limit drawdowns from model error
    kelly_cap: float = 0.02  # never risk more than this share of the bankroll on one bet


def kelly_fraction(p: float, odds: float, commission: float = 0.0) -> float:
    """Full-Kelly share of the bankroll for a back bet: (p*b - (1-p)) / b, floored at 0."""
    b = (odds - 1.0) * (1.0 - commission)
    if b <= 0:
        return 0.0
    return max((p * b - (1.0 - p)) / b, 0.0)


def stake_for(p: float, odds: float, cfg: StakingConfig, commission: float = 0.0) -> float:
    if cfg.rule == "flat":
        return cfg.stake
    if cfg.rule == "kelly":
        f = min(cfg.kelly_fraction * kelly_fraction(p, odds, commission), cfg.kelly_cap)
        return cfg.bankroll * f
    raise ValueError(f"unknown staking rule {cfg.rule!r}")
