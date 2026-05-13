"""移動平均線與黃金/死亡交叉。

對應投影片『移動平均線』的四種狀態：
- 黃金交叉（短期上穿長期）：買入
- 死亡交叉（短期下穿長期）：賣出
- 長期下行中短期黃金交叉：不要買（假突破風險）
- 長期上行中短期死亡交叉：不要賣（多頭回檔）
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SHORT_TERM = [5, 10, 20]
LONG_TERM = [50, 100, 200]
DEFAULT_PAIRS = [(5, 20), (20, 60), (50, 200)]


@dataclass
class CrossSignal:
    short_window: int
    long_window: int
    date: str
    kind: str  # "golden" | "death"
    long_trend: str  # "up" | "down" | "flat"
    verdict: str  # "buy" | "sell" | "hold_no_buy" | "hold_no_sell"


def add_moving_averages(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    out = df.copy()
    for w in windows or (SHORT_TERM + LONG_TERM + [60]):
        out[f"sma{w}"] = out["close"].rolling(window=w, min_periods=w).mean()
    return out


def _long_trend(series: pd.Series, lookback: int = 10) -> str:
    if len(series.dropna()) < lookback + 1:
        return "flat"
    recent = series.dropna().iloc[-lookback - 1 :]
    slope = recent.iloc[-1] - recent.iloc[0]
    pct = slope / recent.iloc[0] if recent.iloc[0] else 0
    if pct > 0.01:
        return "up"
    if pct < -0.01:
        return "down"
    return "flat"


def detect_crosses(df: pd.DataFrame, pairs: list[tuple[int, int]] | None = None, recent_bars: int = 30) -> list[CrossSignal]:
    """偵測最近 N 根 K 內出現的黃金/死亡交叉，並依長期均線方向給出『不要買/不要賣』的修正建議。"""
    pairs = pairs or DEFAULT_PAIRS
    signals: list[CrossSignal] = []
    for short_w, long_w in pairs:
        s_col = f"sma{short_w}"
        l_col = f"sma{long_w}"
        if s_col not in df or l_col not in df:
            continue
        sub = df[["date", s_col, l_col]].dropna().tail(recent_bars + 1).reset_index(drop=True)
        if len(sub) < 2:
            continue
        for i in range(1, len(sub)):
            prev_s, prev_l = sub.loc[i - 1, s_col], sub.loc[i - 1, l_col]
            cur_s, cur_l = sub.loc[i, s_col], sub.loc[i, l_col]
            if prev_s <= prev_l and cur_s > cur_l:
                kind = "golden"
            elif prev_s >= prev_l and cur_s < cur_l:
                kind = "death"
            else:
                continue

            long_series = df[l_col].iloc[: df.index[df["date"] == sub.loc[i, "date"]].max() + 1]
            trend = _long_trend(long_series)

            if kind == "golden":
                verdict = "buy" if trend != "down" else "hold_no_buy"
            else:
                verdict = "sell" if trend != "up" else "hold_no_sell"

            signals.append(
                CrossSignal(
                    short_window=short_w,
                    long_window=long_w,
                    date=str(sub.loc[i, "date"].date()),
                    kind=kind,
                    long_trend=trend,
                    verdict=verdict,
                )
            )
    return signals


def latest_alignment(df: pd.DataFrame) -> dict:
    """回傳最後一根 K 的均線排列（多頭/空頭/糾結）。"""
    last = df.dropna(subset=[f"sma{w}" for w in [5, 20, 60]], how="any").iloc[-1] if len(df) else None
    if last is None:
        return {"state": "unknown"}
    s5, s20, s60 = last["sma5"], last["sma20"], last["sma60"]
    if s5 > s20 > s60:
        state = "bullish_alignment"
    elif s5 < s20 < s60:
        state = "bearish_alignment"
    else:
        state = "tangled"
    return {
        "state": state,
        "sma5": float(s5),
        "sma20": float(s20),
        "sma60": float(s60),
        "close": float(last["close"]),
    }
