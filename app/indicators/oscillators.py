"""擺盪指標：KD（隨機指標）、RSI、MACD。

公式皆為台股常用版本：
- KD：9 日，K、D 用 1/3 平滑（K = 2/3·K_prev + 1/3·RSV），J = 3K - 2D
- RSI：14 日，Wilder smoothing
- MACD：12 / 26 / 9（EMA-based）

訊號規則：
- KD 黃金交叉 (低檔 <20 加強)：買入；死亡交叉 (高檔 >80 加強)：賣出
- RSI > 70 超買 / < 30 超賣
- MACD 柱狀體由負轉正（紅柱）：買入；由正轉負（綠柱）：賣出
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import pandas as pd


def add_kd(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    out = df.copy()
    low_n = out["low"].rolling(window=period, min_periods=period).min()
    high_n = out["high"].rolling(window=period, min_periods=period).max()
    rsv = (out["close"] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)

    k_values: list[float] = []
    d_values: list[float] = []
    k_prev = 50.0
    d_prev = 50.0
    for v in rsv:
        if pd.isna(v):
            k_values.append(float("nan"))
            d_values.append(float("nan"))
            continue
        k_cur = k_prev * 2 / 3 + v / 3
        d_cur = d_prev * 2 / 3 + k_cur / 3
        k_values.append(k_cur)
        d_values.append(d_cur)
        k_prev, d_prev = k_cur, d_cur

    out["kd_k"] = k_values
    out["kd_d"] = d_values
    out["kd_j"] = 3 * out["kd_k"] - 2 * out["kd_d"]
    return out


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing：第一段用平均，之後用 EMA
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    out["rsi"] = 100 - 100 / (1 + rs)
    return out


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    out = df.copy()
    ema_fast = out["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=slow, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=signal, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    return out


# ---------- 訊號偵測 ----------

@dataclass
class KDSignal:
    date: str
    kind: Literal["golden_oversold", "golden", "death_overbought", "death", "overbought", "oversold"]
    k: float
    d: float
    verdict: Literal["buy", "sell", "watch"]
    label_zh: str
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RSISignal:
    date: str
    rsi: float
    kind: Literal["overbought", "oversold"]
    verdict: Literal["sell", "buy"]
    label_zh: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MACDSignal:
    date: str
    kind: Literal["hist_turn_positive", "hist_turn_negative", "golden", "death"]
    macd: float
    signal: float
    verdict: Literal["buy", "sell"]
    label_zh: str

    def to_dict(self) -> dict:
        return asdict(self)


def detect_kd_crosses(df: pd.DataFrame, recent_bars: int = 20) -> list[KDSignal]:
    """偵測最近 N 根 K 內的 KD 交叉與超買超賣。"""
    sub = df.dropna(subset=["kd_k", "kd_d"]).tail(recent_bars + 1).reset_index(drop=True)
    signals: list[KDSignal] = []
    for i in range(1, len(sub)):
        date_str = str(sub.loc[i, "date"].date())
        k_prev, d_prev = sub.loc[i - 1, "kd_k"], sub.loc[i - 1, "kd_d"]
        k_cur, d_cur = sub.loc[i, "kd_k"], sub.loc[i, "kd_d"]

        if k_prev <= d_prev and k_cur > d_cur:
            oversold = k_cur < 20 and d_cur < 20
            signals.append(KDSignal(
                date=date_str,
                kind="golden_oversold" if oversold else "golden",
                k=float(k_cur), d=float(d_cur), verdict="buy",
                label_zh="KD 黃金交叉" + (" (低檔 <20)" if oversold else ""),
                detail=f"K={k_cur:.1f} 上穿 D={d_cur:.1f}" + ("，位於超賣區，買入訊號強化" if oversold else ""),
            ))
        elif k_prev >= d_prev and k_cur < d_cur:
            overbought = k_cur > 80 and d_cur > 80
            signals.append(KDSignal(
                date=date_str,
                kind="death_overbought" if overbought else "death",
                k=float(k_cur), d=float(d_cur), verdict="sell",
                label_zh="KD 死亡交叉" + (" (高檔 >80)" if overbought else ""),
                detail=f"K={k_cur:.1f} 下穿 D={d_cur:.1f}" + ("，位於超買區，賣出訊號強化" if overbought else ""),
            ))

    # 最後一根：超買 / 超賣（不重複）
    if not signals or signals[-1].date != str(sub["date"].iloc[-1].date()):
        last = sub.iloc[-1]
        if last["kd_k"] > 80 and last["kd_d"] > 80:
            signals.append(KDSignal(
                date=str(last["date"].date()), kind="overbought",
                k=float(last["kd_k"]), d=float(last["kd_d"]), verdict="watch",
                label_zh="KD 高檔鈍化（超買）",
                detail=f"K={last['kd_k']:.1f}, D={last['kd_d']:.1f}，警惕回檔",
            ))
        elif last["kd_k"] < 20 and last["kd_d"] < 20:
            signals.append(KDSignal(
                date=str(last["date"].date()), kind="oversold",
                k=float(last["kd_k"]), d=float(last["kd_d"]), verdict="watch",
                label_zh="KD 低檔鈍化（超賣）",
                detail=f"K={last['kd_k']:.1f}, D={last['kd_d']:.1f}，可能醞釀反彈",
            ))
    return signals


def detect_rsi_signal(df: pd.DataFrame) -> RSISignal | None:
    if "rsi" not in df.columns or df["rsi"].dropna().empty:
        return None
    last_date = str(df["date"].iloc[-1].date())
    rsi_val = float(df["rsi"].iloc[-1])
    if rsi_val > 70:
        return RSISignal(date=last_date, rsi=rsi_val, kind="overbought",
                         verdict="sell", label_zh=f"RSI 超買 ({rsi_val:.1f})")
    if rsi_val < 30:
        return RSISignal(date=last_date, rsi=rsi_val, kind="oversold",
                         verdict="buy", label_zh=f"RSI 超賣 ({rsi_val:.1f})")
    return None


def detect_macd_signals(df: pd.DataFrame, recent_bars: int = 10) -> list[MACDSignal]:
    sub = df.dropna(subset=["macd", "macd_signal", "macd_hist"]).tail(recent_bars + 1).reset_index(drop=True)
    signals: list[MACDSignal] = []
    for i in range(1, len(sub)):
        h_prev = sub.loc[i - 1, "macd_hist"]
        h_cur = sub.loc[i, "macd_hist"]
        date_str = str(sub.loc[i, "date"].date())
        if h_prev <= 0 and h_cur > 0:
            signals.append(MACDSignal(
                date=date_str, kind="hist_turn_positive",
                macd=float(sub.loc[i, "macd"]), signal=float(sub.loc[i, "macd_signal"]),
                verdict="buy", label_zh="MACD 柱狀翻紅",
            ))
        elif h_prev >= 0 and h_cur < 0:
            signals.append(MACDSignal(
                date=date_str, kind="hist_turn_negative",
                macd=float(sub.loc[i, "macd"]), signal=float(sub.loc[i, "macd_signal"]),
                verdict="sell", label_zh="MACD 柱狀翻綠",
            ))
    return signals
