"""單根 K 線型態分類。

對應投影片『陽線圖形與意義』：
- 大陽線：實體 > 近 20 根平均實體 * 1.5
- 中陽線：實體 > 平均實體
- 小陽線：實體 < 平均實體 * 0.6
- 十字線（doji）：實體 / 全長 < 0.1
- 帶上影線：上影 > 實體 * 1.5
- 帶下影線：下影 > 實體 * 1.5
- 光頭光腳：上影 + 下影 < 全長 * 0.05

回傳 list[CandleTag]，每根 K 可被多個 tag 同時命中。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import pandas as pd

Direction = Literal["bullish", "bearish", "neutral"]


@dataclass
class CandleTag:
    date: str
    direction: Direction
    body_size: str  # "large" | "medium" | "small" | "doji"
    shadows: list[str]  # ["upper", "lower", "marubozu"]
    meaning: str  # 中文解讀，對應投影片

    def to_dict(self) -> dict:
        return asdict(self)


def _meaning(direction: Direction, body: str, shadows: list[str]) -> str:
    if body == "doji":
        return "多空均衡，市場轉折點"
    if body == "large":
        return "多方絕對優勢" if direction == "bullish" else "空方絕對優勢"
    if body == "medium":
        base = "多方力道" if direction == "bullish" else "空方力道"
        if "upper" in shadows:
            return f"{base}（上方壓力較強）"
        if "lower" in shadows:
            return f"{base}（下方有支撐）"
        return base
    if body == "small":
        if "upper" in shadows and "lower" in shadows:
            return "與壓力及支撐的對抗，方向尚未明確"
        if "lower" in shadows:
            return "下方支撐有壓力，越長越強"
        if "upper" in shadows:
            return "上方壓力比下方支撐更強"
        return "一方面有希望，方向待確認"
    return ""


def classify_candles(df: pd.DataFrame, lookback: int = 20) -> list[CandleTag]:
    if len(df) == 0:
        return []
    work = df.copy()
    work["body"] = (work["close"] - work["open"]).abs()
    work["range"] = work["high"] - work["low"]
    work["upper_shadow"] = work["high"] - work[["open", "close"]].max(axis=1)
    work["lower_shadow"] = work[["open", "close"]].min(axis=1) - work["low"]
    work["avg_body"] = work["body"].rolling(window=lookback, min_periods=5).mean()

    tags: list[CandleTag] = []
    for _, row in work.iterrows():
        avg = row["avg_body"]
        if pd.isna(avg) or avg == 0:
            continue
        body, total = row["body"], row["range"]
        if total == 0:
            continue

        direction: Direction = (
            "bullish" if row["close"] > row["open"]
            else "bearish" if row["close"] < row["open"]
            else "neutral"
        )

        if total > 0 and body / total < 0.1:
            body_size = "doji"
        elif body > avg * 1.5:
            body_size = "large"
        elif body < avg * 0.6:
            body_size = "small"
        else:
            body_size = "medium"

        shadows: list[str] = []
        if row["upper_shadow"] > body * 1.5 and body > 0:
            shadows.append("upper")
        if row["lower_shadow"] > body * 1.5 and body > 0:
            shadows.append("lower")
        if (row["upper_shadow"] + row["lower_shadow"]) < total * 0.05:
            shadows.append("marubozu")

        tags.append(
            CandleTag(
                date=str(row["date"].date()),
                direction=direction,
                body_size=body_size,
                shadows=shadows,
                meaning=_meaning(direction, body_size, shadows),
            )
        )
    return tags


def latest_tag(tags: list[CandleTag]) -> CandleTag | None:
    return tags[-1] if tags else None
