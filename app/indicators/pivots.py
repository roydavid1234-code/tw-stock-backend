"""ZigZag 樞紐點偵測 + 自動支撐壓力水平線。

對應投影片『水平線』訣竅：
> 在反彈兩次以上的地方畫出橫線

實作邏輯：
1. 用百分比門檻（預設 3%）的 ZigZag 找出顯著波峰/波谷。
2. 將波峰/波谷的價位以容差（預設 ±1.5%）分群。
3. 觸碰次數 >= 2 的群即為水平線：高於現價 -> 壓力；低於現價 -> 支撐。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import pandas as pd

PivotKind = Literal["peak", "trough"]


@dataclass
class Pivot:
    date: str
    price: float
    kind: PivotKind

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HorizontalLine:
    price: float
    role: Literal["support", "resistance"]
    touches: int
    touch_dates: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def zigzag(df: pd.DataFrame, pct_threshold: float = 0.03) -> list[Pivot]:
    """簡化 ZigZag：以收盤價計算，若反向變動超過 pct_threshold 即確認轉折。"""
    if len(df) < 3:
        return []

    pivots: list[Pivot] = []
    direction: int = 0  # 1 up, -1 down, 0 unknown
    last_extreme_idx = 0
    last_extreme_price = float(df["close"].iloc[0])

    for i in range(1, len(df)):
        price = float(df["close"].iloc[i])
        if direction >= 0 and price > last_extreme_price:
            last_extreme_idx = i
            last_extreme_price = price
            direction = 1
        elif direction <= 0 and price < last_extreme_price:
            last_extreme_idx = i
            last_extreme_price = price
            direction = -1

        if direction == 1 and price < last_extreme_price * (1 - pct_threshold):
            pivots.append(
                Pivot(
                    date=str(df["date"].iloc[last_extreme_idx].date()),
                    price=float(df["high"].iloc[last_extreme_idx]),
                    kind="peak",
                )
            )
            direction = -1
            last_extreme_idx = i
            last_extreme_price = price
        elif direction == -1 and price > last_extreme_price * (1 + pct_threshold):
            pivots.append(
                Pivot(
                    date=str(df["date"].iloc[last_extreme_idx].date()),
                    price=float(df["low"].iloc[last_extreme_idx]),
                    kind="trough",
                )
            )
            direction = 1
            last_extreme_idx = i
            last_extreme_price = price

    return pivots


def find_horizontal_lines(
    pivots: list[Pivot],
    current_price: float,
    tolerance: float = 0.015,
    min_touches: int = 2,
) -> list[HorizontalLine]:
    if not pivots:
        return []
    points = sorted([p for p in pivots if p.price > 0], key=lambda p: p.price)
    clusters: list[list[Pivot]] = []
    for p in points:
        if not clusters:
            clusters.append([p])
            continue
        center = sum(x.price for x in clusters[-1]) / len(clusters[-1])
        if center > 0 and abs(p.price - center) / center <= tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    lines: list[HorizontalLine] = []
    for cluster in clusters:
        if len(cluster) < min_touches:
            continue
        price = sum(x.price for x in cluster) / len(cluster)
        role: Literal["support", "resistance"] = (
            "resistance" if price >= current_price else "support"
        )
        lines.append(
            HorizontalLine(
                price=round(price, 2),
                role=role,
                touches=len(cluster),
                touch_dates=[x.date for x in cluster],
            )
        )
    # 依與現價距離由近到遠
    lines.sort(key=lambda l: abs(l.price - current_price))
    return lines
