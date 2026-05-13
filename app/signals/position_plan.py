"""建議倉位 + 進場分批 + 停損 + 目標價 + 風險報酬比。

倉位邏輯（依綜合評分）：
| 評分 | 方向 | 目標倉位 |
| ≥ +80 | 多 | 80% |
| +60~79 | 多 | 65% |
| +40~59 | 多 | 50% |
| +20~39 | 多（試單） | 35% |
| ±19 | 觀望 | 0% |
| -20~-39 | 減碼 | 留 40% |
| -40~-59 | 減碼 | 留 25% |
| -60~-79 | 減碼 | 留 10% |
| ≤ -80 | 出清 | 0% |

分批進場邏輯（買入方向）：
- 第 1 批 40%：現價試單
- 第 2 批 35%：回測最近支撐 +0.5%（沒支撐則固定 -3%）
- 第 3 批 25%：突破最近壓力 +1%（沒壓力或太遠則跳過）

停損 / 目標：
- 停損 = max(最近支撐 × 0.97, 現價 × 0.93) 取較高者
- 目標 = 最近 2~3 條壓力線
- 風險報酬比 = (平均目標 − 平均進場) / (平均進場 − 停損)，≥ 2 為佳
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

from ..indicators.pivots import HorizontalLine
from .aggregator import Verdict


Direction = Literal["long", "reduce", "neutral", "exit"]


@dataclass
class PositionBatch:
    order: int
    pct_of_plan: int       # 佔本計畫總倉位的比例（加總 = 100）
    price: float
    condition: str         # 觸發條件描述（中文）
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TakeProfitTarget:
    price: float
    gain_pct: float        # 相對現價的潛在獲利%
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PositionPlan:
    direction: Direction
    direction_label: str
    target_position_pct: int
    summary: str
    strategy: str                # "auto"/"aggressive"/"balanced"/"conservative"/"equal"
    strategy_label: str          # 中文 + 一句說明
    batches: list[PositionBatch]
    stop_loss_price: float | None
    stop_loss_pct: float | None
    stop_loss_reason: str
    targets: list[TakeProfitTarget]
    risk_reward: float | None

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "direction_label": self.direction_label,
            "target_position_pct": self.target_position_pct,
            "summary": self.summary,
            "strategy": self.strategy,
            "strategy_label": self.strategy_label,
            "batches": [b.to_dict() for b in self.batches],
            "stop_loss_price": self.stop_loss_price,
            "stop_loss_pct": self.stop_loss_pct,
            "stop_loss_reason": self.stop_loss_reason,
            "targets": [t.to_dict() for t in self.targets],
            "risk_reward": self.risk_reward,
        }


# ---------- 分批策略 ----------
# 每個策略對應「3 批切法」與「2 批切法」（找不到第 3 批時用 2 批）
BATCH_SCHEMES: dict[str, dict[str, list[int]]] = {
    "aggressive":   {"3_batch": [50, 30, 20], "2_batch": [65, 35]},
    "balanced":     {"3_batch": [40, 35, 25], "2_batch": [55, 45]},
    "conservative": {"3_batch": [25, 35, 40], "2_batch": [35, 65]},
    "equal":        {"3_batch": [34, 33, 33], "2_batch": [50, 50]},
}

STRATEGY_LABELS: dict[str, str] = {
    "aggressive":   "🔴 激進型｜高信心一次重押現價",
    "balanced":     "🟢 標準型｜平均分散三批進場",
    "conservative": "🔵 保守型｜先試單後加碼（突破才重押）",
    "equal":        "⚪ 均分型｜風險完全分散，無偏好",
}


def resolve_strategy(strategy: str, score: int) -> str:
    """auto：依評分自動挑策略；其他直接回傳。"""
    if strategy != "auto":
        return strategy if strategy in BATCH_SCHEMES else "balanced"
    if score >= 60:
        return "aggressive"    # 強買 → 重押現價
    if score >= 20:
        return "balanced"
    if score <= -60:
        return "aggressive"    # 強賣 → 立刻大幅減碼
    if score <= -20:
        return "balanced"
    return "balanced"


def _target_position_long(score: int) -> int:
    if score >= 80: return 80
    if score >= 60: return 65
    if score >= 40: return 50
    if score >= 20: return 35
    return 0


def _target_position_reduce(score: int) -> int:
    """負分時建議「留下」的倉位比例。"""
    if score <= -80: return 0
    if score <= -60: return 10
    if score <= -40: return 25
    if score <= -20: return 40
    return 100  # 不需減碼


def _split_lines(lines: list[HorizontalLine], current_price: float) -> tuple[list[HorizontalLine], list[HorizontalLine]]:
    """回傳 (supports_below_current_desc, resistances_above_current_asc)。"""
    supports = sorted(
        [l for l in lines if l.role == "support" and l.price < current_price],
        key=lambda l: -l.price,  # 由高到低（最近支撐排第一）
    )
    resistances = sorted(
        [l for l in lines if l.role == "resistance" and l.price > current_price],
        key=lambda l: l.price,   # 由低到高（最近壓力排第一）
    )
    return supports, resistances


def _plan_long(score: int, current_price: float,
               supports: list[HorizontalLine], resistances: list[HorizontalLine],
               strategy: str = "balanced") -> PositionPlan:
    target_pct = _target_position_long(score)
    scheme = BATCH_SCHEMES.get(strategy, BATCH_SCHEMES["balanced"])

    nearest_support = supports[0] if supports else None
    nearest_resistance = resistances[0] if resistances else None
    has_resistance = (nearest_resistance is not None
                      and (nearest_resistance.price - current_price) / current_price < 0.10)
    weights = scheme["3_batch"] if has_resistance else scheme["2_batch"]

    batches: list[PositionBatch] = []
    # 第 1 批 — 現價試單
    batches.append(PositionBatch(
        order=1, pct_of_plan=weights[0], price=round(current_price, 2),
        condition="現價試單" if weights[0] < 50 else "現價建立主部位",
        note="先建立基本部位" if weights[0] < 50 else "信心高 → 一次重押現價",
    ))

    # 第 2 批 — 回測支撐加碼
    if nearest_support and (current_price - nearest_support.price) / current_price < 0.10:
        batches.append(PositionBatch(
            order=2, pct_of_plan=weights[1],
            price=round(nearest_support.price * 1.005, 2),
            condition=f"回測支撐 {nearest_support.price} 不破時加碼",
            note=f"觸碰 {nearest_support.touches} 次的支撐，獲得便宜籌碼",
        ))
    else:
        batches.append(PositionBatch(
            order=2, pct_of_plan=weights[1],
            price=round(current_price * 0.97, 2),
            condition="回檔約 3% 時加碼",
            note="無顯著支撐，以固定回檔幅度做加碼點",
        ))

    # 第 3 批 — 突破壓力加碼
    if has_resistance:
        batches.append(PositionBatch(
            order=3, pct_of_plan=weights[2],
            price=round(nearest_resistance.price * 1.01, 2),  # type: ignore[union-attr]
            condition=f"放量突破壓力 {nearest_resistance.price} 後加碼",  # type: ignore[union-attr]
            note=f"站上觸碰 {nearest_resistance.touches} 次的壓力，趨勢確認"  # type: ignore[union-attr]
                 + ("，重押確認趨勢" if weights[2] >= 40 else ""),
        ))

    # 停損：支撐距離現價 ≤ 7% 才用，否則用 -5% 機械停損
    if nearest_support and (current_price - nearest_support.price) / current_price <= 0.07:
        stop_price = round(nearest_support.price * 0.97, 2)
        stop_reason = f"跌破支撐 {nearest_support.price} 下方 3% 出場"
    else:
        stop_price = round(current_price * 0.95, 2)
        stop_reason = "現價下方 5% 機械停損（無近距支撐）"
    stop_pct = round((stop_price - current_price) / current_price * 100, 2)

    # 目標：用近距壓力線；若無，補上 +5% / +10% 兩個機械目標
    targets: list[TakeProfitTarget] = []
    for r in resistances[:3]:
        gain = round((r.price - current_price) / current_price * 100, 2)
        if gain <= 0.5:
            continue
        targets.append(TakeProfitTarget(
            price=r.price, gain_pct=gain,
            rationale=f"觸碰 {r.touches} 次的壓力 → 至少分批獲利 1/3",
        ))
    if not targets:
        targets = [
            TakeProfitTarget(
                price=round(current_price * 1.05, 2), gain_pct=5.0,
                rationale="無近距壓力，先設 +5% 機械停利點",
            ),
            TakeProfitTarget(
                price=round(current_price * 1.10, 2), gain_pct=10.0,
                rationale="第二段機械停利，視突破續攻情況決定是否續抱",
            ),
        ]

    # 風險報酬比（取前 2 個目標的平均 vs 平均進場）
    rr: float | None = None
    if targets and stop_price < current_price:
        avg_entry = sum(b.price * b.pct_of_plan for b in batches) / sum(b.pct_of_plan for b in batches)
        avg_target = sum(t.price for t in targets[:2]) / len(targets[:2])
        if avg_entry > stop_price:
            rr = round((avg_target - avg_entry) / (avg_entry - stop_price), 2)

    summary = (
        f"建議目標倉位 {target_pct}%，分 {len(batches)} 批進場；"
        f"停損 {stop_price}（{stop_pct:+.1f}%）"
        + (f"，風險報酬比約 {rr}:1" if rr else "")
    )

    return PositionPlan(
        direction="long",
        direction_label="建倉買進",
        target_position_pct=target_pct,
        summary=summary,
        strategy=strategy,
        strategy_label=STRATEGY_LABELS.get(strategy, strategy),
        batches=batches,
        stop_loss_price=stop_price,
        stop_loss_pct=stop_pct,
        stop_loss_reason=stop_reason,
        targets=targets,
        risk_reward=rr,
    )


def _plan_reduce(score: int, current_price: float,
                 supports: list[HorizontalLine], resistances: list[HorizontalLine],
                 strategy: str = "balanced") -> PositionPlan:
    keep_pct = _target_position_reduce(score)
    sell_pct = 100 - keep_pct

    # 減碼策略：用 2_batch 切法，但因方向相反需要翻轉「重押位置」：
    # - 激進：立刻多賣（65%）+ 反彈再賣 35%
    # - 標準：立刻 55% + 反彈 45%
    # - 保守：立刻只 35%（先賣一些）+ 反彈到壓力再賣 65%
    # - 均分：50 / 50
    scheme = BATCH_SCHEMES.get(strategy, BATCH_SCHEMES["balanced"])
    base = scheme["2_batch"]
    immediate = base[0]
    rebound = base[1]

    batches: list[PositionBatch] = []
    batches.append(PositionBatch(
        order=1, pct_of_plan=immediate, price=round(current_price, 2),
        condition="現價直接減碼",
        note=("立刻大幅減碼，鎖定利潤" if immediate >= 55
              else "先賣一部分試水，等反彈再多出"),
    ))

    nearest_resistance = resistances[0] if resistances else None
    if nearest_resistance and (nearest_resistance.price - current_price) / current_price < 0.08:
        batches.append(PositionBatch(
            order=2, pct_of_plan=rebound,
            price=round(nearest_resistance.price * 0.99, 2),
            condition=f"反彈到壓力 {nearest_resistance.price} 賣出剩餘",
            note=f"觸碰 {nearest_resistance.touches} 次的壓力是反彈天花板",
        ))
    else:
        batches.append(PositionBatch(
            order=2, pct_of_plan=rebound,
            price=round(current_price * 1.03, 2),
            condition="反彈 3% 賣出剩餘",
            note="無顯著壓力，以固定幅度為出場點",
        ))

    # 停損（出清剩餘部位）— 同樣加上 7% 距離限制
    nearest_support = supports[0] if supports else None
    if nearest_support and (current_price - nearest_support.price) / current_price <= 0.07:
        stop_price = round(nearest_support.price * 0.97, 2)
        stop_reason = f"跌破支撐 {nearest_support.price} 出清所有部位"
    else:
        stop_price = round(current_price * 0.95, 2)
        stop_reason = "現價下方 5% 機械停損出清"

    summary = (
        f"建議減碼 {sell_pct}%，留 {keep_pct}% 觀察；"
        f"停損 {stop_price}（跌破出清剩餘）"
    )

    return PositionPlan(
        direction="reduce" if keep_pct > 0 else "exit",
        direction_label="減碼觀察" if keep_pct > 0 else "全數出清",
        target_position_pct=keep_pct,
        summary=summary,
        strategy=strategy,
        strategy_label=STRATEGY_LABELS.get(strategy, strategy),
        batches=batches,
        stop_loss_price=stop_price,
        stop_loss_pct=round((stop_price - current_price) / current_price * 100, 2),
        stop_loss_reason=stop_reason,
        targets=[],
        risk_reward=None,
    )


def _plan_neutral(current_price: float) -> PositionPlan:
    return PositionPlan(
        direction="neutral",
        direction_label="觀望",
        target_position_pct=0,
        summary="多空訊號相抵，暫不進場；等型態突破或均線排列確立再行動",
        strategy="balanced",
        strategy_label=STRATEGY_LABELS["balanced"],
        batches=[],
        stop_loss_price=None,
        stop_loss_pct=None,
        stop_loss_reason="無持倉，無需停損",
        targets=[],
        risk_reward=None,
    )


def plan_position(verdict: Verdict, current_price: float,
                  horizontal_lines: list[HorizontalLine],
                  strategy: str = "auto") -> PositionPlan:
    supports, resistances = _split_lines(horizontal_lines, current_price)
    score = verdict.score
    resolved = resolve_strategy(strategy, score)
    if score >= 20:
        return _plan_long(score, current_price, supports, resistances, resolved)
    if score <= -20:
        return _plan_reduce(score, current_price, supports, resistances, resolved)
    return _plan_neutral(current_price)
