"""綜合訊號評分 + AI 自然語言評估。

評分權重對齊提供的 6 張投影片：
- 圖 5「賣出警告/買入警告」的百分比軸（100% / 80% / 65% / 50%）
- 圖 6「買入/賣出」排行榜（買入 #1~#5 / 賣出 #6~#10）
- 圖 3「移動平均線」的真假黃金/死亡交叉
- 圖 4「水平線」的支撐壓力突破/跌破
- 圖 1「陽線圖形與意義」的單體 K 線

分數限制 [-100, +100]，對應 7 檔指示牌（圖 5）：
+80 ↑ 迎接暴漲 / +50~79 趕緊買入 / +20~49 偏多分批 / ±19 危險別碰 /
-20~-49 緩慢下跌 / -50~-79 趕快賣出 / -80 ↓ 防範暴跌
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import date, timedelta
from typing import Literal

from ..indicators.candle import CandleTag
from ..indicators.ma import CrossSignal
from ..indicators.oscillators import KDSignal, MACDSignal, RSISignal
from ..indicators.patterns import PatternMatch
from ..indicators.pivots import HorizontalLine


# 每個型態最大可貢獻分數（confidence=1.0 且 base_strength 已乘入時的滿分）
# 數值大致對齊圖 5 的 100/80/65/50% + 圖 6 的排行
PATTERN_MAX_IMPACT: dict[str, int] = {
    # 賣出側
    "double_top": 30,            # M頭 100%
    "triple_top": 32,            # 三重頂（M頭加強版）
    "head_shoulders_top": 28,    # 頭肩頂（賣出排行 #6 同檔）
    "bear_flag": 24,             # 下跌旗形 80%
    "inverted_v_top": 22,        # 倒V反轉 排行 #8
    "diamond_top": 20,           # 菱形頂 65%
    # 買入側
    "head_shoulders_bottom": 30,  # 頭肩底 買入排行 #1
    "double_bottom": 20,          # W底 65%（圖 5）
    "triple_bottom": 28,          # 三重底 排行 #5（W底加強版）
    "bull_flag": 24,              # 上升旗形 80%
    "rising_wedge": 30,           # 上升楔形（圖 5 將其列在買入 100%）
    "falling_wedge": 18,          # 下降楔形
    "inverted_v_bottom": 18,      # V型反轉（谷）
    # 觀望
    "symmetric_triangle": 0,
    "rectangle": 0,
}


@dataclass
class SignalMarker:
    """要畫到圖上的單點標記。
    `pane` 決定畫在主 K 線圖（main）或哪個擺盪指標圖（kd/rsi/macd）。"""
    date: str
    text: str
    position: Literal["aboveBar", "belowBar", "inBar"] = "aboveBar"
    color: str = "#4cc2ff"
    shape: Literal["arrowUp", "arrowDown", "circle", "square"] = "circle"
    pane: Literal["main", "kd", "rsi", "macd"] = "main"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SignalTrendline:
    """連接多個 (date, price) 點的折線（畫在主 K 線圖上）。"""
    points: list[tuple[str, float]]
    color: str = "#ffce5c"
    label: str = ""
    style: Literal["solid", "dashed"] = "solid"

    def to_dict(self) -> dict:
        return {
            "points": [{"date": d, "price": p} for d, p in self.points],
            "color": self.color,
            "label": self.label,
            "style": self.style,
        }


@dataclass
class SignalItem:
    code: str
    label: str
    impact: int
    detail: str
    markers: list[SignalMarker] = field(default_factory=list)
    trendlines: list[SignalTrendline] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)  # 計算邏輯逐步說明

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "label": self.label,
            "impact": self.impact,
            "detail": self.detail,
            "markers": [m.to_dict() for m in self.markers],
            "trendlines": [t.to_dict() for t in self.trendlines],
            "reasoning": list(self.reasoning),
        }


@dataclass
class Verdict:
    score: int
    label: str
    action: Literal["buy", "sell", "hold"]
    confidence: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AIEvaluation:
    headline: str       # 一句話結論
    action: Literal["buy", "sell", "hold"]
    confidence: int     # 0~100
    bullets_pro: list[str]  # 支持目前結論的觀察
    bullets_con: list[str]  # 風險/反方論點
    risk_warnings: list[str]
    suggested_action: str  # 具體操作建議

    def to_dict(self) -> dict:
        return asdict(self)


def _classify(score: int) -> tuple[str, str]:
    if score >= 80:
        return ("迎接暴漲", "buy")
    if score >= 50:
        return ("趕緊買入", "buy")
    if score >= 20:
        return ("偏多 — 可分批買入", "buy")
    if score <= -80:
        return ("防範暴跌", "sell")
    if score <= -50:
        return ("趕快賣出", "sell")
    if score <= -20:
        return ("緩慢下跌 — 建議減碼", "sell")
    return ("危險別碰 — 觀望", "hold")


def _impact_for_pattern(p: PatternMatch) -> int:
    """impact = sign(verdict) × max_impact × base_strength × confidence。"""
    base = PATTERN_MAX_IMPACT.get(p.name, 18)
    weighted = base * p.base_strength * p.confidence
    if p.verdict == "buy":
        return int(round(weighted))
    if p.verdict == "sell":
        return -int(round(weighted))
    return 0  # wait


def _markers_for_pattern(p: PatternMatch) -> list[SignalMarker]:
    color = "#5acc7d" if p.verdict == "buy" else "#f06b6b" if p.verdict == "sell" else "#aaa"
    markers: list[SignalMarker] = []
    for i, d in enumerate(p.pivot_dates):
        markers.append(SignalMarker(
            date=d, text=str(i + 1), color=color, shape="circle",
            position="aboveBar" if i % 2 == 0 else "belowBar",
        ))
    return markers


def _trendlines_for_pattern(p: PatternMatch) -> list[SignalTrendline]:
    if len(p.pivot_dates) < 2:
        return []
    color = "#5acc7d" if p.verdict == "buy" else "#f06b6b" if p.verdict == "sell" else "#aaa"
    points = list(zip(p.pivot_dates, p.pivot_prices))

    if p.name in ("double_top", "double_bottom"):
        # 連接「兩個相近高/低點」呈水平線；中間樞紐另畫一條虛線到那點
        a, mid, b = points
        return [
            SignalTrendline(points=[a, b], color=color, style="dashed", label=f"{p.label_zh}（兩端相近）"),
            SignalTrendline(points=[a, mid, b], color=color, style="solid"),
        ]
    if p.name in ("triple_top", "triple_bottom"):
        return [
            SignalTrendline(points=points, color=color, style="dashed", label=p.label_zh),
        ]
    if p.name in ("head_shoulders_top", "head_shoulders_bottom"):
        lines = [SignalTrendline(points=points, color=color, style="solid", label=p.label_zh)]
        if len(points) >= 5:
            lines.append(SignalTrendline(
                points=[points[1], points[3]], color="#ffce5c", style="dashed", label="頸線"
            ))
        return lines
    # 楔形 / 旗形 / 三角 / 倒V / 菱形：把樞紐折線畫出來即可看出型態形狀
    return [SignalTrendline(points=points, color=color, style="solid", label=p.label_zh)]


def _reasoning_for_pattern(p: PatternMatch, impact: int) -> list[str]:
    base = PATTERN_MAX_IMPACT.get(p.name, 18)
    pivots_str = "、".join(
        f"{d}({pr})" for d, pr in zip(p.pivot_dates, p.pivot_prices)
    )
    return [
        f"型態：{p.label_zh}（{p.note}）",
        f"涉及樞紐點：{pivots_str}",
        f"投影片定義的市場警示等級（base_strength）：{int(p.base_strength * 100)}%",
        f"辨識信心（含時序衰減，半衰期 20 天）：{int(p.confidence * 100)}%",
        f"得分計算：滿分 {base} × {p.base_strength:.2f} × {p.confidence:.2f} = {impact:+d}",
    ]


def aggregate(
    crosses: list[CrossSignal],
    last_candle: CandleTag | None,
    last_date: str | None,
    ma_state: dict,
    patterns: list[PatternMatch],
    horizontal_lines: list[HorizontalLine],
    current_price: float,
    kd_signals: list[KDSignal] | None = None,
    rsi_signal: RSISignal | None = None,
    macd_signals: list[MACDSignal] | None = None,
    recent_window_days: int = 10,
) -> tuple[Verdict, list[SignalItem]]:
    items: list[SignalItem] = []
    score = 0

    cutoff = date.today() - timedelta(days=recent_window_days)

    # ----- 均線交叉 -----
    for c in crosses:
        try:
            d = date.fromisoformat(c.date)
        except ValueError:
            continue
        if d < cutoff:
            continue

        if c.verdict in ("buy", "sell"):
            sign = 1 if c.verdict == "buy" else -1
            base_impact = 25 * sign
            label = "黃金交叉" if c.verdict == "buy" else "死亡交叉"
            code_prefix = "golden_cross" if c.verdict == "buy" else "death_cross"
            color = "#5acc7d" if c.verdict == "buy" else "#f06b6b"
            shape = "arrowUp" if c.verdict == "buy" else "arrowDown"
            reasoning = [
                f"SMA{c.short_window} 從下方上穿 SMA{c.long_window}（{c.date}）" if c.verdict == "buy"
                else f"SMA{c.short_window} 從上方下穿 SMA{c.long_window}（{c.date}）",
                f"長期均線 SMA{c.long_window} 方向：{c.long_trend}",
                f"得分：{base_impact:+d}（真{label}）",
            ]
        elif c.verdict == "hold_no_buy":
            base_impact = -10
            label = "假黃金交叉"
            code_prefix = "fake_golden"
            color = "#aaa"
            shape = "circle"
            reasoning = [
                f"SMA{c.short_window} 上穿 SMA{c.long_window}（{c.date}）",
                f"但 SMA{c.long_window} 仍在下行 → 投影片定義為「不要買」",
                "得分：-10（懲罰假突破）",
            ]
        elif c.verdict == "hold_no_sell":
            base_impact = +10
            label = "假死亡交叉"
            code_prefix = "fake_death"
            color = "#aaa"
            shape = "circle"
            reasoning = [
                f"SMA{c.short_window} 下穿 SMA{c.long_window}（{c.date}）",
                f"但 SMA{c.long_window} 仍在上行 → 投影片定義為「不要賣」",
                "得分：+10（多頭回檔不應放空）",
            ]
        else:
            continue

        score += base_impact
        items.append(SignalItem(
            code=f"{code_prefix}_{c.short_window}_{c.long_window}",
            label=f"{label} SMA{c.short_window}×SMA{c.long_window}",
            impact=base_impact,
            detail=f"{c.date} 短期穿越長期，長期均線方向：{c.long_trend}",
            markers=[SignalMarker(date=c.date, text=label[0:2], color=color, shape=shape,
                                  position="belowBar" if c.verdict in ("buy", "hold_no_sell") else "aboveBar")],
            reasoning=reasoning,
        ))

    # ----- 均線排列 -----
    state = ma_state.get("state")
    if state == "bullish_alignment" and last_date:
        score += 15
        items.append(SignalItem(
            "ma_bullish", "均線多頭排列 (5>20>60)", +15, "短中長期均線正向發散",
            markers=[SignalMarker(date=last_date, text="多排", color="#5acc7d", shape="circle", position="belowBar")],
            reasoning=[
                f"SMA5={ma_state.get('sma5', 0):.2f} > SMA20={ma_state.get('sma20', 0):.2f} > SMA60={ma_state.get('sma60', 0):.2f}",
                "三條均線同步向上發散 → 多頭趨勢確立",
                "得分：+15",
            ],
        ))
    elif state == "bearish_alignment" and last_date:
        score -= 15
        items.append(SignalItem(
            "ma_bearish", "均線空頭排列 (5<20<60)", -15, "短中長期均線負向發散",
            markers=[SignalMarker(date=last_date, text="空排", color="#f06b6b", shape="circle", position="aboveBar")],
            reasoning=[
                f"SMA5={ma_state.get('sma5', 0):.2f} < SMA20={ma_state.get('sma20', 0):.2f} < SMA60={ma_state.get('sma60', 0):.2f}",
                "三條均線同步向下發散 → 空頭趨勢確立",
                "得分：-15",
            ],
        ))

    # ----- 圖形型態 -----
    for p in patterns:
        impact = _impact_for_pattern(p)
        if impact == 0 and p.verdict != "wait":
            continue
        score += impact
        items.append(SignalItem(
            code=p.name,
            label=p.label_zh,
            impact=impact,
            detail=f"{p.note}（辨識信心 {int(p.confidence * 100)}%）" if p.note else "圖形型態確認",
            markers=_markers_for_pattern(p),
            trendlines=_trendlines_for_pattern(p),
            reasoning=_reasoning_for_pattern(p, impact),
        ))

    # ----- 最近一根 K -----
    if last_candle is not None:
        if last_candle.body_size == "large" and last_candle.direction == "bullish":
            score += 10
            items.append(SignalItem(
                "large_bullish_candle", "大陽線", +10, last_candle.meaning,
                markers=[SignalMarker(date=last_candle.date, text="大陽", color="#ef4f4f",
                                      shape="arrowUp", position="belowBar")],
                reasoning=["實體 K 棒 > 近 20 根平均實體的 1.5 倍且收紅", last_candle.meaning, "得分：+10"],
            ))
        elif last_candle.body_size == "large" and last_candle.direction == "bearish":
            score -= 10
            items.append(SignalItem(
                "large_bearish_candle", "大陰線", -10, last_candle.meaning,
                markers=[SignalMarker(date=last_candle.date, text="大陰", color="#3fb950",
                                      shape="arrowDown", position="aboveBar")],
                reasoning=["實體 K 棒 > 近 20 根平均實體的 1.5 倍且收黑", last_candle.meaning, "得分：-10"],
            ))
        elif last_candle.body_size == "doji":
            items.append(SignalItem(
                "doji", "十字線", 0, last_candle.meaning,
                markers=[SignalMarker(date=last_candle.date, text="十", color="#ffce5c",
                                      shape="circle", position="aboveBar")],
                reasoning=["實體 / 全長 < 10% → 十字線", "多空均衡，市場轉折點", "得分：0（中性）"],
            ))

    # ----- 支撐 / 壓力突破跌破 -----
    for line in horizontal_lines[:2]:
        gap = (current_price - line.price) / line.price
        if line.role == "resistance" and 0 < gap < 0.005:
            score += 15
            items.append(SignalItem(
                "break_resistance", f"突破壓力 {line.price}", +15,
                f"觸碰 {line.touches} 次的壓力被站上",
                markers=[SignalMarker(date=last_date or "", text=f"破{line.price}",
                                      color="#5acc7d", shape="arrowUp", position="belowBar")] if last_date else [],
                reasoning=[
                    f"壓力 {line.price} 觸碰 {line.touches} 次",
                    f"現價 {current_price} 已站上壓力（差距 {gap * 100:+.2f}%）",
                    "得分：+15（突破確認）",
                ],
            ))
        if line.role == "support" and -0.005 < gap < 0:
            score -= 15
            items.append(SignalItem(
                "break_support", f"跌破支撐 {line.price}", -15,
                f"觸碰 {line.touches} 次的支撐被跌破",
                markers=[SignalMarker(date=last_date or "", text=f"破{line.price}",
                                      color="#f06b6b", shape="arrowDown", position="aboveBar")] if last_date else [],
                reasoning=[
                    f"支撐 {line.price} 觸碰 {line.touches} 次",
                    f"現價 {current_price} 已跌破支撐（差距 {gap * 100:+.2f}%）",
                    "得分：-15（跌破確認）",
                ],
            ))

    # ----- KD 訊號 -----
    if kd_signals:
        for k in kd_signals:
            try:
                d = date.fromisoformat(k.date)
            except ValueError:
                continue
            if d < cutoff and k.kind not in ("overbought", "oversold"):
                continue
            if k.kind == "golden_oversold":
                imp, color, shape, pos = 18, "#5acc7d", "arrowUp", "belowBar"
            elif k.kind == "golden":
                imp, color, shape, pos = 10, "#5acc7d", "arrowUp", "belowBar"
            elif k.kind == "death_overbought":
                imp, color, shape, pos = -18, "#f06b6b", "arrowDown", "aboveBar"
            elif k.kind == "death":
                imp, color, shape, pos = -10, "#f06b6b", "arrowDown", "aboveBar"
            elif k.kind == "oversold":
                imp, color, shape, pos = 5, "#5acc7d", "circle", "belowBar"
            else:  # overbought
                imp, color, shape, pos = -5, "#f06b6b", "circle", "aboveBar"
            score += imp
            items.append(SignalItem(
                code=f"kd_{k.kind}_{k.date}",
                label=k.label_zh,
                impact=imp,
                detail=k.detail,
                markers=[
                    SignalMarker(date=k.date, text="KD↑" if imp > 0 else "KD↓",
                                 color=color, shape=shape, position=pos),
                    SignalMarker(date=k.date, text="K/D", color=color, shape="circle",
                                 position="aboveBar", pane="kd"),
                ],
                reasoning=[
                    f"日期 {k.date}：K={k.k:.1f}, D={k.d:.1f}",
                    f"KD 訊號類型：{k.kind}",
                    f"得分：{imp:+d}",
                ],
            ))

    # ----- RSI 訊號（僅最後一根，做為輔助參考，不入分） -----
    if rsi_signal is not None:
        items.append(SignalItem(
            code=f"rsi_{rsi_signal.kind}",
            label=rsi_signal.label_zh,
            impact=0,
            detail=f"RSI={rsi_signal.rsi:.1f}，僅供參考",
            markers=[SignalMarker(date=rsi_signal.date, text=f"{rsi_signal.rsi:.0f}",
                                  color="#ffce5c", shape="circle", position="aboveBar", pane="rsi")],
            reasoning=[
                f"RSI(14) = {rsi_signal.rsi:.2f}",
                "RSI > 70 為超買區、< 30 為超賣區",
                "本訊號僅做風險提示，不計入綜合評分",
            ],
        ))

    # ----- MACD 訊號 -----
    if macd_signals:
        for m in macd_signals:
            try:
                d = date.fromisoformat(m.date)
            except ValueError:
                continue
            if d < cutoff:
                continue
            imp = 8 if m.verdict == "buy" else -8
            score += imp
            color = "#5acc7d" if imp > 0 else "#f06b6b"
            shape = "arrowUp" if imp > 0 else "arrowDown"
            items.append(SignalItem(
                code=f"macd_{m.kind}_{m.date}",
                label=m.label_zh,
                impact=imp,
                detail=f"{m.date} MACD={m.macd:.3f}, Signal={m.signal:.3f}",
                markers=[
                    SignalMarker(date=m.date, text="MACD",
                                 color=color, shape=shape, position="belowBar" if imp > 0 else "aboveBar"),
                    SignalMarker(date=m.date, text="柱", color=color, shape="circle",
                                 position="aboveBar", pane="macd"),
                ],
                reasoning=[
                    f"日期 {m.date}：MACD={m.macd:.3f}, Signal={m.signal:.3f}",
                    f"柱狀體{'翻紅 (空轉多)' if imp > 0 else '翻綠 (多轉空)'}",
                    f"得分：{imp:+d}",
                ],
            ))

    score = max(-100, min(100, score))
    label, action = _classify(score)
    verdict = Verdict(score=score, label=label, action=action, confidence=abs(score))
    return verdict, items


# ---------- AI 自然語言評估 ----------

def synthesize(verdict: Verdict, items: list[SignalItem], patterns: list[PatternMatch],
               ma_state: dict, horizontal_lines: list[HorizontalLine], current_price: float) -> AIEvaluation:
    """把所有訊號整合成自然語言推薦，邏輯對應 6 張投影片：
    - 圖 6 排行榜：列出最強的買入/賣出型態
    - 圖 5 警告等級：用百分比換算的信心度敘述
    - 圖 3 均線：黃金/死亡交叉與排列
    - 圖 4 水平線：靠近的支撐/壓力
    - 圖 1 K 線：最近一根的氛圍
    """
    pros: list[str] = []
    cons: list[str] = []
    risks: list[str] = []

    # 拆分正負訊號
    pos_items = sorted([i for i in items if i.impact > 0], key=lambda x: -x.impact)
    neg_items = sorted([i for i in items if i.impact < 0], key=lambda x: x.impact)

    # 排行榜：取最強 3 個
    for i in pos_items[:3]:
        pros.append(f"{i.label}（+{i.impact}分）：{i.detail}")
    for i in neg_items[:3]:
        cons.append(f"{i.label}（{i.impact}分）：{i.detail}")

    # 型態風險警示：賣出排行榜出現任一即標紅
    sell_patterns = [p for p in patterns if p.verdict == "sell" and p.confidence >= 0.4]
    buy_patterns = [p for p in patterns if p.verdict == "buy" and p.confidence >= 0.4]
    if sell_patterns:
        risks.append(
            "出現賣出型態："
            + "、".join(f"{p.label_zh}({int(p.confidence * 100)}%)" for p in sell_patterns[:3])
        )
    if verdict.action == "buy" and ma_state.get("state") == "bearish_alignment":
        risks.append("均線仍為空頭排列，反彈可能為弱勢回測，停損紀律要嚴")
    if verdict.action == "sell" and ma_state.get("state") == "bullish_alignment":
        risks.append("均線仍為多頭排列，回檔風險屬正常修正，勿過度放空")

    # 靠近水平線提醒
    for line in horizontal_lines[:2]:
        gap = (current_price - line.price) / line.price
        if abs(gap) < 0.02:
            if line.role == "resistance":
                risks.append(f"距離壓力線 {line.price}（{line.touches} 次觸碰）僅 {gap * 100:+.2f}%，注意是否站上")
            else:
                risks.append(f"距離支撐線 {line.price}（{line.touches} 次觸碰）僅 {gap * 100:+.2f}%，注意是否守住")

    # 一句話結論 + 操作建議
    score = verdict.score
    headline_map = {
        "迎接暴漲": "技術面強烈轉多，可積極布局",
        "趕緊買入": "多項買入訊號共振，建議分批進場",
        "偏多 — 可分批買入": "偏多但訊號未完全強化，建議試單分批",
        "危險別碰 — 觀望": "多空訊號相抵，盤整觀望為佳",
        "緩慢下跌 — 建議減碼": "偏空訊號累積中，建議減碼或避險",
        "趕快賣出": "賣出訊號共振，建議獲利了結",
        "防範暴跌": "技術面強烈轉空，避免接刀子",
    }
    headline = headline_map.get(verdict.label, verdict.label)

    if verdict.action == "buy":
        if score >= 50:
            suggested = "可一次性買入或分 2~3 批建立 8 成倉位，停損設於最近支撐下方"
        else:
            suggested = "建議先試單 3~4 成倉位，等待訊號加強或突破壓力再加碼"
    elif verdict.action == "sell":
        if score <= -50:
            suggested = "建議減碼至 2 成以下或全數出場，反彈即賣，停損設於最近壓力上方"
        else:
            suggested = "建議減碼 3~5 成，剩餘部位嚴設停損，等待止跌訊號"
    else:
        suggested = "暫不進場，等待方向明確（型態突破或均線排列確立）後再行動"

    return AIEvaluation(
        headline=headline,
        action=verdict.action,
        confidence=verdict.confidence,
        bullets_pro=pros,
        bullets_con=cons,
        risk_warnings=risks,
        suggested_action=suggested,
    )
