"""5 位台灣知名分析師的「投資哲學」模型。

每位分析師會基於原始綜合分數 + signals + ma/rsi/kd 重新加權，得出他/她「最可能推薦」的 Top 5。
這只是基於公開言論風格的模擬，**不代表他們本人實際持倉或推薦**。

| 代號 | 人物 | 流派 | 核心邏輯 |
| --- | --- | --- | --- |
| laowang | 老王 (王倚隆) | 技術 / 趨勢派 | 均線交叉、量價、飆股、轉折點 |
| zhu | 朱家泓 | 經典 K 線派 | 波段、形態學、結構化判讀 |
| you | 游庭皓 | 總體經濟 / 週期派 | 景氣循環、大盤方向、大型股 / ETF |
| guo | 郭恭克 (豹大) | 基本面 / 財務派 | 財報、現金流、藍籌、避開投機股 |
| gucan | 股癌 (謝孟恭) | 市場邏輯 / 反直覺 | 反向操作、動態修正、超賣 / 超買警示 |
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..indicators.oscillators import KDSignal, RSISignal
from ..signals.aggregator import SignalItem


# ---------- 資料結構 ----------

@dataclass
class ScanContext:
    stock_id: str
    stock_name: str
    category: str
    current_price: float
    base_score: int
    base_label: str
    base_action: str
    signals: list[SignalItem]
    ma_state: dict
    rsi_signal: RSISignal | None
    kd_signals: list[KDSignal]
    last_rsi: float | None  # RSI 數值（即使未觸發超買超賣）


@dataclass
class AnalystPick:
    stock_id: str
    stock_name: str
    category: str
    analyst_score: float
    base_score: int
    base_label: str
    rationale: str
    verdict: str  # "buy" / "avoid"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalystPicks:
    analyst_id: str
    analyst_name: str
    school: str       # 流派
    philosophy: str   # 核心邏輯一句話
    top_buys: list[AnalystPick]
    top_avoids: list[AnalystPick]

    def to_dict(self) -> dict:
        return {
            "analyst_id": self.analyst_id,
            "analyst_name": self.analyst_name,
            "school": self.school,
            "philosophy": self.philosophy,
            "top_buys": [p.to_dict() for p in self.top_buys],
            "top_avoids": [p.to_dict() for p in self.top_avoids],
        }


# ---------- 大型權值股 / ETF 名單（給游庭皓、郭恭克用） ----------

BLUE_CHIPS = {"2330", "2317", "2454", "2412", "2308", "1216", "3008", "2882", "2881"}
ETFS = {"0050", "0056", "00878", "006208", "00919"}


# ---------- 單檔評分 ----------

def _laowang(ctx: ScanContext) -> tuple[float, str]:
    """老王：趨勢派 — 重均線交叉與大 K 線、突破。"""
    score = 0.0
    why = []
    for s in ctx.signals:
        if "golden_cross" in s.code:
            score += s.impact * 1.6
            why.append("黃金交叉成立")
        elif "death_cross" in s.code:
            score += s.impact * 1.6
            why.append("死亡交叉成立")
        elif s.code == "ma_bullish":
            score += s.impact * 2.0
            why.append("均線多頭排列")
        elif s.code == "ma_bearish":
            score += s.impact * 2.0
            why.append("均線空頭排列")
        elif s.code in ("bull_flag", "rising_wedge"):
            score += s.impact * 1.6
            why.append(f"持續型態{s.label}")
        elif s.code == "bear_flag":
            score += s.impact * 1.6
        elif "large_bullish_candle" in s.code:
            score += s.impact * 1.4
            why.append("大陽線轉強")
        elif "large_bearish_candle" in s.code:
            score += s.impact * 1.4
        elif "break_resistance" in s.code:
            score += s.impact * 1.5
            why.append("突破壓力")
        elif "break_support" in s.code:
            score += s.impact * 1.5
        else:
            score += s.impact * 0.4  # 不重視其他訊號（型態、KD 等）

    if ctx.ma_state.get("state") == "tangled":
        score -= 8
        why.append("均線糾結（迴避）")

    rationale = "、".join(why[:2]) or "趨勢未明"
    return score, rationale


def _zhu(ctx: ScanContext) -> tuple[float, str]:
    """朱家泓：K 線形態派 — 重經典形態與支撐壓力。"""
    score = 0.0
    why = []
    pattern_codes = {"double_top", "double_bottom", "triple_top", "triple_bottom",
                     "head_shoulders_top", "head_shoulders_bottom",
                     "diamond_top", "rising_wedge", "falling_wedge",
                     "bull_flag", "bear_flag", "inverted_v_top", "inverted_v_bottom"}
    for s in ctx.signals:
        if s.code in pattern_codes:
            score += s.impact * 2.0
            why.append(f"{s.label}成立")
        elif "break_resistance" in s.code or "break_support" in s.code:
            score += s.impact * 1.5
            why.append(s.label)
        elif "golden_cross" in s.code or "death_cross" in s.code:
            score += s.impact * 0.6  # 純均線他較不重視
        elif s.code in ("ma_bullish", "ma_bearish"):
            score += s.impact * 0.7
        else:
            score += s.impact * 0.3

    rationale = "、".join(why[:2]) or "型態尚未明確"
    return score, rationale


def _you(ctx: ScanContext) -> tuple[float, str]:
    """游庭皓：總體 / 週期派 — 偏好 ETF / 大型權值股，個股技術訊號減弱。"""
    base = sum(s.impact for s in ctx.signals) * 0.5  # 個股技術全面減半
    bonus = 0.0
    why = []

    if ctx.stock_id in ETFS:
        bonus += 30
        why.append("ETF（追蹤大盤週期）")
    elif ctx.stock_id in BLUE_CHIPS:
        bonus += 20
        why.append("權值大型股")

    # 大盤類產業加成（電子代工、金融、半導體最敏感）
    if ctx.category in ("金融", "半導體", "電子代工"):
        bonus += 8
        why.append(f"景氣敏感類股（{ctx.category}）")

    # 偏好趨勢延續、避開反轉型態
    for s in ctx.signals:
        if s.code == "ma_bullish":
            bonus += 8
        elif s.code == "ma_bearish":
            bonus -= 8

    rationale = "、".join(why[:2]) or "非景氣指標標的"
    return base + bonus, rationale


def _guo(ctx: ScanContext) -> tuple[float, str]:
    """郭恭克：基本面派 — 偏好藍籌大型股，對短期暴漲與投機性訊號扣分。"""
    base = sum(s.impact for s in ctx.signals) * 0.7
    bonus = 0.0
    why = []

    if ctx.stock_id in BLUE_CHIPS:
        bonus += 25
        why.append("產業龍頭 / 基本面穩健")
    elif ctx.stock_id in ETFS:
        bonus += 15
        why.append("被動式長期配置")
    else:
        bonus -= 5  # 沒財報資料無法評，先扣一點

    # RSI 超買代表估值過熱（毒舌看不順眼）
    if ctx.last_rsi is not None:
        if ctx.last_rsi > 75:
            bonus -= 20
            why.append(f"RSI {ctx.last_rsi:.1f} 過熱")
        elif ctx.last_rsi < 35:
            bonus += 10
            why.append(f"RSI {ctx.last_rsi:.1f} 估值合理")

    # 對「投機型態」嚴格扣分：倒V、菱形頂、楔形高點
    for s in ctx.signals:
        if s.code in ("inverted_v_top", "diamond_top", "rising_wedge"):
            bonus -= 12
            why.append("出現投機性高點型態")
        elif s.code in ("ma_bullish",):
            bonus += 6

    rationale = "、".join(why[:2]) or "缺乏明顯優勢"
    return base + bonus, rationale


def _gucan(ctx: ScanContext) -> tuple[float, str]:
    """股癌：反直覺派 — 在大家恐慌時找買點、在過熱時警示賣出。"""
    base = sum(s.impact for s in ctx.signals) * 0.6
    bonus = 0.0
    why = []

    # 反向加碼：RSI 超賣 + KD 低檔
    if ctx.last_rsi is not None:
        if ctx.last_rsi < 30:
            bonus += 30
            why.append(f"RSI {ctx.last_rsi:.1f} 嚴重超賣，逆向買")
        elif ctx.last_rsi < 40:
            bonus += 10
            why.append(f"RSI {ctx.last_rsi:.1f} 偏弱，可佈局")
        elif ctx.last_rsi > 75:
            bonus -= 25
            why.append(f"RSI {ctx.last_rsi:.1f} 過熱，反向警示")

    for s in ctx.signals:
        if s.code.startswith("kd_golden_oversold"):
            bonus += 20
            why.append("KD 低檔黃金交叉")
        elif s.code.startswith("kd_death_overbought"):
            bonus -= 20
            why.append("KD 高檔死亡交叉，過熱")
        elif s.code in ("inverted_v_bottom", "double_bottom", "triple_bottom"):
            bonus += 10  # 恐慌底加成
            why.append(f"恐慌底部型態（{s.label}）")
        elif s.code in ("inverted_v_top", "rising_wedge"):
            bonus -= 15  # 慶祝中的頂部要警惕
            why.append("頂部訊號警示")

    # 接近強支撐（觸碰 ≥ 3 次）會額外加分；但 ScanItem 沒有 lines 資料，故略

    rationale = "、".join(why[:2]) or "暫無明顯反向訊號"
    return base + bonus, rationale


# ---------- 公開介面 ----------

ANALYSTS = [
    ("laowang", "AI老王 (王倚隆)", "技術派 / 趨勢", "均線交叉、量價、飆股與轉折點", _laowang),
    ("zhu", "AI朱家泓", "經典 K 線派", "波段操作、形態學、結構化判讀", _zhu),
    ("you", "AI游庭皓", "總體經濟 / 週期派", "景氣循環、大盤方向；偏好 ETF 與大型權值股", _you),
    ("guo", "AI郭恭克 (豹大)", "基本面 / 財務派", "藍籌穩健、現金流；嚴格避開過熱與投機", _guo),
    ("gucan", "AI股癌 (謝孟恭)", "市場邏輯 / 反直覺", "反向操作、動態修正、超賣加碼超買警示", _gucan),
]


def picks_for_all_analysts(contexts: list[ScanContext], top_n: int = 5) -> list[AnalystPicks]:
    out: list[AnalystPicks] = []
    for aid, name, school, philosophy, fn in ANALYSTS:
        scored: list[tuple[float, str, ScanContext]] = []
        for ctx in contexts:
            score, why = fn(ctx)
            scored.append((score, why, ctx))
        scored.sort(key=lambda x: -x[0])

        buys = [
            AnalystPick(
                stock_id=ctx.stock_id, stock_name=ctx.stock_name, category=ctx.category,
                analyst_score=round(score, 1), base_score=ctx.base_score,
                base_label=ctx.base_label, rationale=why, verdict="buy",
            )
            for score, why, ctx in scored if score > 5
        ][:top_n]

        avoids = [
            AnalystPick(
                stock_id=ctx.stock_id, stock_name=ctx.stock_name, category=ctx.category,
                analyst_score=round(score, 1), base_score=ctx.base_score,
                base_label=ctx.base_label, rationale=why, verdict="avoid",
            )
            for score, why, ctx in sorted(scored, key=lambda x: x[0]) if score < -5
        ][:top_n]

        out.append(AnalystPicks(
            analyst_id=aid, analyst_name=name, school=school, philosophy=philosophy,
            top_buys=buys, top_avoids=avoids,
        ))
    return out
