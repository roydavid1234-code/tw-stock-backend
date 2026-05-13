"""圖形型態辨識（基於 zigzag 樞紐點的規則演算法）。

對應投影片『買入/賣出』型態 + 警告圖表 + 賣出/等待/買入 三欄圖。

實作清單：
反轉型態（reversal）
- 雙頂 M頭 / 雙底 W底
- 三重頂 / 三重底
- 頭肩頂 / 頭肩底
- 倒V型反轉（spike top / spike bottom）
- 菱形頂

持續型態（continuation）
- 上升旗形 / 下跌旗形
- 上升楔形（投影片歸類為買入 100%）/ 下降楔形

盤整型態（wait）
- 三角收斂（對稱三角形）
- 箱型盤整

時序衰減：每個型態都會根據「最後一個樞紐距今的交易日」打折，距今越久 confidence 越低。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from datetime import date
from typing import Literal

from .pivots import Pivot

PatternName = Literal[
    # 反轉
    "double_top",
    "double_bottom",
    "triple_top",
    "triple_bottom",
    "head_shoulders_top",
    "head_shoulders_bottom",
    "inverted_v_top",
    "inverted_v_bottom",
    "diamond_top",
    # 持續
    "bull_flag",
    "bear_flag",
    "rising_wedge",
    "falling_wedge",
    # 盤整 (wait)
    "symmetric_triangle",
    "rectangle",
]

PatternVerdict = Literal["buy", "sell", "wait"]


@dataclass
class PatternMatch:
    name: PatternName
    label_zh: str
    verdict: PatternVerdict
    confidence: float  # 0~1（已含時序衰減）
    base_strength: float  # 0~1（依投影片定義的權重）
    pivot_dates: list[str]
    pivot_prices: list[float]
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# 兩高/兩低相近的容差（百分比）
LEVEL_TOLERANCE = 0.03
# 三重頂/底用更鬆的容差，因為三點都要相近
TRIPLE_TOLERANCE = 0.04


def _close(a: float, b: float, tol: float = LEVEL_TOLERANCE) -> bool:
    return abs(a - b) / max(a, b) <= tol


def _slope(points: list[tuple[int, float]]) -> float:
    """簡單線性回歸斜率（每根 K 的價格變動）。"""
    n = len(points)
    if n < 2:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def _decay_by_recency(pattern_last_date: str, today: date | None = None, half_life_days: int = 20) -> float:
    """指數衰減：last pivot 距今超過 half_life 天，confidence 折半。"""
    today = today or date.today()
    try:
        d = date.fromisoformat(pattern_last_date)
    except ValueError:
        return 0.5
    days = max(0, (today - d).days)
    return float(math.pow(0.5, days / half_life_days))


def _detect_double(pivots: list[Pivot]) -> list[PatternMatch]:
    matches: list[PatternMatch] = []
    recent = list(enumerate(pivots[-12:]))
    peaks = [(i, p) for i, p in recent if p.kind == "peak"]
    troughs = [(i, p) for i, p in recent if p.kind == "trough"]

    for i in range(len(peaks) - 1):
        idx_a, a = peaks[i]
        idx_b, b = peaks[i + 1]
        if idx_b - idx_a < 2 or not _close(a.price, b.price):
            continue
        mid_trough = [t for j, t in troughs if idx_a < j < idx_b]
        if not mid_trough:
            continue
        decay = _decay_by_recency(b.date)
        matches.append(PatternMatch(
            name="double_top", label_zh="M頭（雙頂）",
            verdict="sell", confidence=0.8 * decay, base_strength=1.00,
            pivot_dates=[a.date, mid_trough[0].date, b.date],
            pivot_prices=[a.price, mid_trough[0].price, b.price],
            note="兩個相近高點 → 賣出警告 100%",
        ))

    for i in range(len(troughs) - 1):
        idx_a, a = troughs[i]
        idx_b, b = troughs[i + 1]
        if idx_b - idx_a < 2 or not _close(a.price, b.price):
            continue
        mid_peak = [t for j, t in peaks if idx_a < j < idx_b]
        if not mid_peak:
            continue
        decay = _decay_by_recency(b.date)
        matches.append(PatternMatch(
            name="double_bottom", label_zh="W底（雙底）",
            verdict="buy", confidence=0.8 * decay, base_strength=0.65,
            pivot_dates=[a.date, mid_peak[0].date, b.date],
            pivot_prices=[a.price, mid_peak[0].price, b.price],
            note="兩個相近低點 → 買入警告 65%",
        ))
    return matches


def _detect_triple(pivots: list[Pivot]) -> list[PatternMatch]:
    matches: list[PatternMatch] = []
    recent = list(enumerate(pivots[-14:]))
    peaks = [(i, p) for i, p in recent if p.kind == "peak"]
    troughs = [(i, p) for i, p in recent if p.kind == "trough"]

    for i in range(len(peaks) - 2):
        idx_a, a = peaks[i]
        idx_b, b = peaks[i + 1]
        idx_c, c = peaks[i + 2]
        if idx_c - idx_a < 4:
            continue
        if (
            _close(a.price, b.price, TRIPLE_TOLERANCE)
            and _close(b.price, c.price, TRIPLE_TOLERANCE)
            and _close(a.price, c.price, TRIPLE_TOLERANCE)
        ):
            decay = _decay_by_recency(c.date)
            matches.append(PatternMatch(
                name="triple_top", label_zh="三重頂",
                verdict="sell", confidence=0.9 * decay, base_strength=0.95,
                pivot_dates=[a.date, b.date, c.date],
                pivot_prices=[a.price, b.price, c.price],
                note="三個相近高點，賣出力道強於 M 頭",
            ))

    for i in range(len(troughs) - 2):
        idx_a, a = troughs[i]
        idx_b, b = troughs[i + 1]
        idx_c, c = troughs[i + 2]
        if idx_c - idx_a < 4:
            continue
        if (
            _close(a.price, b.price, TRIPLE_TOLERANCE)
            and _close(b.price, c.price, TRIPLE_TOLERANCE)
            and _close(a.price, c.price, TRIPLE_TOLERANCE)
        ):
            decay = _decay_by_recency(c.date)
            matches.append(PatternMatch(
                name="triple_bottom", label_zh="三重底",
                verdict="buy", confidence=0.9 * decay, base_strength=0.90,
                pivot_dates=[a.date, b.date, c.date],
                pivot_prices=[a.price, b.price, c.price],
                note="三個相近低點，買入力道強於 W 底",
            ))
    return matches


def _detect_head_shoulders(pivots: list[Pivot]) -> list[PatternMatch]:
    matches: list[PatternMatch] = []
    recent = pivots[-10:]
    if len(recent) < 5:
        return matches
    for i in range(len(recent) - 4):
        seq = recent[i : i + 5]
        kinds = [p.kind for p in seq]
        if kinds == ["peak", "trough", "peak", "trough", "peak"]:
            ls, _, head, _, rs = seq
            if head.price > ls.price and head.price > rs.price and _close(ls.price, rs.price):
                decay = _decay_by_recency(seq[-1].date)
                matches.append(PatternMatch(
                    name="head_shoulders_top", label_zh="頭肩頂",
                    verdict="sell", confidence=0.8 * decay, base_strength=0.90,
                    pivot_dates=[p.date for p in seq],
                    pivot_prices=[p.price for p in seq],
                    note="左肩-頭-右肩，頸線跌破即確認",
                ))
        if kinds == ["trough", "peak", "trough", "peak", "trough"]:
            ls, _, head, _, rs = seq
            if head.price < ls.price and head.price < rs.price and _close(ls.price, rs.price):
                decay = _decay_by_recency(seq[-1].date)
                matches.append(PatternMatch(
                    name="head_shoulders_bottom", label_zh="頭肩底（倒頭肩）",
                    verdict="buy", confidence=0.8 * decay, base_strength=1.00,
                    pivot_dates=[p.date for p in seq],
                    pivot_prices=[p.price for p in seq],
                    note="買入排行榜第 1 名",
                ))
    return matches


def _detect_inverted_v(pivots: list[Pivot]) -> list[PatternMatch]:
    """單一極端高/低點，前後變動幅度很大且時間很短 → 倒 V 反轉。"""
    matches: list[PatternMatch] = []
    if len(pivots) < 3:
        return matches
    for i in range(1, len(pivots) - 1):
        prev_p, p, next_p = pivots[i - 1], pivots[i], pivots[i + 1]
        up = (p.price - prev_p.price) / prev_p.price if prev_p.price else 0
        down = (p.price - next_p.price) / p.price if p.price else 0
        if p.kind == "peak" and up > 0.15 and down > 0.10:
            decay = _decay_by_recency(next_p.date)
            matches.append(PatternMatch(
                name="inverted_v_top", label_zh="倒V型反轉（峰）",
                verdict="sell", confidence=0.75 * decay, base_strength=0.70,
                pivot_dates=[prev_p.date, p.date, next_p.date],
                pivot_prices=[prev_p.price, p.price, next_p.price],
                note="急漲後急跌的尖峰",
            ))
        if p.kind == "trough" and up < -0.10 and down < -0.10:
            # for troughs: drop then quick recovery
            drop = (prev_p.price - p.price) / prev_p.price
            rebound = (next_p.price - p.price) / p.price
            if drop > 0.10 and rebound > 0.10:
                decay = _decay_by_recency(next_p.date)
                matches.append(PatternMatch(
                    name="inverted_v_bottom", label_zh="V型反轉（谷）",
                    verdict="buy", confidence=0.7 * decay, base_strength=0.65,
                    pivot_dates=[prev_p.date, p.date, next_p.date],
                    pivot_prices=[prev_p.price, p.price, next_p.price],
                    note="急跌後急彈的尖底",
                ))
    return matches


def _detect_diamond_top(pivots: list[Pivot]) -> list[PatternMatch]:
    """菱形頂：先擴散（高更高、低更低）再收斂（高更低、低更高）。

    需要至少 5 個交替樞紐。檢查方法：把最近 5~6 個樞紐分前後段，前段擴散、後段收斂。
    """
    matches: list[PatternMatch] = []
    if len(pivots) < 5:
        return matches
    seq = pivots[-6:]
    if len(seq) < 5:
        return matches
    peaks = [p for p in seq if p.kind == "peak"]
    troughs = [p for p in seq if p.kind == "trough"]
    if len(peaks) < 3 or len(troughs) < 2:
        return matches
    # 前段擴散：第二個 peak 高於第一個，第二個 trough 低於第一個
    if peaks[1].price <= peaks[0].price or troughs[1].price >= troughs[0].price:
        return matches
    # 後段收斂：第三個 peak 低於第二個（如果有第三個 peak）
    if len(peaks) >= 3 and peaks[2].price < peaks[1].price:
        # 且需要在相對高位 → 用最後 peak 與 sequence 整體相比
        max_peak = max(p.price for p in peaks)
        if peaks[1].price >= max_peak * 0.97:
            decay = _decay_by_recency(seq[-1].date)
            matches.append(PatternMatch(
                name="diamond_top", label_zh="菱形頂",
                verdict="sell", confidence=0.6 * decay, base_strength=0.65,
                pivot_dates=[p.date for p in seq],
                pivot_prices=[p.price for p in seq],
                note="先擴散後收斂的高位反轉，賣出警告 65%",
            ))
    return matches


def _trendlines(pivots: list[Pivot]) -> tuple[float, float, float, float]:
    """回傳 (peak_slope, peak_intercept_proxy, trough_slope, trough_intercept_proxy)。

    intercept 用最後一個 pivot 的價位做代表，方便判斷上下軌位置。
    """
    if not pivots:
        return 0, 0, 0, 0
    peaks = [(i, p.price) for i, p in enumerate(pivots) if p.kind == "peak"]
    troughs = [(i, p.price) for i, p in enumerate(pivots) if p.kind == "trough"]
    return (
        _slope(peaks),
        peaks[-1][1] if peaks else 0,
        _slope(troughs),
        troughs[-1][1] if troughs else 0,
    )


def _detect_flags_and_wedges(pivots: list[Pivot]) -> list[PatternMatch]:
    """旗形與楔形：分析最近 4~6 個樞紐的上下軌斜率組合。

    令 m_up = 高點連線斜率、m_dn = 低點連線斜率：
    - 上升旗形：先 pole（看 zigzag 前段大漲）+ 後段 m_up < 0 ~ slight negative, m_dn < 0, |m_up| ≈ |m_dn|（平行小幅下傾）
    - 下跌旗形：先 pole 大跌 + 後段 m_up > 0, m_dn > 0, |m_up| ≈ |m_dn|（平行小幅上傾）
    - 上升楔形：m_up > 0, m_dn > 0, m_dn > m_up（兩線同向上傾且收斂）→ 投影片歸類為買入警告 100%
    - 下降楔形：m_up < 0, m_dn < 0, m_up > m_dn（兩線同向下傾且收斂）→ 買入
    """
    matches: list[PatternMatch] = []
    if len(pivots) < 5:
        return matches
    pole = pivots[-7:-4] if len(pivots) >= 7 else pivots[:-4]
    channel = pivots[-4:]
    if len(channel) < 4 or len(pole) < 2:
        return matches

    pole_return = (pole[-1].price - pole[0].price) / pole[0].price if pole[0].price else 0
    pm_slope, _, tm_slope, _ = _trendlines(channel)

    # 旗形：pole 強勢 + 後段反向小斜率平行通道
    avg_price = sum(p.price for p in channel) / len(channel)
    same_sign = (pm_slope > 0) == (tm_slope > 0)
    parallel = abs(pm_slope - tm_slope) / max(abs(pm_slope), abs(tm_slope), 1e-6) < 0.4
    last_date = channel[-1].date

    if pole_return > 0.10 and same_sign and parallel and pm_slope < 0:
        decay = _decay_by_recency(last_date)
        matches.append(PatternMatch(
            name="bull_flag", label_zh="上升旗形",
            verdict="buy", confidence=0.75 * decay, base_strength=0.80,
            pivot_dates=[p.date for p in channel],
            pivot_prices=[p.price for p in channel],
            note="大漲後的小幅回檔平行通道 → 買入警告 80%",
        ))
    if pole_return < -0.10 and same_sign and parallel and pm_slope > 0:
        decay = _decay_by_recency(last_date)
        matches.append(PatternMatch(
            name="bear_flag", label_zh="下跌旗形",
            verdict="sell", confidence=0.75 * decay, base_strength=0.80,
            pivot_dates=[p.date for p in channel],
            pivot_prices=[p.price for p in channel],
            note="大跌後的小幅反彈平行通道 → 賣出警告 80%",
        ))

    # 楔形：兩線同向且收斂
    converging = (pm_slope > 0 and tm_slope > 0 and tm_slope > pm_slope) or \
                 (pm_slope < 0 and tm_slope < 0 and pm_slope > tm_slope)
    if converging:
        if pm_slope > 0 and tm_slope > 0:
            decay = _decay_by_recency(last_date)
            matches.append(PatternMatch(
                name="rising_wedge", label_zh="上升楔形",
                verdict="buy", confidence=0.7 * decay, base_strength=1.00,
                pivot_dates=[p.date for p in channel],
                pivot_prices=[p.price for p in channel],
                note="兩條向上收斂的趨勢線 → 投影片買入警告 100%",
            ))
        elif pm_slope < 0 and tm_slope < 0:
            decay = _decay_by_recency(last_date)
            matches.append(PatternMatch(
                name="falling_wedge", label_zh="下降楔形",
                verdict="buy", confidence=0.65 * decay, base_strength=0.70,
                pivot_dates=[p.date for p in channel],
                pivot_prices=[p.price for p in channel],
                note="兩條向下收斂的趨勢線，等待突破",
            ))
    return matches


def _detect_consolidation(pivots: list[Pivot]) -> list[PatternMatch]:
    """三角收斂與箱型盤整，皆屬『等待』訊號。"""
    matches: list[PatternMatch] = []
    if len(pivots) < 5:
        return matches
    seq = pivots[-6:]
    pm_slope, _, tm_slope, _ = _trendlines(seq)
    last_date = seq[-1].date

    # 對稱三角形：peak 連線下降、trough 連線上升
    if pm_slope < -1e-3 and tm_slope > 1e-3:
        decay = _decay_by_recency(last_date)
        matches.append(PatternMatch(
            name="symmetric_triangle", label_zh="三角收斂（對稱三角形）",
            verdict="wait", confidence=0.6 * decay, base_strength=0.50,
            pivot_dates=[p.date for p in seq],
            pivot_prices=[p.price for p in seq],
            note="上下軌收斂，等待突破方向",
        ))

    # 箱型盤整：上下軌幾乎水平
    flat = abs(pm_slope) < 5e-3 and abs(tm_slope) < 5e-3
    if flat:
        decay = _decay_by_recency(last_date)
        matches.append(PatternMatch(
            name="rectangle", label_zh="箱型盤整",
            verdict="wait", confidence=0.5 * decay, base_strength=0.50,
            pivot_dates=[p.date for p in seq],
            pivot_prices=[p.price for p in seq],
            note="上下軌平行，等待箱型突破",
        ))
    return matches


def detect_patterns(pivots: list[Pivot]) -> list[PatternMatch]:
    """跑全部偵測器，依信心度由高到低排序，且過濾掉信心度 < 0.15 的雜訊。"""
    if len(pivots) < 4:
        return []
    matches: list[PatternMatch] = []
    matches.extend(_detect_double(pivots))
    matches.extend(_detect_triple(pivots))
    matches.extend(_detect_head_shoulders(pivots))
    matches.extend(_detect_inverted_v(pivots))
    matches.extend(_detect_diamond_top(pivots))
    matches.extend(_detect_flags_and_wedges(pivots))
    matches.extend(_detect_consolidation(pivots))

    matches = [m for m in matches if m.confidence >= 0.15]
    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches
