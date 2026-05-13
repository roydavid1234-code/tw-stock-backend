"""批次掃描：對任意一組台股代碼跑完整分析管線，回傳精簡結果。

效能策略：
- ThreadPoolExecutor 平行抓取（FinMind I/O bound）
- 同一組代碼結果快取 30 分鐘（避免重覆掃描）
- 抓取失敗的單一檔不會中斷整體掃描
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Iterable

from ..data.finmind import FinMindError, fetch_daily_kline
from ..indicators.candle import classify_candles, latest_tag
from ..indicators.ma import add_moving_averages, detect_crosses, latest_alignment
from ..indicators.oscillators import (
    add_kd, add_macd, add_rsi,
    detect_kd_crosses, detect_macd_signals, detect_rsi_signal,
)
from ..indicators.patterns import detect_patterns
from ..indicators.pivots import find_horizontal_lines, zigzag
from ..signals.aggregator import aggregate
from .analysts import ScanContext, picks_for_all_analysts
from .universe import all_codes, category_of, name_of


@dataclass
class ScanItem:
    stock_id: str
    stock_name: str
    category: str
    current_price: float
    as_of: str
    score: int
    label: str
    action: str          # "buy" / "sell" / "hold"
    confidence: int      # |score|
    top_signals: list[str]
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


_CACHE: dict[str, tuple[datetime, list[ScanItem]]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = timedelta(minutes=30)


def _analyze_full(stock_id: str) -> tuple[ScanItem, ScanContext | None]:
    """完整跑一次分析；回傳精簡 ScanItem 與完整 ScanContext。"""
    cat = category_of(stock_id)
    name = name_of(stock_id)
    try:
        df = fetch_daily_kline(stock_id, lookback_days=300)
        if len(df) < 30:
            return ScanItem(stock_id, name, cat, 0.0, "", 0, "資料不足", "hold", 0, [], "資料筆數不足"), None
        df_ma = add_moving_averages(df)
        crosses = detect_crosses(df_ma)
        ma_state = latest_alignment(df_ma)
        df_osc = add_macd(add_rsi(add_kd(df)))
        kd_signals = detect_kd_crosses(df_osc)
        rsi_signal = detect_rsi_signal(df_osc)
        macd_signals = detect_macd_signals(df_osc)
        last_candle = latest_tag(classify_candles(df))
        pivots = zigzag(df, pct_threshold=0.03)
        current_price = float(df["close"].iloc[-1])
        h_lines = find_horizontal_lines(pivots, current_price=current_price)
        last_date = str(df["date"].iloc[-1].date())
        verdict, items = aggregate(
            crosses=crosses, last_candle=last_candle, last_date=last_date,
            ma_state=ma_state, patterns=detect_patterns(pivots),
            horizontal_lines=h_lines, current_price=current_price,
            kd_signals=kd_signals, rsi_signal=rsi_signal, macd_signals=macd_signals,
        )
        top = sorted(items, key=lambda s: -abs(s.impact))[:3]
        item = ScanItem(
            stock_id=stock_id, stock_name=name, category=cat,
            current_price=round(current_price, 2),
            as_of=last_date,
            score=verdict.score, label=verdict.label,
            action=verdict.action, confidence=verdict.confidence,
            top_signals=[f"{'+' if s.impact > 0 else ''}{s.impact} {s.label}" for s in top],
        )
        import math
        rsi_val = df_osc["rsi"].iloc[-1] if "rsi" in df_osc else None
        last_rsi = float(rsi_val) if rsi_val is not None and not math.isnan(rsi_val) else None
        ctx = ScanContext(
            stock_id=stock_id, stock_name=name, category=cat,
            current_price=current_price, base_score=verdict.score,
            base_label=verdict.label, base_action=verdict.action,
            signals=items, ma_state=ma_state,
            rsi_signal=rsi_signal, kd_signals=kd_signals,
            last_rsi=last_rsi,
        )
        return item, ctx
    except FinMindError as e:
        return ScanItem(stock_id, name, cat, 0.0, "", 0, "資料抓取失敗", "hold", 0, [], str(e)), None
    except Exception as e:
        return ScanItem(stock_id, name, cat, 0.0, "", 0, "分析失敗", "hold", 0, [], str(e)), None


def _analyze_single(stock_id: str) -> ScanItem:
    return _analyze_full(stock_id)[0]


def scan_batch(codes: Iterable[str], workers: int = 8) -> list[ScanItem]:
    code_list = list(dict.fromkeys(codes))  # 去重保序
    results: list[ScanItem] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_analyze_single, c): c for c in code_list}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                code = futures[fut]
                results.append(ScanItem(code, name_of(code), category_of(code), 0.0, "", 0, "錯誤", "hold", 0, [], str(e)))
    # 維持輸入順序輸出
    order_map = {c: i for i, c in enumerate(code_list)}
    results.sort(key=lambda r: order_map.get(r.stock_id, 999))
    return results


def scan_top(force: bool = False) -> dict:
    """掃描預設清單並回傳 Top 5 買入 / Top 5 賣出 + 全部結果。30 分鐘快取。"""
    key = "top"
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and not force:
            ts, items = cached
            if datetime.now() - ts < _CACHE_TTL:
                return _format_top(ts, items, from_cache=True)

    items = scan_batch(all_codes())
    with _CACHE_LOCK:
        _CACHE[key] = (datetime.now(), items)
    return _format_top(datetime.now(), items, from_cache=False)


def scan_analysts(force: bool = False) -> dict:
    """跑掃描 + 產出 5 位分析師的 Top 5 推薦。同 30 分鐘快取機制（獨立 cache key）。"""
    key = "analysts"
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and not force:
            ts, payload = cached  # type: ignore[misc]
            if datetime.now() - ts < _CACHE_TTL:
                return {**payload, "from_cache": True}

    codes = all_codes()
    contexts: list[ScanContext] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_analyze_full, c): c for c in codes}
        for fut in as_completed(futures):
            _, ctx = fut.result()
            if ctx is not None:
                contexts.append(ctx)

    picks = picks_for_all_analysts(contexts, top_n=5)
    payload = {
        "scanned_at": datetime.now().isoformat(timespec="seconds"),
        "from_cache": False,
        "universe_size": len(codes),
        "succeeded": len(contexts),
        "analysts": [p.to_dict() for p in picks],
    }
    with _CACHE_LOCK:
        _CACHE[key] = (datetime.now(), payload)  # type: ignore[assignment]
    return payload


def _format_top(scanned_at: datetime, items: list[ScanItem], from_cache: bool) -> dict:
    valid = [it for it in items if it.error is None]
    top_buy = sorted(
        [it for it in valid if it.action == "buy"],
        key=lambda x: -x.score,
    )[:5]
    top_sell = sorted(
        [it for it in valid if it.action == "sell"],
        key=lambda x: x.score,
    )[:5]
    return {
        "scanned_at": scanned_at.isoformat(timespec="seconds"),
        "from_cache": from_cache,
        "universe_size": len(items),
        "succeeded": len(valid),
        "top_buy": [it.to_dict() for it in top_buy],
        "top_sell": [it.to_dict() for it in top_sell],
        "all": [it.to_dict() for it in sorted(valid, key=lambda x: -x.score)],
    }
