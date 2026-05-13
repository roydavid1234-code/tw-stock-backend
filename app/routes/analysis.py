"""分析路由：把所有 indicator 串起來，回傳給前端的 JSON。"""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from ..data.finmind import FinMindError, fetch_daily_kline, search_stock
from ..indicators.candle import classify_candles, latest_tag
from ..indicators.ma import (
    add_moving_averages,
    detect_crosses,
    latest_alignment,
)
from ..indicators.oscillators import (
    add_kd, add_macd, add_rsi,
    detect_kd_crosses, detect_macd_signals, detect_rsi_signal,
)
from ..indicators.patterns import detect_patterns
from ..indicators.pivots import find_horizontal_lines, zigzag
from ..signals.aggregator import aggregate, synthesize
from ..signals.position_plan import plan_position

router = APIRouter(prefix="/api", tags=["analysis"])


@router.get("/search")
def search(q: str = Query(..., min_length=1, max_length=20)):
    try:
        return {"results": search_stock(q)}
    except FinMindError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/stocks/{stock_id}/analysis")
def analyze(
    stock_id: str,
    days: int = Query(365, ge=60, le=1500),
    strategy: str = Query("auto", description="分批策略：auto/aggressive/balanced/conservative/equal"),
):
    try:
        df = fetch_daily_kline(stock_id, lookback_days=days)
    except FinMindError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if len(df) < 30:
        raise HTTPException(status_code=400, detail="資料筆數不足，無法分析")

    df_ma = add_moving_averages(df)
    crosses = detect_crosses(df_ma)
    ma_state = latest_alignment(df_ma)

    df_osc = add_kd(df)
    df_osc = add_rsi(df_osc)
    df_osc = add_macd(df_osc)
    kd_signals = detect_kd_crosses(df_osc)
    rsi_signal = detect_rsi_signal(df_osc)
    macd_signals = detect_macd_signals(df_osc)

    candles = classify_candles(df)
    last_candle = latest_tag(candles)

    pivots = zigzag(df, pct_threshold=0.03)
    current_price = float(df["close"].iloc[-1])
    last_date = str(df["date"].iloc[-1].date())
    h_lines = find_horizontal_lines(pivots, current_price=current_price)

    patterns = detect_patterns(pivots)

    verdict, items = aggregate(
        crosses=crosses,
        last_candle=last_candle,
        last_date=last_date,
        ma_state=ma_state,
        patterns=patterns,
        horizontal_lines=h_lines,
        current_price=current_price,
        kd_signals=kd_signals,
        rsi_signal=rsi_signal,
        macd_signals=macd_signals,
    )
    ai_eval = synthesize(verdict, items, patterns, ma_state, h_lines, current_price)
    position = plan_position(verdict, current_price, h_lines, strategy=strategy)

    kline = [
        {
            "date": str(row["date"].date()),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"] or 0),
        }
        for _, row in df.iterrows()
    ]

    ma_series = {}
    for w in [5, 10, 20, 50, 60, 100, 200]:
        col = f"sma{w}"
        if col in df_ma:
            ma_series[col] = [
                {"date": str(d.date()), "value": (None if pd.isna(v) else float(v))}
                for d, v in zip(df_ma["date"], df_ma[col])
            ]

    def _series(col: str) -> list[dict]:
        return [
            {"date": str(d.date()), "value": (None if pd.isna(v) else float(v))}
            for d, v in zip(df_osc["date"], df_osc[col])
        ]

    oscillators = {
        "kd": {"k": _series("kd_k"), "d": _series("kd_d"), "j": _series("kd_j")},
        "rsi": {"rsi": _series("rsi")},
        "macd": {
            "macd": _series("macd"),
            "signal": _series("macd_signal"),
            "hist": _series("macd_hist"),
        },
    }

    return {
        "stock_id": stock_id,
        "as_of": str(df["date"].iloc[-1].date()),
        "current_price": current_price,
        "verdict": verdict.to_dict(),
        "ai_evaluation": ai_eval.to_dict(),
        "position_plan": position.to_dict(),
        "signals": [s.to_dict() for s in items],
        "ma_state": ma_state,
        "crosses": [c.__dict__ for c in crosses],
        "last_candle": last_candle.to_dict() if last_candle else None,
        "patterns": [p.to_dict() for p in patterns],
        "horizontal_lines": [l.to_dict() for l in h_lines],
        "pivots": [p.to_dict() for p in pivots],
        "kline": kline,
        "ma": ma_series,
        "oscillators": oscillators,
        "kd_signals": [k.to_dict() for k in kd_signals],
        "rsi_signal": rsi_signal.to_dict() if rsi_signal else None,
        "macd_signals": [m.to_dict() for m in macd_signals],
    }
