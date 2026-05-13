"""FinMind 免費 API 客戶端。

只用 anonymous endpoint（每小時 300 次配額已足夠本機自用）。
回傳 pandas.DataFrame，欄位統一為小寫：date / open / high / low / close / volume。
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import httpx
import pandas as pd

FINMIND_BASE = "https://api.finmindtrade.com/api/v4/data"


class FinMindError(RuntimeError):
    pass


def _request(dataset: str, params: dict) -> list[dict]:
    full = {"dataset": dataset, **params}
    with httpx.Client(timeout=15.0) as client:
        r = client.get(FINMIND_BASE, params=full)
    if r.status_code != 200:
        raise FinMindError(f"FinMind HTTP {r.status_code}: {r.text[:200]}")
    body = r.json()
    if body.get("status") != 200:
        raise FinMindError(f"FinMind status {body.get('status')}: {body.get('msg')}")
    return body.get("data", [])


def fetch_daily_kline(stock_id: str, lookback_days: int = 365) -> pd.DataFrame:
    """抓取台股日 K。`stock_id` 例如 '2330'、'0050'。"""
    end = date.today()
    # 多抓一些以涵蓋非交易日
    start = end - timedelta(days=int(lookback_days * 1.6) + 30)
    rows = _request(
        "TaiwanStockPrice",
        {
            "data_id": stock_id,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        },
    )
    if not rows:
        raise FinMindError(f"FinMind 無 {stock_id} 的資料")

    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            "date": "date",
            "open": "open",
            "max": "high",
            "min": "low",
            "close": "close",
            "Trading_Volume": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    keep = ["date", "open", "high", "low", "close", "volume"]
    df = df[keep].sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    # 只保留請求視窗
    return df.tail(lookback_days).reset_index(drop=True)


@lru_cache(maxsize=64)
def search_stock(keyword: str) -> list[dict]:
    """以名稱/代碼搜尋台股。回傳 [{stock_id, stock_name, industry_category}]。"""
    rows = _request("TaiwanStockInfo", {})
    keyword = keyword.strip().lower()
    matches = [
        {
            "stock_id": r.get("stock_id"),
            "stock_name": r.get("stock_name"),
            "industry_category": r.get("industry_category"),
        }
        for r in rows
        if keyword in str(r.get("stock_id", "")).lower()
        or keyword in str(r.get("stock_name", "")).lower()
    ]
    return matches[:30]
