"""掃描路由：批次分析多檔股票。"""
from __future__ import annotations

from fastapi import APIRouter, Query

from ..scan.scanner import scan_analysts, scan_batch, scan_top

router = APIRouter(prefix="/api/scan", tags=["scan"])


@router.get("/top")
def get_top(force: bool = Query(False, description="略過快取重新掃描")) -> dict:
    """掃描預設熱門股清單，回傳 Top 5 買入 / Top 5 賣出。"""
    return scan_top(force=force)


@router.get("/batch")
def get_batch(codes: str = Query(..., description="逗號分隔的股票代碼，例如 2330,2317,0050")) -> dict:
    """掃描指定代碼（給自選清單用）。不快取。"""
    code_list = [c.strip() for c in codes.split(",") if c.strip()][:30]
    if not code_list:
        return {"items": []}
    items = scan_batch(code_list)
    return {"items": [it.to_dict() for it in items]}


@router.get("/analysts")
def get_analysts(force: bool = Query(False, description="略過快取重新掃描")) -> dict:
    """5 位台灣知名分析師依各自流派模擬出來的 Top 5 推薦 / 避開清單。"""
    return scan_analysts(force=force)
