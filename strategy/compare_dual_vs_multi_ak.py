# -*- coding: utf-8 -*-
"""
Dual vs Multi ETF Rotation (AKShare only) 2015-01-01 ~ 2024-12-31
- 双ETF：沪深300(510300.SH) + 黄金(518880.SH)
- 多ETF：沪深300 + 黄金 + 国债ETF(511260.SH)
- 月度动量赢家通吃（回看期 L ∈ {3,6,9,12}；上月末动量、下月执行）
- 交易成本：首月单边、换仓双边（手续费+滑点）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import akshare as ak
from typing import Optional, Dict, List

# ===================== 参数 =====================
TICKS = {
    "510300.SH": "510300",   # 沪深300 ETF
    "518880.SH": "518880",   # 黄金 ETF
    "511260.SH": "511260",   # 国债 ETF
}

PAIR  = ["510300.SH", "518880.SH"]
MULTI = ["510300.SH", "518880.SH", "511260.SH"]

START = "20150101"
END   = "20241231"   # 覆盖 2015-01 ~ 2024-12

LOOKBACKS = [3, 6, 9, 12]

LEVER    = 1.0       # 杠杆倍数
FIN_ANN  = 0.045     # 年化融资利率（LEVER>1 时，对超出 1 的部分计提）
COMM_BPS = 1.0       # 手续费（万分比，单边）
SLIP_BPS = 2.0       # 滑点（万分比，单边）
RF_ANN   = 0.02      # 年化无风险利率（Sharpe用）

OUTDIR = "outputs_dual_vs_multi_ak"
os.makedirs(OUTDIR, exist_ok=True)

# ===================== 工具函数 =====================
def get_close(symbol: str, start: str, end: str) -> Optional[pd.Series]:
    """
    仅用 akshare 获取日频收盘价，返回 Series(index=Date, name=symbol)
    """
    code = TICKS[symbol]
    df = ak.fund_etf_hist_em(
        symbol=code, period="daily", start_date=start, end_date=end, adjust=""
    )
    if df is None or df.empty:
        return None

    # 兼容列名
    rename_map = {}
    if "日期" in df.columns:   rename_map["日期"] = "Date"
    if "收盘" in df.columns:   rename_map["收盘"] = "Close"
    if "收盘价" in df.columns: rename_map["收盘价"] = "Close"
    df = df.rename(columns=rename_map)

    if "Date" not in df.columns or "Close" not in df.columns:
        return None

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    s = pd.to_numeric(df["Close"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s.name = symbol
    return s

def metr(r: pd.Series, rf: float = 0.02) -> dict:
    """年化收益、波动、夏普、最大回撤、胜率、样本月数"""
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan,
                "MDD": np.nan, "WinRate": np.nan, "Months": 0}
    cagr  = (1 + r).prod() ** (12 / n) - 1
    vol   = r.std() * np.sqrt(12) if n > 1 else np.nan
    rf_m  = rf / 12.0
    shar  = ((r.mean() - rf_m) / r.std() * np.sqrt(12)) if r.std() > 0 else np.nan
    curve = (1 + r).cumprod()
    mdd   = (curve / curve.cummax() - 1.0).min()
    win   = (r > 0).mean()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": shar,
            "MDD": mdd, "WinRate": win, "Months": n}

def ensure_mon_end(prices: pd.DataFrame) -> pd.DataFrame:
    """
    转月末频率（ME = Month End）
    """
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    m_px = prices.resample("ME").last().dropna(how="all")
    return m_px

def winner_take_all_rotation(
    m_px: pd.DataFrame,
    m_ret: pd.DataFrame,
    universe: List[str],
    lb: int,
    lever: float = 1.0,
    fin_ann: float = 0.045,
    comm_bps: float = 1.0,
    slip_bps: float = 2.0,
) -> pd.Series:
    """
    赢家通吃轮动：动量 = 过去 lb 个月累计涨幅（上月末动量，下月执行）
    返回：月度收益序列（含交易成本与融资成本）
    """
    px = m_px[universe].copy()
    ret = m_ret[universe].copy()

    # 上月末动量（过去 lb 个月累计）
    mom = (px / px.shift(lb)) - 1.0
    mom = mom.shift(1).dropna(how="all")
    ret = ret.loc[mom.index]  # 对齐到动量可用期

    # 当月仓位 = 上月末动量赢家；下月初执行（再 shift(1)）
    w = pd.DataFrame(0.0, index=mom.index, columns=universe)
    win_asset = mom.idxmax(axis=1)
    for d, sym in win_asset.items():
        if sym in w.columns:
            w.loc[d, sym] = 1.0
    w = w.shift(1).fillna(0.0).loc[ret.index]

    # 换仓与成本
    prev = w.shift(1).fillna(0.0)
    switch = pd.Series((w.values != prev.values).any(axis=1), index=w.index)

    one_side  = (comm_bps + slip_bps) / 10000.0
    roundtrip = one_side * 2.0
    tc = pd.Series(0.0, index=w.index)

    active_mask = (w.sum(axis=1) > 0)
    if active_mask.any():
        first_active = active_mask.idxmax()
        tc.loc[first_active] = -one_side
    tc.loc[switch] += -roundtrip
    if active_mask.any():
        tc.loc[first_active] = -one_side  # 覆盖可能叠加

    # 杠杆与融资（仅对超出1倍部分计息）
    gross = (w * ret).sum(axis=1)
    fin   = (fin_ann / 12.0) if lever > 1.0 else 0.0
    rot   = lever * gross - max(0.0, lever - 1.0) * fin + tc
    return rot

def pick_best_lb(met_df: pd.DataFrame, prefer_lb: int = 6) -> int:
    if prefer_lb in met_df.index:
        return prefer_lb
    return int(met_df["Sharpe"].idxmax())

def pick_metric_row(met_df: pd.DataFrame, prefer_lb: int = 6) -> pd.Series:
    if prefer_lb in met_df.index:
        return met_df.loc[prefer_lb]
    return met_df.loc[met_df["Sharpe"].idxmax()]

# ===================== 获取/整理数据 =====================
series = []
for sym in TICKS.keys():
    s = get_close(sym, START, END)
    if s is None or s.empty:
        raise RuntimeError(f"no data: {sym}")
    series.append(s)

prices = pd.concat(series, axis=1).dropna(how="all")

# 关键：先做月末采样并裁剪目标区间，再计算月度收益，避免 KeyError
m_px_all = ensure_mon_end(prices)
m_px = m_px_all.loc[(m_px_all.index >= pd.Timestamp("2015-01-31")) &
                    (m_px_all.index <= pd.Timestamp("2024-12-31"))]
m_ret = m_px.pct_change().dropna(how="all")

# ===================== 买入持有（BH）基准 =====================
bh_curves: Dict[str, pd.Series] = {}
bh_m: Dict[str, dict] = {}
for sym in TICKS.keys():
    if sym in m_ret.columns:
        r = m_ret[sym].copy()
        bh_curves[sym] = (1 + r).cumprod()
        bh_m[sym] = metr(r, RF_ANN)

# ===================== 轮动：双ETF & 多ETF =====================
rotation_dual: Dict[int, pd.Series] = {}
rotation_multi: Dict[int, pd.Series] = {}
metrics_dual, metrics_multi = [], []

for lb in LOOKBACKS:
    # 双ETF
    rot_d = winner_take_all_rotation(
        m_px, m_ret, PAIR, lb,
        lever=LEVER, fin_ann=FIN_ANN,
        comm_bps=COMM_BPS, slip_bps=SLIP_BPS
    )
    rotation_dual[lb] = rot_d.copy()
    m_d = metr(rot_d, RF_ANN); m_d["LookbackM"] = lb; m_d["Group"] = "Dual(300+Gold)"
    metrics_dual.append(m_d)

    # 多ETF
    rot_m = winner_take_all_rotation(
        m_px, m_ret, MULTI, lb,
        lever=LEVER, fin_ann=FIN_ANN,
        comm_bps=COMM_BPS, slip_bps=SLIP_BPS
    )
    rotation_multi[lb] = rot_m.copy()
    m_m = metr(rot_m, RF_ANN); m_m["LookbackM"] = lb; m_m["Group"] = "Multi(300+Gold+Tbond)"
    metrics_multi.append(m_m)

    # —— 每个LB的净值/回撤图 —— 
    for group_name, curve in [
        (f"Dual LB={lb}", (1 + rot_d).cumprod()),
        (f"Multi LB={lb}", (1 + rot_m).cumprod())
    ]:
        plt.figure(figsize=(10, 6))
        plt.plot(curve, label=group_name, linewidth=2)
        # 参考：两只资产的买入持有
        for sym in PAIR:
            if sym in bh_curves:
                plt.plot(bh_curves[sym].loc[curve.index], label=f"BH {sym}", alpha=0.7)
        plt.title(f"{group_name} Rotation vs BH (2015-01 ~ 2024-12)")
        plt.grid(True, ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        fname = group_name.replace(" ", "_").replace(":", "").replace("=", "")
        plt.savefig(os.path.join(OUTDIR, f"nav_{fname}.png"), dpi=200)
        plt.close()

        peak = curve.cummax()
        dd = curve / peak - 1.0
        plt.figure(figsize=(10, 4))
        plt.plot(dd, label=f"DD {group_name}")
        plt.title(f"Drawdown {group_name}")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"dd_{fname}.png"), dpi=200)
        plt.close()

# ===================== 汇总输出 =====================
met_dual_df  = pd.DataFrame(metrics_dual).set_index("LookbackM").sort_index()
met_multi_df = pd.DataFrame(metrics_multi).set_index("LookbackM").sort_index()
bh_df        = pd.DataFrame({f"BH_{sym}": bh_m[sym] for sym in bh_m.keys()})

met_dual_df.to_csv(os.path.join(OUTDIR, "metrics_rotation_dual_10y.csv"), encoding="utf-8-sig")
met_multi_df.to_csv(os.path.join(OUTDIR, "metrics_rotation_multi_10y.csv"), encoding="utf-8-sig")
bh_df.to_csv(os.path.join(OUTDIR, "metrics_buyhold_10y.csv"), encoding="utf-8-sig")

# —— Sharpe vs Lookback（双 & 多）——
plt.figure(figsize=(10, 6))
for lb in LOOKBACKS:
    if lb in met_dual_df.index:
        plt.scatter(lb, met_dual_df.loc[lb, "Sharpe"], s=90)             # Dual: 点
    if lb in met_multi_df.index:
        plt.scatter(lb, met_multi_df.loc[lb, "Sharpe"], s=90, marker="x")# Multi: 叉
plt.xticks(LOOKBACKS)
plt.title("Sharpe vs Lookback (Dual vs Multi, 2015-2024)")
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "sharpe_vs_lookback_dual_vs_multi.png"), dpi=200)
plt.close()

# ===================== 选最佳LB并三线图对比 =====================
lb_best_dual  = pick_best_lb(met_dual_df, 6)
lb_best_multi = pick_best_lb(met_multi_df, 6)

curve_dual  = (1 + rotation_dual[lb_best_dual]).cumprod()
curve_multi = (1 + rotation_multi[lb_best_multi]).cumprod()

common_idx = curve_dual.index.intersection(curve_multi.index)
for sym in PAIR:
    if sym in bh_curves:
        common_idx = common_idx.intersection(bh_curves[sym].index)

plt.figure(figsize=(10, 6))
plt.plot(curve_dual.loc[common_idx],  label=f"Dual Rotation LB={lb_best_dual}", linewidth=2)
plt.plot(curve_multi.loc[common_idx], label=f"Multi Rotation LB={lb_best_multi}", linewidth=2)
if "510300.SH" in bh_curves:
    plt.plot(bh_curves["510300.SH"].loc[common_idx], label="BH 510300", alpha=0.8)
if "518880.SH" in bh_curves:
    plt.plot(bh_curves["518880.SH"].loc[common_idx], label="BH 518880", alpha=0.8)
plt.title("Dual vs Multi Rotation (Best LB) vs Buy&Hold (2015-01 ~ 2024-12)")
plt.grid(True, ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "nav_dual_vs_multi_best.png"), dpi=200)
plt.close()

# —— 回撤：双/多最佳 —— 
for label, curve in [
    (f"Dual LB={lb_best_dual}", curve_dual),
    (f"Multi LB={lb_best_multi}", curve_multi),
]:
    peak = curve.cummax()
    dd = curve / peak - 1.0
    plt.figure(figsize=(10, 4))
    plt.plot(dd, label=f"DD {label}")
    plt.title(f"Drawdown {label}")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    tag = label.replace(" ", "_")
    plt.savefig(os.path.join(OUTDIR, f"dd_{tag}.png"), dpi=200)
    plt.close()

# ===================== 文本对比（报告用） =====================
def _fmt(x):
    return "NA" if pd.isna(x) else f"{x*100:,.2f}%"

def pick_metric_row_print(met_df: pd.DataFrame, prefer_lb: int = 6) -> pd.Series:
    return met_df.loc[prefer_lb] if prefer_lb in met_df.index else met_df.loc[met_df["Sharpe"].idxmax()]

best_dual_row  = pick_metric_row_print(met_dual_df, 6)
best_multi_row = pick_metric_row_print(met_multi_df, 6)

print("====== 数据区间 ======")
print("月末价格范围:", m_px.index.min().date(), "~", m_px.index.max().date())
print("月度收益样本数:", len(m_ret))

print("\n====== 买入持有（BH）指标 ======")
print(pd.DataFrame({f"BH_{k}": v for k, v in bh_m.items()}))

print("\n====== 轮动（Dual：沪深300+黄金）各LB ======")
print(met_dual_df)

print("\n====== 轮动（Multi：沪深300+黄金+国债）各LB ======")
print(met_multi_df)

print("\n====== 最佳方案对比（优先LB=6，否则Sharpe最高） ======")
print(f"Dual  : LB={lb_best_dual}  Sharpe={best_dual_row['Sharpe']:.3f}  CAGR={_fmt(best_dual_row['CAGR'])}")
print(f"Multi : LB={lb_best_multi} Sharpe={best_multi_row['Sharpe']:.3f}  CAGR={_fmt(best_multi_row['CAGR'])}")

better_by_sharpe = "Dual" if (best_dual_row["Sharpe"] > best_multi_row["Sharpe"]) else "Multi"
better_by_cagr   = "Dual" if (best_dual_row["CAGR"]  > best_multi_row["CAGR"])  else "Multi"

print("\n—— 汇总判断 ——")
print(f"按夏普(Sharpe)：{better_by_sharpe} 更优")
print(f"按年化收益(CAGR)：{better_by_cagr} 更优")

print("\n输出目录:", os.path.abspath(OUTDIR))



