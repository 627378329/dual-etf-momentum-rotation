import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import akshare as ak

# ---------------- 参数 ----------------
TICKS = {
    "510300.SH": "510300",  # 沪深300 ETF
    "518880.SH": "518880",  # 黄金 ETF
}
START = "20150101"
END   = "20241231"  # 固定到年末，月频刚好覆盖 2015-01 ~ 2024-12

LOOKBACKS = [3, 6, 9, 12]

LEVER    = 1.0      # 杠杆倍数
FIN_ANN  = 0.045    # 年化融资利率（LEVER>1 时，对超出 1 的部分计提）
COMM_BPS = 1.0      # 手续费（万分比）
SLIP_BPS = 2.0      # 滑点（万分比）
RF_ANN   = 0.02     # 年化无风险利率（Sharpe用）

OUTDIR = "outputs_dual_etf"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------- 工具函数 ----------------
def get_close(symbol: str, start: str, end: str) -> pd.Series | None:
    """用 akshare 获取日频收盘价，返回 Series(index=Date, name=symbol)"""
    code = TICKS[symbol]
    df = ak.fund_etf_hist_em(
        symbol=code,
        period="daily",
        start_date=start,
        end_date=end,
        adjust=""
    )
    if df is None or df.empty:
        return None
    df = df.rename(columns={"日期": "Date", "收盘": "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    s = df["Close"].astype(float)
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

# ---------------- 获取/整理数据 ----------------
series = []
for sym in TICKS.keys():
    s = get_close(sym, START, END)
    if s is None or s.empty:
        raise RuntimeError(f"no data: {sym}")
    series.append(s)

prices = pd.concat(series, axis=1).dropna(how="all")
prices = prices[~prices.index.duplicated(keep="last")].sort_index()

# 月末频率（ME = Month End）
m_px  = prices.resample("ME").last().dropna(how="all")  # 月末价格
m_ret = m_px.pct_change().dropna(how="all")             # 月度收益

# ---------------- 买入持有基准 ----------------
bh_curves, bh_m = {}, {}
for sym in TICKS.keys():
    r = m_ret[sym].copy()
    bh_curves[sym] = (1 + r).cumprod()
    bh_m[sym]      = metr(r, RF_ANN)

# 用于保存每个 Lookback 的月度收益序列（用于后续画“最佳轮动”曲线）
rotation_rets = {}

# ---------------- 轮动策略（赢家通吃） ----------------
all_metrics = []

for lb in LOOKBACKS:
    # 动量 = 过去 lb 个月累计涨幅；用“上月末”的动量决定“本月”持仓（再 shift 一月执行）
    mom = (m_px / m_px.shift(lb)) - 1.0
    mom = mom.shift(1).dropna(how="all")
    ret = m_ret.loc[mom.index]  # 对齐收益

    # 赢家=1，其余=0；下月初执行
    w = pd.DataFrame(0.0, index=mom.index, columns=mom.columns)
    win_asset = mom.idxmax(axis=1)
    for d, sym in win_asset.items():
        w.loc[d, sym] = 1.0
    w = w.shift(1).fillna(0.0).loc[ret.index]

    # 换仓检测
    prev   = w.shift(1).fillna(0.0)
    switch = pd.Series((w.values != prev.values).any(axis=1), index=w.index)

    # 成本设置：首次进场单边，之后换仓双边
    one_side  = (COMM_BPS + SLIP_BPS) / 10000.0
    roundtrip = one_side * 2.0
    tc = pd.Series(0.0, index=w.index)
    active_mask = (w.sum(axis=1) > 0)
    if active_mask.any():
        first_active = active_mask.idxmax()  # 第一个 True 的时间戳
        tc.loc[first_active] = -one_side
    tc.loc[switch] += -roundtrip  # 若与首月重叠，此处会被下面单边覆盖
    # 确保首次进场最终是单边（覆盖可能的叠加）
    if active_mask.any():
        tc.loc[first_active] = -one_side

    # 杠杆 & 融资
    gross = (w * ret).sum(axis=1)
    fin = (FIN_ANN / 12.0) if LEVER > 1.0 else 0.0

    rot = LEVER * gross - max(0.0, LEVER - 1.0) * fin + tc

    # 存收益用于后续画“最佳轮动”曲线
    rotation_rets[lb] = rot.copy()

    # 记录指标
    rot_m = metr(rot, RF_ANN); rot_m["LookbackM"] = lb
    all_metrics.append(rot_m)

    # 可选：逐LB净值/回撤图（如不需要可注释）
    curve = (1 + rot).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(curve, label=f"Rotation LB={lb}m")
    for sym in TICKS.keys():
        plt.plot(bh_curves[sym].loc[curve.index], label=f"BH {sym}")
    plt.title(f"Dual-ETF Rotation (LB={lb}m): 510300.SH vs 518880.SH")
    plt.grid(True, ls="--", alpha=0.5); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"nav_curve_gold_lb{lb}.png"), dpi=200); plt.close()

    peak = curve.cummax(); dd = curve / peak - 1.0
    plt.figure(figsize=(10, 4)); plt.plot(dd, label=f"DD LB={lb}m")
    plt.title(f"Drawdown Rotation (LB={lb}m) Gold Pair")
    plt.grid(True, ls="--", alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"drawdown_gold_lb{lb}.png"), dpi=200); plt.close()

# ---------------- 汇总输出 ----------------
met_df = pd.DataFrame(all_metrics).set_index("LookbackM").sort_index()
bh_df  = pd.DataFrame({f"BH_{sym}": bh_m[sym] for sym in TICKS.keys()})

met_df.to_csv(os.path.join(OUTDIR, "metrics_rotation_10y_gold.csv"), encoding="utf-8-sig")
bh_df.to_csv(os.path.join(OUTDIR, "metrics_buyhold_10y_gold.csv"), encoding="utf-8-sig")

# Sharpe vs Lookback（直接用内存）
plt.figure(figsize=(10, 6))
for lb in LOOKBACKS:
    plt.scatter(lb, met_df.loc[lb, "Sharpe"], s=80)
plt.xticks(LOOKBACKS)
plt.title("Sharpe vs Lookback (10Y) Gold Pair")
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "sharpe_vs_lookback_10y_gold.png"), dpi=200)
plt.close()

# ---------------- 三线净值对比图 + 最佳轮动回撤图 ----------------
# 选“最佳轮动”：优先用 LB=6（预设参数），否则取 Sharpe 最高的
if 6 in LOOKBACKS:
    lb_best = 6
else:
    lb_best = met_df["Sharpe"].idxmax()

bh_300  = bh_curves["510300.SH"]
bh_gold = bh_curves["518880.SH"]
rot_best_curve = (1 + rotation_rets[lb_best]).cumprod()

# 对齐索引并画三线净值图
common_idx = bh_300.index.intersection(bh_gold.index).intersection(rot_best_curve.index)
plt.figure(figsize=(10, 6))
plt.plot(bh_300.loc[common_idx],  label="Buy&Hold 510300")
plt.plot(bh_gold.loc[common_idx], label="Buy&Hold 518880")
plt.plot(rot_best_curve.loc[common_idx], label=f"Rotation LB={lb_best}m", linewidth=2)
plt.title("Dual-ETF Rotation vs Buy&Hold (2015-01 ~ 2024-12)")
plt.grid(True, ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "nav_curve_BH_vs_rotation_best.png"), dpi=200)
plt.close()

# 最佳轮动回撤图
peak = rot_best_curve.cummax()
dd_best = rot_best_curve / peak - 1.0
plt.figure(figsize=(10, 4))
plt.plot(dd_best, label=f"Drawdown Rotation LB={lb_best}m")
plt.title(f"Drawdown (Rotation LB={lb_best}m)")
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, f"drawdown_rotation_best_lb{lb_best}.png"), dpi=200)
plt.close()

# ---------------- 打印检查 ----------------
print("m_px range (月末价格):", m_px.index.min().date(), "~", m_px.index.max().date())
print("m_ret range (月度收益):", m_ret.index.min().date(), "~", m_ret.index.max().date())
print("months (月度收益样本数):", len(m_ret))
print("\nBuy & Hold:\n", bh_df)
print("\nRotation (fixed 10Y):\n", met_df)
print("最佳LB:", lb_best)
print("输出目录:", os.path.abspath(OUTDIR))






