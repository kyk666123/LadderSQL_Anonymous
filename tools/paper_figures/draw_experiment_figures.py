#!/usr/bin/env python3
"""重画实验章节三张数据图 v2 (Fig2 数据-EX散点 / Fig3 奖励信号 / Fig4 best-of-n).

v2 修改:
- 全部字体换 Times 风格衬线体 (STIXGeneral, 学术界 Times 替代标准)
- Fig4: 每个 n 只留最好基线; 基线用彩色符号+图例置顶, 图内不放名字
- Fig2: 基线各配专属颜色+符号, 图例置顶; 只有我们的星形在图内标名
- Fig3: 改为横排分组柱状图 (x=阶段), 每阶段双柱 Binary vs Hierarchical, 多色
"""
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, "figstyle", "publication.mplstyle"))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

# Okabe-Ito 色盲安全配色
C14B = "#0072B2"   # blue      -> LadderSQL-14B (全文一致)
C7B = "#D55E00"    # vermillion-> LadderSQL-7B  (全文一致)
GREEN = "#009E73"
SKY = "#56B4E9"
PINK = "#CC79A7"
ORANGE = "#E69F00"
BLACK = "#222222"
GRAY = "#8a8a8a"


# ============================================================
# Figure 3: 奖励信号分布 (横排分组柱: x=阶段, Binary vs Hierarchical)
# ============================================================

def fig_reward_density():
    # Bird-train 口径 (初始策略, 500 题 / 59 库 / 1252 错误候选)
    d = json.load(open("/path/to/LadderSQL/inference_selection/cache/reward_density_train.json"))
    bins = d["bins"]
    n = d["n_valid"]
    pct = {k: 100.0 * v / n for k, v in bins.items()}
    graded = 100.0 - pct["zero"]

    cats = ["zero", "from", "where", "select", "top"]
    xlabels = ["0\n(no signal)", "0.50–0.55\nFROM/JOIN", "0.60\n+ WHERE",
               "0.65\n+ SELECT", "0.70\nall stages"]
    hier_colors = ["#bdbdbd", ORANGE, SKY, GREEN, C14B]
    # 注: 0.70 = 通过全部活跃阶段(k*=n), 差异落在阶梯之外(LIMIT/DISTINCT), 非"差一步"
    binary_vals = [100.0, 0.0, 0.0, 0.0, 0.0]
    hier_vals = [pct[c] for c in cats]

    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    xs = range(5)
    bw = 0.36
    # Binary: 灰色斜纹柱
    ax.bar([x - bw / 2 for x in xs], binary_vals, width=bw, color="#e0e0e0",
           edgecolor="#888888", linewidth=0.5, hatch="///", label="Binary reward")
    # Hierarchical: 每阶段专属颜色
    ax.bar([x + bw / 2 for x in xs], hier_vals, width=bw,
           color=hier_colors, edgecolor="white", linewidth=0.5)

    # 柱顶数值标注
    for x, v in zip(xs, binary_vals):
        ax.text(x - bw / 2, v + 1.5, f"{v:.0f}" if v > 0 else "0",
                ha="center", va="bottom", fontsize=6.5, color="#666666")
    for x, v, c in zip(xs, hier_vals, hier_colors):
        ax.text(x + bw / 2, v + 1.5, f"{v:.1f}", ha="center", va="bottom",
                fontsize=6.5, color=c if v > 8 else "#333333", fontweight="bold")

    # 44.8% graded credit: 浅色背景带 + 顶部标注
    ax.axvspan(0.55, 4.55, color="#0072B2", alpha=0.05, zorder=0)
    ax.annotate(f"{graded:.1f}% receive graded partial credit",
                xy=(2.55, 88), ha="center", va="center", fontsize=7,
                color=C14B, fontweight="bold")
    ax.annotate("", xy=(0.6, 82), xytext=(4.5, 82),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=C14B))

    # 图例 (只需说明双柱身份; hierarchical 用无色代理)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#e0e0e0", edgecolor="#888888", hatch="///",
                     linewidth=0.5, label="Binary reward"),
               Patch(facecolor=GREEN, edgecolor="white", linewidth=0.5,
                     label="Hierarchical reward (colored by stage)")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.72),
              fontsize=6.5, frameon=False, handlelength=1.4)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(xlabels, fontsize=6.3, linespacing=1.15)
    ax.set_ylabel("Share of incorrect candidates (%)")
    ax.set_ylim(0, 108)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_reward_density.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] n={n}  zero={pct['zero']:.1f}%  graded={graded:.1f}%")


# ============================================================
# Figure 4: best-of-n 折线 (每个 n 仅保留最好基线, 符号图例置顶)
# ============================================================

def fig_best_of_n():
    ns = [1, 8, 16, 24, 32]
    ours7 = [66.4, 71.5, 71.9, 72.1, 72.5]
    ours14 = [68.8, 73.2, 73.1, 73.2, 73.7]
    # 图例名带 backbone 参数规模 + 候选数 (方法, [(n, EX)...], 颜色, marker)
    baselines = [
        ("Arctic-32B ($n$=1/8)",                   [(1, 70.5), (8, 71.5)], GREEN, "^"),
        ("XiYan-SQL (GPT-4o+32B, $n$=10)",         [(10, 73.3)],           PINK,  "D"),
        ("CHESS (Gemini-1.5-Pro, $n$=20)",         [(20, 68.3)],           GRAY,  "v"),
        ("OpenSearch-SQL (GPT-4o, $n$=21)",        [(21, 69.3)],           SKY,   "P"),
        ("OpenSQL-32B ($n$=24)",                   [(24, 70.0)],           ORANGE, "X"),
        ("DeepEye-SQL (30B-A3B, $n$=36)",          [(36, 73.5)],           BLACK, "d"),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    # 我们的曲线: 离散采样点, 用虚线连接
    l14, = ax.plot(ns, ours14, "--s", color=C14B, ms=4, lw=1.1, label="LadderSQL-14B", zorder=5)
    l7, = ax.plot(ns, ours7, "--o", color=C7B, ms=4, lw=1.1, label="LadderSQL-7B", zorder=5)
    handles = [l14, l7]
    for name, pts, color, marker in baselines:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        # zorder=7 使基线符号盖在我们折线之上; 白边提升重叠处可读性
        h, = ax.plot(xs, ys, marker, color=color, ms=5.5, mew=0.6, mec="white",
                     ls="none", label=name, zorder=7)
        handles.append(h)

    # 曲线终点数值
    ax.annotate("73.7", (32, 73.7), textcoords="offset points", xytext=(0, 6),
                fontsize=7, color=C14B, ha="center", fontweight="bold")
    ax.annotate("72.5", (32, 72.5), textcoords="offset points", xytext=(0, -10),
                fontsize=7, color=C7B, ha="center", fontweight="bold")

    # 20/21 相邻刻度: 同行展示, 分别左右对齐拉开间距
    ax.set_xticks([1, 8, 10, 16, 20, 21, 24, 32, 36])
    ax.tick_params(axis="x", labelsize=6.5)
    for lbl in ax.get_xticklabels():
        if lbl.get_text() == "20":
            lbl.set_ha("right")
        elif lbl.get_text() == "21":
            lbl.set_ha("left")
    ax.set_xlabel("Candidate budget $n$")
    ax.set_ylabel("Bird-dev EX (%)")
    ax.set_ylim(65.5, 74.8)
    ax.set_xlim(-1.5, 39)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, fontsize=5.6, frameon=False, handlelength=1.2,
              columnspacing=1.0, handletextpad=0.3, borderaxespad=0.0)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_best_of_n.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[fig4] done")


# ============================================================
# Figure 2: 训练数据量 vs EX 散点 (基线彩色符号 + 图例置顶)
# ============================================================

def fig_data_ex():
    # (名称, size, EX, 颜色, marker) —— 数值严格对齐论文 Table 2 (只计 SQL generator 训练样本)
    baselines = [
        ("SQL-Trail-14B",     2000,    66.7, SKY,    "v"),
        ("Reasoning-SQL-14B", 8000,    65.3, GREEN,  "^"),
        ("Arctic-32B",        28000,   71.5, GREEN,  "s"),
        ("SHARE-8B",          28000,   64.1, PINK,   "D"),
        ("OpenSQL-32B",       29000,   70.0, ORANGE, "X"),
        ("Reward-SQL-8B",     52000,   70.3, "#7E57C2", "<"),
        ("Text2SQL-Flow-7B",  75000,   61.5, GRAY,   "P"),
        ("SQL-R1-14B",        200000,  67.1, BLACK,  "d"),
        ("OmniSQL-32B",       2500000, 67.0, PINK,   "o"),
    ]
    ours = [
        ("LadderSQL-14B", 5298, 73.7, C14B),
        ("LadderSQL-7B",  7371, 72.5, C7B),
    ]

    # 分段对数标度: 在 log10 空间内给每段不同拉伸权重,
    # 2k–10k 适度压缩(0.9), 10k–100k 拉宽(2.0), 100k–450k 正常(1.1)
    import numpy as np
    BP = [np.log10(1000), 4.0, 5.0, np.log10(4.5e5)]
    W = [0.9, 2.0, 1.1]

    def pos(x):
        lx = np.log10(np.asarray(x, dtype=float))
        out = np.zeros_like(lx)
        for i in range(len(BP) - 1):
            lo, hi, w = BP[i], BP[i + 1], W[i]
            out = out + (np.clip(lx, lo, hi) - lo) * w
        return out

    # 断轴: 左段为分段对数主体, 右段仅 OmniSQL(2.5M)
    fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, figsize=(3.5, 2.55),
                                  gridspec_kw={"width_ratios": [8, 1], "wspace": 0.06})

    handles = []
    for name, x, y, c in ours:
        h, = ax.plot(pos(x), y, marker="*", color=c, ms=11, mew=0, ls="none",
                     label=name, zorder=5)
        handles.append(h)
    for name, x, y, c, m in baselines:
        if x > 5e5:
            h, = ax2.plot([0.5], [y], m, color=c, ms=5, mew=0, ls="none", label=name, zorder=3)
        else:
            h, = ax.plot(pos(x), y, m, color=c, ms=5, mew=0, ls="none", label=name, zorder=3)
        handles.append(h)

    # 图例已列出我们的方案, 图内不重复标名

    # better 方向箭头 (右上空白区, 指向左上)
    ax.annotate("better", xy=(0.04, 0.985), xytext=(0.20, 0.93),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=6.5, color="#999999", ha="center", va="center", style="italic",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#999999"))

    # 左段轴
    ax.set_ylim(59.5, 75.8)
    ax.set_yticks([60, 64, 68, 72])
    ax.set_xlim(pos(1000), pos(4.5e5))
    ax.set_xticks([pos(v) for v in (2e3, 1e4, 1e5, 4e5)],
                  labels=["2k", "10k", "100k", "400k"])
    ax.set_ylabel("Bird-dev EX (%)")
    # 右段轴 (仅 OmniSQL)
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0.5], labels=["2.5M"])
    ax2.tick_params(axis="y", left=False)

    # 断轴记号 (两段相邻处的斜线)
    ax.spines.right.set_visible(True); ax.spines.right.set_linestyle((0, (2, 2)))
    ax.spines.right.set_linewidth(0.5); ax.spines.right.set_color("#aaaaaa")
    ax2.spines.left.set_visible(False)
    d = 0.02
    kw = dict(transform=ax.transAxes, color="black", clip_on=False, lw=0.7)
    ax.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw2 = dict(transform=ax2.transAxes, color="black", clip_on=False, lw=0.7)
    ax2.plot((-d * 7, +d * 7), (-d, +d), **kw2)

    fig.supxlabel("Training data size (piecewise-log scale)", fontsize=8, y=0.01)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.885),
               ncol=4, fontsize=5.2, frameon=False, handlelength=1.0,
               columnspacing=0.6, handletextpad=0.25)
    fig.tight_layout(pad=0.3, rect=(0, 0.03, 1, 0.885))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_data_ex.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[fig2] done")


if __name__ == "__main__":
    fig_reward_density()
    fig_best_of_n()
    fig_data_ex()
    print("all saved to", HERE)
