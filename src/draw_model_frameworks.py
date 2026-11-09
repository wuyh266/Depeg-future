import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def configure_fonts():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "PingFang SC"]
    chosen = next((name for name in candidates if name in installed), None)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ([chosen] if chosen else []) + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 180
    plt.rcParams["savefig.dpi"] = 320


configure_fonts()


PALETTE = {
    "bg": "#fbf8f1",
    "text": "#1f2937",
    "muted": "#667085",
    "line": "#8a8172",
    "lstm": "#2d6a4f",
    "transformer": "#1d4e89",
    "dg": "#a44a3f",
    "light_green": "#dcefe6",
    "light_blue": "#dce8f6",
    "light_red": "#f4ddd8",
    "light_gold": "#f1e7cf",
    "white": "#fffdfa",
}


def add_box(ax, x, y, w, h, text, fc, ec, fontsize=11, weight="normal", rounding=0.025):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={rounding}",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=PALETTE["text"],
        fontweight=weight,
        linespacing=1.3,
    )
    return patch


def add_arrow(ax, x1, y1, x2, y2, color=None, lw=1.8, style="-|>", mutation=13, alpha=1.0):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color or PALETTE["line"],
        alpha=alpha,
        shrinkA=5,
        shrinkB=5,
        connectionstyle="arc3",
    )
    ax.add_patch(arrow)
    return arrow


def add_caption(ax, x, y, text, fontsize=9, color=None, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color or PALETTE["muted"])


def draw_panel_title(ax, x0, width, title, subtitle, accent):
    ax.text(
        x0 + width / 2,
        0.93,
        title,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=accent,
    )
    ax.text(
        x0 + width / 2,
        0.895,
        subtitle,
        ha="center",
        va="center",
        fontsize=10,
        color=PALETTE["muted"],
    )


def draw_lstm_panel(ax, x0, width):
    accent = PALETTE["lstm"]
    draw_panel_title(ax, x0, width, "LSTM 基线", "递归时序建模 + 单头回归", accent)

    add_box(ax, x0 + 0.06, 0.77, 0.24, 0.08, "输入序列\n历史价格 / 偏离度 / 统计特征", PALETTE["light_green"], accent, 11, "bold")
    add_caption(ax, x0 + 0.18, 0.74, "形状: seq_len × d_in")

    add_box(ax, x0 + 0.06, 0.60, 0.24, 0.08, "标准化\nStandardScaler", PALETTE["white"], accent, 11)
    add_box(ax, x0 + 0.06, 0.42, 0.24, 0.10, "单层 LSTM\nhidden_dim = 32", PALETTE["light_green"], accent, 12, "bold")
    add_box(ax, x0 + 0.06, 0.22, 0.24, 0.10, "末端隐藏状态 h_T\n序列摘要表示 z", PALETTE["white"], accent, 11)
    add_box(ax, x0 + 0.06, 0.06, 0.24, 0.08, "线性层 + Sigmoid\n输出风险分数 ŷ", PALETTE["light_green"], accent, 11, "bold")

    add_arrow(ax, x0 + 0.18, 0.77, x0 + 0.18, 0.68, accent)
    add_arrow(ax, x0 + 0.18, 0.60, x0 + 0.18, 0.52, accent)
    add_arrow(ax, x0 + 0.18, 0.42, x0 + 0.18, 0.32, accent)
    add_arrow(ax, x0 + 0.18, 0.22, x0 + 0.18, 0.14, accent)

    add_caption(ax, x0 + 0.18, 0.005, "训练目标: MSELoss\n特点: 结构简单, 连续值拟合稳定", 9.5)


def draw_transformer_panel(ax, x0, width):
    accent = PALETTE["transformer"]
    draw_panel_title(ax, x0, width, "纯 Transformer", "自注意力编码 + 单头回归", accent)

    add_box(ax, x0 + 0.06, 0.77, 0.24, 0.08, "输入序列\n历史价格 / 偏离度 / 统计特征", PALETTE["light_blue"], accent, 11, "bold")
    add_caption(ax, x0 + 0.18, 0.74, "形状: seq_len × d_in")

    add_box(ax, x0 + 0.06, 0.60, 0.24, 0.08, "线性投影\ninput_proj → d_model=64", PALETTE["white"], accent, 11)
    add_box(ax, x0 + 0.06, 0.39, 0.24, 0.13, "Transformer Encoder\n2 层, 4 头注意力\ndropout = 0.3", PALETTE["light_blue"], accent, 11.5, "bold")
    add_box(ax, x0 + 0.06, 0.22, 0.24, 0.10, "取最后时间步表示\nz = h[:, -1]", PALETTE["white"], accent, 11)
    add_box(ax, x0 + 0.06, 0.06, 0.24, 0.08, "线性层 + Sigmoid\n输出风险分数 ŷ", PALETTE["light_blue"], accent, 11, "bold")

    add_arrow(ax, x0 + 0.18, 0.77, x0 + 0.18, 0.68, accent)
    add_arrow(ax, x0 + 0.18, 0.60, x0 + 0.18, 0.52, accent)
    add_arrow(ax, x0 + 0.18, 0.39, x0 + 0.18, 0.32, accent)
    add_arrow(ax, x0 + 0.18, 0.22, x0 + 0.18, 0.14, accent)

    add_caption(ax, x0 + 0.18, 0.005, "训练目标: MSELoss\n特点: 并行性强, 但当前任务中稳定性较弱", 9.5)


def draw_dg_panel(ax, x0, width):
    accent = PALETTE["dg"]
    draw_panel_title(ax, x0, width, "DG Final", "序列表征 + 环境路由 + 多环境头融合", accent)

    # main input branch
    add_box(ax, x0 + 0.02, 0.79, 0.18, 0.07, "输入序列 X\n历史价格 / 偏离度 / 时序特征", PALETTE["light_red"], accent, 10.5, "bold")
    add_box(ax, x0 + 0.02, 0.62, 0.18, 0.08, "PriceTransformer 编码器\ninput_proj + TransformerEncoder", PALETTE["white"], accent, 10.5)
    add_box(ax, x0 + 0.02, 0.47, 0.18, 0.07, "序列表征 z", PALETTE["light_red"], accent, 11, "bold")

    # stats branch
    add_box(ax, x0 + 0.24, 0.79, 0.14, 0.07, "统计特征 s\nvol / dev / skew 等", PALETTE["light_gold"], accent, 10.5, "bold")
    add_box(ax, x0 + 0.24, 0.62, 0.14, 0.08, "stats_encoder\n线性映射", PALETTE["white"], accent, 10.5)
    add_box(ax, x0 + 0.24, 0.47, 0.14, 0.07, "统计表示", PALETTE["light_gold"], accent, 10.5)

    # routing
    add_box(ax, x0 + 0.10, 0.29, 0.20, 0.10, "环境编码器 env_encoder\n融合 [z ; stats_emb]\n得到 env_repr", PALETTE["light_red"], accent, 10.5, "bold")
    add_box(ax, x0 + 0.10, 0.13, 0.20, 0.10, "到环境中心的距离\nsoftmax(-dist / temperature)\n得到路由概率 p_env", PALETTE["white"], accent, 10.5)

    # heads
    add_box(ax, x0 + 0.34, 0.29, 0.10, 0.10, "K 个环境头\nhead_1 ... head_K", PALETTE["light_red"], accent, 10.5, "bold")
    add_box(ax, x0 + 0.34, 0.13, 0.10, 0.10, "加权融合\nŷ = Σ p_env · head_k", PALETTE["white"], accent, 10.5)

    add_box(ax, x0 + 0.24, 0.02, 0.14, 0.07, "输出风险分数 ŷ", PALETTE["light_red"], accent, 11, "bold")

    # arrows
    add_arrow(ax, x0 + 0.11, 0.79, x0 + 0.11, 0.70, accent)
    add_arrow(ax, x0 + 0.11, 0.62, x0 + 0.11, 0.54, accent)
    add_arrow(ax, x0 + 0.31, 0.79, x0 + 0.31, 0.70, accent)
    add_arrow(ax, x0 + 0.31, 0.62, x0 + 0.31, 0.54, accent)
    add_arrow(ax, x0 + 0.11, 0.47, x0 + 0.18, 0.39, accent)
    add_arrow(ax, x0 + 0.31, 0.47, x0 + 0.22, 0.39, accent)
    add_arrow(ax, x0 + 0.20, 0.29, x0 + 0.20, 0.23, accent)
    add_arrow(ax, x0 + 0.30, 0.18, x0 + 0.34, 0.18, accent)
    add_arrow(ax, x0 + 0.39, 0.13, x0 + 0.31, 0.09, accent)

    # environment centers callout
    add_box(ax, x0 + 0.42, 0.57, 0.11, 0.11, "环境中心\nc1 ... cK\nKMeans 初始化\n周期刷新", PALETTE["light_gold"], accent, 10.0, "bold")
    add_arrow(ax, x0 + 0.42, 0.57, x0 + 0.26, 0.19, accent, lw=1.5, style="-|>", mutation=11)

    # training objective callout
    add_box(ax, x0 + 0.41, 0.01, 0.12, 0.14, "训练目标\nERM + VREx\n+ 熵正则\n+ 动态刷新环境中心", PALETTE["light_gold"], accent, 10.0, "bold")
    add_arrow(ax, x0 + 0.38, 0.055, x0 + 0.34, 0.07, accent, lw=1.5, mutation=11)

    add_caption(ax, x0 + 0.26, -0.005, "特点: 通过环境路由与多头融合提升跨域泛化能力", 9.5)


def draw_shared_header(ax):
    ax.text(
        0.5,
        0.985,
        "三类风险预测模型框架对比",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        color=PALETTE["text"],
    )
    ax.text(
        0.5,
        0.955,
        "用于稳定币未来风险评分任务的三种代表性序列建模方案",
        ha="center",
        va="top",
        fontsize=11,
        color=PALETTE["muted"],
    )


def draw_legend(ax):
    legend_y = 0.92
    entries = [
        (PALETTE["light_green"], PALETTE["lstm"], "LSTM 组件"),
        (PALETTE["light_blue"], PALETTE["transformer"], "Transformer 组件"),
        (PALETTE["light_red"], PALETTE["dg"], "DG Final 核心组件"),
        (PALETTE["light_gold"], PALETTE["dg"], "辅助机制 / 训练策略"),
    ]
    x = 0.04
    for fc, ec, label in entries:
        add_box(ax, x, legend_y, 0.028, 0.026, "", fc, ec, rounding=0.01)
        ax.text(x + 0.036, legend_y + 0.013, label, va="center", ha="left", fontsize=9.5, color=PALETTE["muted"])
        x += 0.19


def make_figure():
    fig = plt.figure(figsize=(18, 7.8), facecolor=PALETTE["bg"])
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(PALETTE["bg"])

    draw_shared_header(ax)
    draw_legend(ax)

    panel_specs = [
        (0.03, 0.28, draw_lstm_panel),
        (0.36, 0.28, draw_transformer_panel),
        (0.67, 0.30, draw_dg_panel),
    ]

    for x0, width, drawer in panel_specs:
        panel = FancyBboxPatch(
            (x0, 0.04),
            width,
            0.86,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.2,
            edgecolor="#d8cfbf",
            facecolor=PALETTE["white"],
        )
        ax.add_patch(panel)
        drawer(ax, x0, width)

    ax.text(
        0.5,
        0.018,
        "注: 图中结构与当前代码实现一致。LSTM 与纯 Transformer 使用单头回归；DG Final 在序列表征之上显式引入环境路由、多环境头与环境中心刷新机制。",
        ha="center",
        va="center",
        fontsize=9.3,
        color=PALETTE["muted"],
    )
    return fig


def draw_single_title(ax, title, subtitle, accent):
    ax.text(0.5, 0.965, title, ha="center", va="top", fontsize=20, fontweight="bold", color=accent)
    ax.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=11, color=PALETTE["muted"])


def draw_single_lstm(ax):
    accent = PALETTE["lstm"]
    draw_single_title(ax, "LSTM 基线模型框架", "递归时序建模与单头风险回归", accent)

    x, w = 0.24, 0.52
    boxes = [
        (0.76, 0.085, "输入序列\n历史价格 / 偏离度 / 统计特征", PALETTE["light_green"], True),
        (0.61, 0.075, "标准化\nStandardScaler", PALETTE["white"], False),
        (0.43, 0.105, "单层 LSTM\nhidden_dim = 32", PALETTE["light_green"], True),
        (0.26, 0.085, "末端隐藏状态 h_T\n作为序列摘要表示 z", PALETTE["white"], False),
        (0.11, 0.085, "线性层 + Sigmoid\n输出风险分数 ŷ", PALETTE["light_green"], True),
    ]

    prev = None
    for y, h, text, fc, bold in boxes:
        add_box(ax, x, y, w, h, text, fc, accent, fontsize=12 if bold else 11, weight="bold" if bold else "normal")
        center = (x + w / 2, y + h / 2)
        if prev is not None:
            add_arrow(ax, prev[0], prev[1] - 0.055, center[0], center[1] + 0.055, accent)
        prev = center

    add_caption(ax, 0.5, 0.045, "训练目标: MSELoss    输出: 连续风险评分", fontsize=10)


def draw_single_transformer(ax):
    accent = PALETTE["transformer"]
    draw_single_title(ax, "纯 Transformer 模型框架", "自注意力序列编码与单头风险回归", accent)

    x, w = 0.22, 0.56
    boxes = [
        (0.76, 0.085, "输入序列\n历史价格 / 偏离度 / 统计特征", PALETTE["light_blue"], True),
        (0.61, 0.078, "线性投影\ninput_proj → d_model = 64", PALETTE["white"], False),
        (0.40, 0.135, "Transformer Encoder\n2 层编码器, 4 头注意力\n前馈层维度 = 256, dropout = 0.3", PALETTE["light_blue"], True),
        (0.22, 0.092, "取最后时间步表示\nz = h[:, -1]", PALETTE["white"], False),
        (0.08, 0.085, "线性层 + Sigmoid\n输出风险分数 ŷ", PALETTE["light_blue"], True),
    ]

    prev = None
    for y, h, text, fc, bold in boxes:
        add_box(ax, x, y, w, h, text, fc, accent, fontsize=11.5 if bold else 11, weight="bold" if bold else "normal")
        center = (x + w / 2, y + h / 2)
        if prev is not None:
            add_arrow(ax, prev[0], prev[1] - 0.06, center[0], center[1] + 0.06, accent)
        prev = center

    add_caption(ax, 0.5, 0.03, "训练目标: MSELoss    特点: 并行计算友好, 但当前任务稳定性较弱", fontsize=10)


def draw_single_dg(ax):
    accent = PALETTE["dg"]
    draw_single_title(ax, "DG Final 模型框架", "环境路由、多环境头与动态中心刷新的跨域建模方案", accent)

    add_box(ax, 0.08, 0.77, 0.34, 0.085, "输入序列 X\n历史价格 / 偏离度 / 时序特征", PALETTE["light_red"], accent, 11.5, "bold")
    add_box(ax, 0.58, 0.77, 0.24, 0.085, "统计特征 s\nvol / dev / skew 等", PALETTE["light_gold"], accent, 11.5, "bold")

    add_box(ax, 0.08, 0.61, 0.34, 0.082, "PriceTransformer 编码器\ninput_proj + TransformerEncoder", PALETTE["white"], accent, 11)
    add_box(ax, 0.58, 0.61, 0.24, 0.082, "stats_encoder\n线性映射", PALETTE["white"], accent, 11)
    add_arrow(ax, 0.25, 0.77, 0.25, 0.69, accent)
    add_arrow(ax, 0.70, 0.77, 0.70, 0.69, accent)

    add_box(ax, 0.11, 0.47, 0.18, 0.072, "序列表征 z", PALETTE["light_red"], accent, 11.5, "bold")
    add_box(ax, 0.60, 0.47, 0.16, 0.072, "统计表示", PALETTE["light_gold"], accent, 11.5, "bold")
    add_arrow(ax, 0.25, 0.61, 0.20, 0.54, accent)
    add_arrow(ax, 0.70, 0.61, 0.68, 0.54, accent)

    add_box(ax, 0.27, 0.33, 0.32, 0.10, "环境编码器 env_encoder\n融合 [z ; stats_emb] 得到 env_repr", PALETTE["light_red"], accent, 11, "bold")
    add_arrow(ax, 0.20, 0.47, 0.37, 0.43, accent)
    add_arrow(ax, 0.68, 0.47, 0.49, 0.43, accent)

    add_box(ax, 0.27, 0.17, 0.32, 0.10, "环境路由\nsoftmax(-dist(env_repr, centers) / temperature)\n得到路由概率 p_env", PALETTE["white"], accent, 10.5)
    add_arrow(ax, 0.43, 0.33, 0.43, 0.27, accent)

    add_box(ax, 0.67, 0.27, 0.20, 0.13, "环境中心\nc1 ... cK\nKMeans 初始化\n训练过程中周期刷新", PALETTE["light_gold"], accent, 10.4, "bold")
    add_arrow(ax, 0.67, 0.31, 0.59, 0.22, accent, lw=1.4)

    add_box(ax, 0.08, 0.09, 0.16, 0.12, "K 个环境头\nhead1 ... headK", PALETTE["light_red"], accent, 10.8, "bold")
    add_box(ax, 0.67, 0.09, 0.20, 0.12, "加权融合\nŷ = Σ p_env · head_k", PALETTE["white"], accent, 10.8)
    add_arrow(ax, 0.27, 0.18, 0.24, 0.17, accent)
    add_arrow(ax, 0.59, 0.18, 0.67, 0.17, accent)
    add_box(ax, 0.38, 0.015, 0.18, 0.06, "输出风险分数 ŷ", PALETTE["light_red"], accent, 11.5, "bold")
    add_arrow(ax, 0.77, 0.09, 0.50, 0.075, accent)

    add_caption(ax, 0.5, 0.002, "训练目标: ERM + VREx + 熵正则    目标: 提升跨币种泛化稳定性", fontsize=9.6)


def make_single_panel_figure(panel_name):
    panel_map = {
        "lstm": draw_single_lstm,
        "transformer": draw_single_transformer,
        "dg_final": draw_single_dg,
    }

    fig = plt.figure(figsize=(9.2, 7.2), facecolor=PALETTE["bg"])
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(PALETTE["bg"])
    panel = FancyBboxPatch(
        (0.06, 0.05),
        0.88,
        0.86,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.25,
        edgecolor="#d8cfbf",
        facecolor=PALETTE["white"],
    )
    ax.add_patch(panel)
    panel_map[panel_name](ax)
    return fig


def export_all(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", facecolor=fig.get_facecolor())


def main():
    parser = argparse.ArgumentParser(description="Draw publication-style framework diagrams for three model families.")
    parser.add_argument(
        "--output-dir",
        default="data/paper_figures/model_frameworks_20260325",
        help="Output directory for generated framework diagrams.",
    )
    parser.add_argument(
        "--stem",
        default="figure_model_frameworks",
        help="Output filename stem without extension.",
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Only export one SVG per model instead of the combined comparison figure.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not args.single_only:
        fig = make_figure()
        export_all(fig, output_dir, args.stem)
        plt.close(fig)

    for panel_name, stem in [
        ("lstm", "figure_lstm_framework"),
        ("transformer", "figure_transformer_framework"),
        ("dg_final", "figure_dg_final_framework"),
    ]:
        single_fig = make_single_panel_figure(panel_name)
        (output_dir / f"{stem}.svg").parent.mkdir(parents=True, exist_ok=True)
        single_fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", facecolor=single_fig.get_facecolor())
        plt.close(single_fig)

    print(f"Wrote framework figures to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
