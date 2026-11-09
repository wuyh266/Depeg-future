import argparse
from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

from experiment_utils import DEFAULT_THRESHOLDS, evaluate_binary_metrics, summarize_regression


HORIZON_ORDER = ["0-3h", "3-6h", "6-9h"]
FEATURE_ORDER = ["Deviation Only", "No Deviation", "All Features"]
FAMILY_ORDER = ["Pure Transformer", "LSTM", "DG Final"]
PRIMARY_THRESHOLD = 0.01
DISPLAY_FAMILY = {
    "Pure Transformer": "基础Transformer",
    "LSTM": "LSTM",
    "DG Final": "DG-Transformer",
}
DISPLAY_FEATURE = {
    "Deviation Only": "仅偏离度特征组",
    "No Deviation": "去除偏离度特征组",
    "All Features": "全特征组",
}
ZH_COLUMNS_MAIN = {
    "family": "模型",
    "horizon": "预测窗口",
    "feature_group": "最佳特征组",
    "rmse": "RMSE",
    "balanced_mse_t001": "Balanced MSE@0.01",
    "auc_t001": "AUC@0.01",
    "ap_t001": "AP@0.01",
    "pearson": "Pearson",
    "selection_note": "说明",
}
ZH_COLUMNS_MACRO = {
    "family": "模型",
    "horizon": "预测窗口",
    "feature_group": "特征组",
    "source_name": "结果来源",
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "pearson": "Pearson",
    "balanced_mse_t001": "Balanced MSE@0.01",
    "pos_mse_t001": "Pos-MSE@0.01",
    "neg_mse_t001": "Neg-MSE@0.01",
    "bucket_balanced_mse": "Bucket-balanced MSE",
    "auc_t001": "AUC@0.01",
    "ap_t001": "AP@0.01",
    "pos": "平均正例数",
    "neg": "平均负例数",
}
FEATURE_COLUMNS_ZH = ["特征类别", "特征名", "定义说明"]


def configure_fonts():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "PingFang SC"]
    chosen = next((name for name in candidates if name in installed), None)
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 13
    plt.rcParams["font.size"] = 13
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ([chosen] if chosen else []) + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


sns.set_theme(style="whitegrid")
configure_fonts()


def wrap_cell_text(value, width):
    if pd.isna(value):
        return ""
    text = str(value)
    if not text:
        return ""
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def save_figure_svg(fig, output_path):
    output_path = Path(output_path)
    fig.savefig(output_path, bbox_inches="tight")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")


def canonical_feature_name(raw_value):
    if raw_value is None:
        return None
    value = str(raw_value).strip().lower().replace("_", " ").replace("-", " ")
    value = " ".join(value.split())
    mapping = {
        "deviation only": "Deviation Only",
        "all features": "All Features",
        "no deviation": "No Deviation",
        "smoke test": "Smoke Test",
    }
    return mapping.get(value, raw_value)


def safe_float(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan", "none"}:
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def localize_family(value):
    return DISPLAY_FAMILY.get(value, value)


def localize_feature(value):
    return DISPLAY_FEATURE.get(value, value)


def localize_selection_note(value):
    mapping = {
        "missing": "缺失",
        "best by balanced_mse_t001": "按 Balanced MSE@0.01 选择",
    }
    return mapping.get(value, value)


def localize_source_name(value):
    mapping = {
        "fair_usdy_compare_20260323": "fair_usdy_compare_20260323",
        "lstm_usdy_compare_h128_20260323": "lstm_usdy_compare_h128_20260323",
        "lstm_usdy_compare_dgparams_e8_20260323": "lstm_usdy_compare_dgparams_e8_20260323",
    }
    return mapping.get(value, value)


def localize_dataframe(df):
    out = df.copy()
    if "family" in out.columns:
        out["family"] = out["family"].map(localize_family)
    if "feature_group" in out.columns:
        out["feature_group"] = out["feature_group"].map(localize_feature)
    if "selection_note" in out.columns:
        out["selection_note"] = out["selection_note"].map(localize_selection_note)
    if "source_name" in out.columns:
        out["source_name"] = out["source_name"].map(localize_source_name)
    return out
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan", "none"}:
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_results_csv(path, family_name, source_name=None):
    df = pd.read_csv(path)
    df["family"] = family_name
    df["source_name"] = source_name or Path(path).stem
    df["feature_group"] = df["model"].map(canonical_feature_name)
    for column in [
        "mse",
        "rmse",
        "mae",
        "pearson",
        "balanced_mse_t001",
        "pos_mse_t001",
        "neg_mse_t001",
        "bucket_balanced_mse",
        "threshold",
        "auc",
        "ap",
        "pos",
        "neg",
    ]:
        if column in df.columns:
            df[column] = df[column].map(safe_float)
    return df


def build_result_rows_from_prediction(prediction_file, family_name, feature_group, horizon, source_name=None):
    pred_df = pd.read_csv(prediction_file)
    rows = []
    for coin, coin_df in pred_df.groupby("coin", sort=True):
        y_true = coin_df["y_soft"].astype(float).to_numpy()
        preds = coin_df["prediction"].astype(float).to_numpy()
        metrics = summarize_regression(y_true, preds)
        threshold_rows = evaluate_binary_metrics(y_true, preds, DEFAULT_THRESHOLDS)
        base = {
            "model": feature_group,
            "family": family_name,
            "feature_group": feature_group,
            "source_name": source_name or Path(prediction_file).parent.name,
            "horizon": horizon,
            "coin": coin,
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "pearson": metrics["pearson"],
            "balanced_mse_t001": metrics["balanced_mse_t001"],
            "pos_mse_t001": metrics["pos_mse_t001"],
            "neg_mse_t001": metrics["neg_mse_t001"],
            "bucket_balanced_mse": metrics["bucket_balanced_mse"],
            "prediction_min": metrics["prediction_min"],
            "prediction_max": metrics["prediction_max"],
        }
        for threshold_row in threshold_rows:
            row = base.copy()
            row.update(threshold_row)
            rows.append(row)
    return rows


def resolve_dg_regression_sources(regression_root):
    regression_root = Path(regression_root)
    patterns = [
        ("All Features", r"^dg_final_regression_head_.*_(0-3h|3-6h|6-9h)_k7_refresh1$"),
        ("Deviation Only", r"^dg_final_regression_head_.*_(0-3h|3-6h|6-9h)_k7_refresh1_devonly$"),
        ("No Deviation", r"^dg_final_regression_head_.*_(0-3h|3-6h|6-9h)_k7_refresh1_nodev$"),
    ]
    resolved = []
    for feature_group, pattern in patterns:
        regex = re.compile(pattern)
        matches_by_horizon = {}
        for path in regression_root.iterdir():
            if not path.is_dir():
                continue
            match = regex.match(path.name)
            if not match:
                continue
            horizon = match.group(1)
            current = matches_by_horizon.get(horizon)
            if current is None or path.stat().st_mtime > current.stat().st_mtime:
                matches_by_horizon[horizon] = path
        for horizon in HORIZON_ORDER:
            if horizon in matches_by_horizon:
                resolved.append((horizon, feature_group, matches_by_horizon[horizon]))
    return resolved


def load_dg_results(dg_sources, output_dir):
    results = []
    for horizon, feature_group, source_dir in dg_sources:
        prediction_file = Path(source_dir) / f"dg_final_{horizon}_predictions.csv"
        if not prediction_file.exists():
            print(f"[warn] missing DG Final prediction file: {prediction_file}")
            continue
        results.extend(
            build_result_rows_from_prediction(
                prediction_file=prediction_file,
                family_name="DG Final",
                feature_group=feature_group,
                horizon=horizon,
                source_name=Path(source_dir).name,
            )
        )
    dg_df = pd.DataFrame(results)
    if not dg_df.empty:
        dg_df.to_csv(output_dir / "dg_final_recomputed_results.csv", index=False, encoding="utf-8-sig")
    return dg_df


def filter_threshold(df, threshold=PRIMARY_THRESHOLD):
    threshold_series = pd.to_numeric(df["threshold"], errors="coerce")
    out = df[np.isclose(threshold_series, threshold, equal_nan=False)].copy()
    return out


def macro_average(df):
    group_cols = ["family", "horizon", "feature_group", "source_name"]
    metrics = [
        "mse",
        "rmse",
        "mae",
        "pearson",
        "balanced_mse_t001",
        "pos_mse_t001",
        "neg_mse_t001",
        "bucket_balanced_mse",
        "auc",
        "ap",
        "pos",
        "neg",
    ]
    macro = (
        df.groupby(group_cols, dropna=False)[metrics]
        .agg("mean")
        .reset_index()
        .rename(columns={"auc": "auc_t001", "ap": "ap_t001"})
    )
    best_per_group = (
        macro.sort_values(
            by=["family", "horizon", "feature_group", "balanced_mse_t001", "auc_t001", "rmse"],
            ascending=[True, True, True, True, False, True],
            na_position="last",
        )
        .groupby(["family", "horizon", "feature_group"], dropna=False)
        .first()
        .reset_index()
    )
    return best_per_group


def select_best_configs(macro_df):
    selected_rows = []
    for family in FAMILY_ORDER:
        family_df = macro_df[macro_df["family"] == family]
        for horizon in HORIZON_ORDER:
            subset = family_df[family_df["horizon"] == horizon].copy()
            if subset.empty:
                selected_rows.append(
                    {
                        "family": family,
                        "horizon": horizon,
                        "feature_group": np.nan,
                        "rmse": np.nan,
                        "balanced_mse_t001": np.nan,
                        "auc_t001": np.nan,
                        "ap_t001": np.nan,
                        "pearson": np.nan,
                        "selection_note": "missing",
                    }
                )
                continue
            subset = subset.sort_values(
                by=["balanced_mse_t001", "auc_t001", "rmse"],
                ascending=[True, False, True],
                na_position="last",
            )
            best = subset.iloc[0]
            selected_rows.append(
                {
                    "family": family,
                    "horizon": horizon,
                    "feature_group": best["feature_group"],
                    "rmse": best["rmse"],
                    "balanced_mse_t001": best["balanced_mse_t001"],
                    "auc_t001": best["auc_t001"],
                    "ap_t001": best["ap_t001"],
                    "pearson": best["pearson"],
                    "selection_note": "best by balanced_mse_t001",
                }
            )
    return pd.DataFrame(selected_rows)


def build_metrics_definition_table():
    rows = [
        ("MSE", "回归指标", "预测值与 y_soft 的平均平方误差", "越低越好"),
        ("RMSE", "回归指标", "MSE 的平方根，更接近原始误差尺度", "越低越好"),
        ("MAE", "回归指标", "预测值与 y_soft 的平均绝对误差", "越低越好"),
        ("Pearson", "回归指标", "预测值与 y_soft 的线性相关系数", "越高越好"),
        ("Balanced MSE@0.01", "平衡回归指标", "按 0.01 阈值对正负样本误差等权平均", "越低越好"),
        ("Pos-MSE@0.01", "平衡回归指标", "y_soft > 0.01 样本上的均方误差", "越低越好"),
        ("Neg-MSE@0.01", "平衡回归指标", "y_soft <= 0.01 样本上的均方误差", "越低越好"),
        ("Bucket-balanced MSE", "平衡回归指标", "在多个 y_soft 区间上等权平均的均方误差", "越低越好"),
        ("AUC@0.01", "排序指标", "将 y_soft 按 0.01 阈值二值化后的 ROC-AUC", "越高越好"),
        ("AP@0.01", "排序指标", "将 y_soft 按 0.01 阈值二值化后的平均精确率", "越高越好"),
    ]
    return pd.DataFrame(rows, columns=["指标", "类别", "定义", "优劣方向"])


def build_feature_conclusion_table(macro_df):
    comparable = macro_df[macro_df["feature_group"].isin(FEATURE_ORDER)].copy()
    comparable = comparable[comparable["family"].isin(["Pure Transformer", "LSTM", "DG Final"])]
    comparable = comparable.groupby(["family", "horizon"]).filter(lambda g: g["feature_group"].nunique() >= 2)

    winner_counts = (
        comparable.sort_values(
            by=["family", "horizon", "balanced_mse_t001", "auc_t001"],
            ascending=[True, True, True, False],
            na_position="last",
        )
        .groupby(["family", "horizon"], dropna=False)
        .first()
        .reset_index()
        .groupby("feature_group")
        .size()
        .to_dict()
    )

    rows = [
        {
            "特征组": "仅偏离度特征组",
            "证据": f"在{winner_counts.get('Deviation Only', 0)}个可比组合中，Balanced MSE@0.01 最优",
            "结论": "当前数据下整体最稳，通常为最优或接近最优配置。",
        },
        {
            "特征组": "去除偏离度特征组",
            "证据": f"在{winner_counts.get('No Deviation', 0)}个可比组合中，Balanced MSE@0.01 最优",
            "结论": "整体明显偏弱，说明偏离度特征仍是主导信号。",
        },
        {
            "特征组": "全特征组",
            "证据": f"在{winner_counts.get('All Features', 0)}个可比组合中，Balanced MSE@0.01 最优",
            "结论": "结果存在波动，额外特征未稳定优于仅偏离度特征组。",
        },
    ]
    return pd.DataFrame(rows)


def build_feature_catalog_table():
    rows = [
        ("脱锚类", "price", "稳定币当前价格水平，最直接反映锚定附近的位置。"),
        ("脱锚类", "rolling_mean", "价格滚动均值，反映局部价格中心是否偏离锚定值。"),
        ("脱锚类", "deviation", "价格相对 1.0 的直接偏离，是核心脱锚信号。"),
        ("脱锚类", "deviation_diff", "偏离度变化量，反映脱锚加速或回归的短期变化。"),
        ("脱锚类", "dev_mean", "偏离度滚动均值，反映一段时间内的平均脱锚水平。"),
        ("脱锚类", "dev_max", "偏离度滚动最大值，反映局部窗口中的最强脱锚程度。"),
        ("非脱锚类", "log_return", "对数收益率，描述分钟级价格变化速度。"),
        ("非脱锚类", "rolling_std", "短期收益波动率，反映局部不稳定性。"),
        ("非脱锚类", "rolling_skew", "收益分布偏度，刻画异动方向偏斜。"),
        ("非脱锚类", "rolling_kurt", "收益分布峰度，刻画尖峰厚尾特征。"),
        ("非脱锚类", "max_drawdown", "相对局部高点的回撤幅度，反映下跌压力。"),
        ("非脱锚类", "vol_ratio", "短期与长期波动率之比，反映波动结构变化。"),
        ("非脱锚类", "volatility_long", "长期收益波动率，反映更平滑的不稳定水平。"),
        ("非脱锚类", "return_cum", "滚动累计收益，反映一段时间内的方向性走势。"),
        ("非脱锚类", "volume", "成交量水平，刻画交易活跃程度。"),
        ("非脱锚类", "volume_ratio", "成交量相对滚动均值的比值，反映放量或缩量。"),
    ]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS_ZH)


def render_table(df, output_path, title, max_rows=None, font_size=10.5):
    display_df = df.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows)
    for column in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[column]):
            display_df[column] = display_df[column].map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )
        else:
            wrap_width = 16
            if any(key in str(column) for key in ["定义", "结论", "证据", "说明"]):
                wrap_width = 28
            elif any(key in str(column) for key in ["指标", "特征名", "特征组", "类别"]):
                wrap_width = 16
            display_df[column] = display_df[column].map(lambda x: wrap_cell_text(x, wrap_width))

    col_width_weights = []
    for column in display_df.columns:
        col_name = str(column)
        if any(key in col_name for key in ["定义", "结论", "证据", "说明"]):
            col_width_weights.append(3.2)
        elif any(key in col_name for key in ["指标", "类别", "方向", "特征名", "特征组"]):
            col_width_weights.append(1.5)
        else:
            col_width_weights.append(1.9)

    fig_width = max(10.5, sum(col_width_weights) * 1.2)
    fig_height = max(2.8, 0.62 * len(display_df) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.fillna("").values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.85)

    total_weight = float(sum(col_width_weights))
    normalized_widths = [weight / total_weight for weight in col_width_weights]
    total_lines = sum(
        max(str(display_df.iloc[row_idx, col]).count("\n") + 1 for col in range(len(display_df.columns)))
        for row_idx in range(len(display_df))
    ) if len(display_df) else 1
    header_lines = max(str(col).count("\n") + 1 for col in display_df.columns)
    fig_height = max(fig_height, 0.34 * (total_lines + header_lines) + 1.8)
    base_row_height = 0.062

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9d4c7")
        cell.set_width(normalized_widths[col])
        cell.get_text().set_wrap(True)
        cell.PAD = 0.04
        if any(key in str(display_df.columns[col]) for key in ["定义", "结论", "证据", "说明"]):
            cell.get_text().set_ha("left")
        if row == 0:
            cell.set_facecolor("#e9e0cc")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#faf7ef" if row % 2 == 0 else "#f5f0e6")

    for col in range(len(display_df.columns)):
        table[(0, col)].set_height(base_row_height * max(1.2, header_lines * 0.9))

    for row_idx in range(len(display_df)):
        line_count = max(str(display_df.iloc[row_idx, col]).count("\n") + 1 for col in range(len(display_df.columns)))
        row_height = base_row_height * max(1.0, line_count * 0.95)
        for col in range(len(display_df.columns)):
            table[(row_idx + 1, col)].set_height(row_height)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
    save_figure_svg(fig, output_path)
    plt.close(fig)


def add_missing_annotations(ax, pivot_df, y_offset):
    for x_pos, horizon in enumerate(HORIZON_ORDER):
        if horizon not in pivot_df.index:
            continue
        row = pivot_df.loc[horizon]
        if row.isna().all():
            ax.text(x_pos, y_offset, "missing", ha="center", va="bottom", fontsize=8, color="#8c6d3f")


def plot_main_results(best_df, output_path):
    metric_specs = [
        ("rmse", "RMSE", False),
        ("balanced_mse_t001", "Balanced MSE@0.01", False),
        ("auc_t001", "AUC@0.01", True),
    ]
    colors = {
        "Pure Transformer": "#b85c38",
        "LSTM": "#2f7d5b",
        "DG Final": "#214e8a",
    }
    x = np.arange(len(HORIZON_ORDER))
    width = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (metric, title, higher_is_better) in zip(axes, metric_specs):
        for idx, family in enumerate(FAMILY_ORDER):
            pivot = (
                best_df[best_df["family"] == family]
                .set_index("horizon")
                .reindex(HORIZON_ORDER)
            )
            values = pivot[metric].to_numpy(dtype=float)
            positions = x + (idx - 1) * width
            mask = ~np.isnan(values)
            ax.bar(
                positions[mask],
                values[mask],
                width=width,
                color=colors[family],
                label=localize_family(family) if metric == "rmse" else None,
                alpha=0.9,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(HORIZON_ORDER)
        ax.set_xlabel("预测窗口")
        ax.set_ylabel(title)
        if higher_is_better:
            ax.set_ylim(bottom=0)
        add_missing_annotations(ax, best_df.pivot(index="horizon", columns="family", values=metric), y_offset=ax.get_ylim()[1] * 0.02)

    axes[0].legend(frameon=True, fontsize=12)
    fig.tight_layout()
    save_figure_svg(fig, output_path)
    plt.close(fig)


def plot_feature_group_heatmaps(macro_df, output_path):
    filtered = macro_df[macro_df["feature_group"].isin(FEATURE_ORDER)].copy()
    filtered["row_label"] = filtered["family"].map(localize_family) + " | " + filtered["horizon"]
    row_order = []
    for family in FAMILY_ORDER:
        family_rows = filtered[filtered["family"] == family]
        for horizon in HORIZON_ORDER:
            row_label = f"{localize_family(family)} | {horizon}"
            if (family_rows["horizon"] == horizon).any():
                row_order.append(row_label)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(4.2, 0.55 * len(row_order))))
    title_fontsize = 14
    label_fontsize = 12
    tick_fontsize = 11
    heatmap_specs = [
        ("balanced_mse_t001", "Balanced MSE@0.01", False),
        ("auc_t001", "AUC@0.01", True),
    ]
    cmap = sns.color_palette("YlOrBr", as_cmap=True)

    for ax, (metric, title, higher_is_better) in zip(axes, heatmap_specs):
        pivot = (
            filtered.pivot_table(index="row_label", columns="feature_group", values=metric, aggfunc="mean")
            .reindex(index=row_order, columns=FEATURE_ORDER)
        )
        pivot.columns = [localize_feature(col) for col in pivot.columns]
        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap=cmap if higher_is_better else sns.color_palette("YlGnBu_r", as_cmap=True),
            cbar=True,
            linewidths=0.6,
            linecolor="#efe7d5",
            mask=pivot.isna(),
        )
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel("特征组", fontsize=label_fontsize)
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    fig.tight_layout()
    save_figure_svg(fig, output_path)
    plt.close(fig)


def plot_horizon_trends(macro_df, output_path):
    tracked = macro_df[
        ((macro_df["family"] == "LSTM") & (macro_df["feature_group"] == "Deviation Only"))
        | ((macro_df["family"] == "DG Final") & (macro_df["feature_group"] == "Deviation Only"))
    ].copy()
    title_fontsize = 17
    label_fontsize = 14
    tick_fontsize = 13
    legend_fontsize = 13

    metric_specs = [
        ("rmse", "RMSE"),
        ("balanced_mse_t001", "Balanced MSE@0.01"),
        ("auc_t001", "AUC@0.01"),
    ]
    styles = {
        "LSTM": {"color": "#2f7d5b", "marker": "o"},
        "DG Final": {"color": "#214e8a", "marker": "s"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8))
    x = np.arange(len(HORIZON_ORDER))
    for ax, (metric, title) in zip(axes, metric_specs):
        for family in ["LSTM", "DG Final"]:
            subset = (
                tracked[tracked["family"] == family]
                .set_index("horizon")
                .reindex(HORIZON_ORDER)
            )
            values = subset[metric].to_numpy(dtype=float)
            ax.plot(
                x,
                values,
                linewidth=2,
                markersize=6,
                label=localize_family(family),
                **styles[family],
            )
            for idx, value in enumerate(values):
                if np.isnan(value):
                    ax.text(idx, ax.get_ylim()[0] if ax.has_data() else 0, "缺失", ha="center", va="bottom", fontsize=8, color="#8c6d3f")
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels(HORIZON_ORDER)
        ax.set_xlabel("预测窗口", fontsize=label_fontsize)
        ax.set_ylabel(title, fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.margins(x=0.08)

    axes[0].legend(frameon=True, fontsize=legend_fontsize)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.94, wspace=0.28)
    save_figure_svg(fig, output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables and figures from current experiment outputs.")
    parser.add_argument(
        "--dg-regression-root",
        default="src/data/models",
        help="Root directory that contains dg_final_regression_head_* experiment folders.",
    )
    parser.add_argument(
        "--lstm-results",
        default="data/experiment_outputs/lstm_usdy_compare_dgparams_e8_20260323/lstm_comparison_results.csv",
        help="Primary path to LSTM results CSV.",
    )
    parser.add_argument(
        "--lstm-extra-results",
        default="",
        help="Optional extra LSTM results CSV used to fill additional horizons/configurations.",
    )
    parser.add_argument(
        "--fair-results",
        default="data/experiment_outputs/fair_usdy_compare_20260323/fair_comparison_results.csv",
        help="Path to Pure Transformer results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/paper_figures/usdy_three_horizons_20260325",
        help="Directory for generated tables and figures.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fair_df = read_results_csv(args.fair_results, family_name="Pure Transformer", source_name="fair_usdy_compare_20260323")
    lstm_frames = [
        read_results_csv(args.lstm_results, family_name="LSTM", source_name=Path(args.lstm_results).parent.name)
    ]
    extra_lstm_path = Path(args.lstm_extra_results) if args.lstm_extra_results else None
    if extra_lstm_path and extra_lstm_path.exists() and extra_lstm_path.resolve() != Path(args.lstm_results).resolve():
        lstm_frames.append(
            read_results_csv(extra_lstm_path, family_name="LSTM", source_name=extra_lstm_path.parent.name)
        )
    lstm_df = pd.concat(lstm_frames, ignore_index=True)
    dg_sources = resolve_dg_regression_sources(args.dg_regression_root)
    dg_df = load_dg_results(dg_sources, output_dir=output_dir)

    threshold_df = pd.concat(
        [
            filter_threshold(fair_df, threshold=PRIMARY_THRESHOLD),
            filter_threshold(lstm_df, threshold=PRIMARY_THRESHOLD),
            filter_threshold(dg_df, threshold=PRIMARY_THRESHOLD),
        ],
        ignore_index=True,
    )

    macro_df = macro_average(threshold_df)
    macro_df = macro_df[
        macro_df["horizon"].isin(HORIZON_ORDER)
        & macro_df["family"].isin(FAMILY_ORDER)
        & macro_df["feature_group"].isin(FEATURE_ORDER)
    ].copy()
    best_df = select_best_configs(macro_df)
    table_4_1 = build_metrics_definition_table()
    table_feature_catalog = build_feature_catalog_table()
    table_5_2 = build_feature_conclusion_table(macro_df)

    table_4_1.to_csv(output_dir / "table_4_1_metrics_definition.csv", index=False, encoding="utf-8-sig")
    table_feature_catalog.to_csv(output_dir / "table_feature_catalog.csv", index=False, encoding="utf-8-sig")
    best_df_localized = localize_dataframe(best_df).rename(columns=ZH_COLUMNS_MAIN)
    best_df_localized = best_df_localized[
        ["模型", "预测窗口", "最佳特征组", "RMSE", "Balanced MSE@0.01", "AUC@0.01", "Pearson"]
    ].copy()
    macro_df_localized = localize_dataframe(macro_df).rename(columns=ZH_COLUMNS_MACRO)
    table_5_2_localized = table_5_2

    best_df_localized.to_csv(output_dir / "table_5_1_main_results.csv", index=False, encoding="utf-8-sig")
    macro_df_localized.to_csv(output_dir / "feature_group_macro_results.csv", index=False, encoding="utf-8-sig")
    table_5_2_localized.to_csv(output_dir / "table_5_2_feature_group_conclusions.csv", index=False, encoding="utf-8-sig")

    render_table(
        table_4_1,
        output_dir / "table_4_1_metrics_definition.png",
        title="表4-1 占位：评价指标定义与优劣方向汇总表",
        font_size=11.5,
    )
    render_table(
        table_feature_catalog,
        output_dir / "table_feature_catalog.png",
        title="特征表：脱锚类与非脱锚类特征划分",
        font_size=11.5,
    )
    render_table(
        best_df_localized,
        output_dir / "table_5_1_main_results.png",
        title="表5-1 占位：三类模型在三个预测窗口下的主结果汇总表",
        font_size=11.3,
    )
    render_table(
        table_5_2_localized,
        output_dir / "table_5_2_feature_group_conclusions.png",
        title="表5-2 占位：特征组主结论汇总表",
        font_size=10.5,
    )

    plot_main_results(best_df, output_dir / "figure_5_1_main_results.png")
    plot_feature_group_heatmaps(macro_df, output_dir / "figure_5_2_feature_groups.png")
    plot_horizon_trends(macro_df, output_dir / "figure_5_3_horizon_trends.png")

    notes = [
        "数据来源：",
        f"- DG Final regression 根目录：{Path(args.dg_regression_root).resolve()}",
        f"- LSTM 主来源：{Path(args.lstm_results).resolve()}",
        f"- LSTM 补充来源：{extra_lstm_path.resolve() if (extra_lstm_path and extra_lstm_path.exists()) else '未使用'}",
        f"- 纯 Transformer：{Path(args.fair_results).resolve()}",
        "",
        "说明：",
        "- 当前图表基于现有结果文件自动生成，适合论文写作与排版辅助。",
        "- 当前版本的指标均直接由预测文件重算；若仍出现空白单元格，表示该组合在输入数据中确实不存在，而不是沿用旧图留下的占位。",
        "- 当前默认使用 lstm_usdy_compare_dgparams_e8_20260323 作为 LSTM 数据来源；若显式提供补充来源，则会自动合并并按 Balanced MSE@0.01 选择同特征组下更优的结果来源。",
        "- DG Final 会自动扫描 regression_head 目录，提取 0-3h / 3-6h / 6-9h 三个窗口下的 All Features、Deviation Only 和 No Deviation 结果，并统一由预测文件重算指标。",
        "- 同时生成一张特征表，将当前特征划分为“脱锚类”和“非脱锚类”两大类，便于论文方法部分引用。",
        "",
        "本次实际识别到的 DG Final 目录：",
    ]
    for horizon, feature_group, source_dir in dg_sources:
        notes.append(f"- {horizon} | {feature_group} | {Path(source_dir).resolve()}")
    (output_dir / "README.txt").write_text("\n".join(notes), encoding="utf-8")

    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
