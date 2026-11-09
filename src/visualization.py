"""
Depeg Detection Visualization - 论文图表生成
生成4张核心图 + 1张汇总表格
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr
import os

plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150
sns.set_style("whitegrid")

HORIZONS = ['0-3h', '3-6h', '6-12h', '12-18h', '18-24h', '24-30h', '48-54h']
TAUS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
COINS = ['EURC', 'TUSD', 'XUSD', 'USDY', 'UST']
DATA_PATH = 'data/models/'
OUTPUT_PATH = 'data/figures/'

os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_predictions(horizon):
    """加载指定horizon的预测结果"""
    path = f'{DATA_PATH}dg_final_{horizon}_predictions.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_all_predictions():
    """加载所有horizon的预测结果"""
    all_data = {}
    for h in HORIZONS:
        df = load_predictions(h)
        if df is not None:
            all_data[h] = df
    return all_data


def compute_regression_metrics(y_true, y_pred):
    """计算回归指标"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    
    # Correlation
    pearson, _ = pearsonr(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Pearson': pearson, 'Spearman': spearman}


def compute_tolerance_curve(y_true, y_pred, eps_range=(0, 0.3), n_points=50):
    """计算容忍度曲线"""
    epsilons = np.linspace(eps_range[0], eps_range[1], n_points)
    accuracies = [(np.abs(y_true - y_pred) <= e).mean() for e in epsilons]
    return epsilons, accuracies


# ============================================================================
# Fig.1: ROC + PR Curves
# ============================================================================
def plot_roc_pr_curves(all_data):
    """生成 Fig.1: ROC 和 PR 曲线"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    # 选择几个代表性horizon
    selected_horizons = ['0-3h', '3-6h', '6-12h', '12-18h']
    colors = plt.cm.viridis(np.linspace(0, 1, len(TAUS)))
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    for idx, horizon in enumerate(selected_horizons):
        ax_roc = axes[0, idx]
        ax_pr = axes[1, idx]
        
        if horizon not in all_data:
            ax_roc.set_title(f'{horizon}\n(No Data)')
            ax_pr.set_title(f'{horizon}\n(No Data)')
            continue
        
        df = all_data[horizon]
        
        for i, tau in enumerate(TAUS):
            y_binary = (df['y_soft'] > tau).astype(int)
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                continue
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_binary, df['prediction'])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=colors[i], linestyle=linestyles[i],
                       label=f'τ={tau} (AUC={roc_auc:.3f})', linewidth=1.5)
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_binary, df['prediction'])
            ap = average_precision_score(y_binary, df['prediction'])
            ax_pr.plot(recall, precision, color=colors[i], linestyle=linestyles[i],
                      label=f'τ={tau} (AP={ap:.3f})', linewidth=1.5)
        
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC - {horizon}')
        ax_roc.legend(loc='lower right', fontsize=7)
        ax_roc.set_xlim([0, 1])
        ax_roc.set_ylim([0, 1.05])
        
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'PR - {horizon}')
        ax_pr.legend(loc='upper right', fontsize=7)
        ax_pr.set_xlim([0, 1])
        ax_pr.set_ylim([0, 1.05])
    
    plt.tight_layout()
    output_path = f'{OUTPUT_PATH}fig1_roc_pr_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Fig.1 saved: {output_path}')
    return output_path


# ============================================================================
# Fig.2: Tolerance Curve (核心创新)
# ============================================================================
def plot_tolerance_curves(all_data):
    """生成 Fig.2: 容忍度曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizon_colors = plt.cm.Set2(np.linspace(0, 1, len(HORIZONS)))
    
    summary_data = []
    
    for i, horizon in enumerate(HORIZONS):
        if horizon not in all_data:
            continue
        
        df = all_data[horizon]
        epsilons, accuracies = compute_tolerance_curve(df['y_soft'], df['prediction'])
        
        ax.plot(epsilons, accuracies, color=horizon_colors[i], 
                label=horizon, linewidth=2, alpha=0.8)
        
        # 记录关键点的值
        for eps in [0.05, 0.10, 0.20]:
            acc = (np.abs(df['y_soft'] - df['prediction']) <= eps).mean()
            summary_data.append({
                'horizon': horizon,
                'epsilon': eps,
                'accuracy': acc
            })
    
    ax.set_xlabel('Error Tolerance (ε)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Figure 2: Tolerance-based Accuracy under Varying Error Bounds', fontsize=14)
    ax.legend(title='Horizon', loc='lower right')
    ax.set_xlim([0, 0.3])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f'{OUTPUT_PATH}fig2_tolerance_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Fig.2 saved: {output_path}')
    
    # 打印关键数据
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        pivot = summary_df.pivot(index='horizon', columns='epsilon', values='accuracy')
        print('\nTolerance Accuracy @ ε = 0.05, 0.10, 0.20:')
        print(pivot.round(4))
    
    return output_path


# ============================================================================
# Fig.3: Cross-coin Generalization (Heatmap)
# ============================================================================
def plot_generalization_heatmap(all_data):
    """生成 Fig.3: 跨币种泛化热力图"""
    tau = 0.01  # 使用 τ=0.01 作为主要指标
    
    auc_matrix = []
    ap_matrix = []
    pos_rate_matrix = []
    
    for horizon in HORIZONS:
        if horizon not in all_data:
            auc_matrix.append([np.nan] * len(COINS))
            ap_matrix.append([np.nan] * len(COINS))
            pos_rate_matrix.append([np.nan] * len(COINS))
            continue
        
        df = all_data[horizon]
        auc_row = []
        ap_row = []
        pos_row = []
        
        for coin in COINS:
            coin_df = df[df['coin'] == coin]
            if len(coin_df) == 0:
                auc_row.append(np.nan)
                ap_row.append(np.nan)
                pos_row.append(np.nan)
                continue
            
            y_binary = (coin_df['y_soft'] > tau).astype(int)
            pos_rate = y_binary.mean()
            
            if y_binary.sum() > 0 and (len(y_binary) - y_binary.sum()) > 0:
                auc_val = roc_auc_score(y_binary, coin_df['prediction'])
                ap_val = average_precision_score(y_binary, coin_df['prediction'])
            else:
                auc_val = np.nan
                ap_val = np.nan
            
            auc_row.append(auc_val)
            ap_row.append(ap_val)
            pos_row.append(pos_rate)
        
        auc_matrix.append(auc_row)
        ap_matrix.append(ap_row)
        pos_rate_matrix.append(pos_row)
    
    # 创建双子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # AUC Heatmap
    auc_df = pd.DataFrame(auc_matrix, index=HORIZONS, columns=COINS)
    sns.heatmap(auc_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.5, vmax=1.0, ax=axes[0], 
                cbar_kws={'label': 'AUC'})
    axes[0].set_title(f'Figure 3a: AUC (τ={tau})', fontsize=14)
    axes[0].set_xlabel('Stablecoin')
    axes[0].set_ylabel('Horizon')
    
    # AP Heatmap
    ap_df = pd.DataFrame(ap_matrix, index=HORIZONS, columns=COINS)
    sns.heatmap(ap_df, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=axes[1],
                cbar_kws={'label': 'AP'})
    axes[1].set_title(f'Figure 3b: Average Precision (τ={tau})', fontsize=14)
    axes[1].set_xlabel('Stablecoin')
    axes[1].set_ylabel('Horizon')
    
    plt.tight_layout()
    output_path = f'{OUTPUT_PATH}fig3_generalization_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Fig.3 saved: {output_path}')
    
    # 保存数据
    auc_df.to_csv(f'{OUTPUT_PATH}fig3_auc_matrix.csv')
    print(f'\nAUC Matrix (τ={tau}):')
    print(auc_df.round(4))
    
    return output_path


# ============================================================================
# Fig.4: Feature Analysis (误差分布 + 关键指标)
# ============================================================================
def plot_feature_analysis(all_data):
    """生成 Fig.4: 特征分析"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 合并所有数据
    all_df = pd.concat([df.assign(horizon=h) for h, df in all_data.items()], ignore_index=True)
    
    # 4.1 预测误差分布
    ax1 = axes[0, 0]
    error = all_df['prediction'] - all_df['y_soft']
    ax1.hist(error, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='r', linestyle='--', label='Zero Error')
    ax1.axvline(x=error.mean(), color='g', linestyle='-', label=f'Mean={error.mean():.3f}')
    ax1.set_xlabel('Prediction Error (ŷ - y)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Figure 4a: Prediction Error Distribution')
    ax1.legend()
    
    # 4.2 误差 vs 真实值
    ax2 = axes[0, 1]
    abs_error = np.abs(all_df['prediction'] - all_df['y_soft'])
    ax2.scatter(all_df['y_soft'], abs_error, alpha=0.1, s=1)
    ax2.set_xlabel('True Label (y_soft)')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Figure 4b: Error vs True Value')
    
    # 4.3 跨horizon RMSE
    ax3 = axes[1, 0]
    rmse_by_horizon = []
    for h in HORIZONS:
        if h in all_data:
            df = all_data[h]
            rmse = np.sqrt(np.mean((df['prediction'] - df['y_soft'])**2))
            rmse_by_horizon.append(rmse)
        else:
            rmse_by_horizon.append(np.nan)
    
    bars = ax3.bar(HORIZONS, rmse_by_horizon, color='steelblue', alpha=0.8)
    ax3.set_xlabel('Horizon')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Figure 4c: RMSE by Horizon')
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, rmse_by_horizon):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4.4 关键统计
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = "Summary Statistics:\n" + "="*40 + "\n\n"
    for h in HORIZONS[:4]:  # 展示前4个
        if h in all_data:
            df = all_data[h]
            metrics = compute_regression_metrics(df['y_soft'], df['prediction'])
            stats_text += f"{h}:\n"
            stats_text += f"  RMSE: {metrics['RMSE']:.4f}\n"
            stats_text += f"  MAE:  {metrics['MAE']:.4f}\n"
            stats_text += f"  R²:   {metrics['R2']:.4f}\n"
            stats_text += f"  Corr: {metrics['Pearson']:.4f}\n\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('Figure 4d: Summary Statistics')
    
    plt.tight_layout()
    output_path = f'{OUTPUT_PATH}fig4_feature_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Fig.4 saved: {output_path}')
    
    return output_path


# ============================================================================
# Fig.5: Regression Metrics Summary Table
# ============================================================================
def plot_metrics_table(all_data):
    """生成 Fig.5: 回归指标汇总表"""
    
    # 计算所有horizon的指标
    metrics_data = []
    for horizon in HORIZONS:
        if horizon not in all_data:
            continue
        df = all_data[horizon]
        metrics = compute_regression_metrics(df['y_soft'], df['prediction'])
        metrics['horizon'] = horizon
        metrics['n_samples'] = len(df)
        metrics_data.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df = metrics_df[['horizon', 'RMSE', 'MAE', 'R2', 'Pearson', 'Spearman', 'n_samples']]
    
    # 保存CSV
    csv_path = f'{OUTPUT_PATH}fig5_metrics_summary.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f'Fig.5 data saved: {csv_path}')
    print('\nRegression Metrics Summary:')
    print(metrics_df.round(4).to_string(index=False))
    
    # 绘制表格图片
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 格式化数据
    table_data = []
    headers = ['Horizon', 'RMSE', 'MAE', 'R²', 'Pearson', 'Spearman', 'Samples']
    for _, row in metrics_df.iterrows():
        table_data.append([
            row['horizon'],
            f"{row['RMSE']:.4f}",
            f"{row['MAE']:.4f}",
            f"{row['R2']:.4f}",
            f"{row['Pearson']:.4f}",
            f"{row['Spearman']:.4f}",
            f"{int(row['n_samples']):,}"
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # 设置表头样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 高亮最佳R²行
    best_r2_idx = metrics_df['R2'].idxmax()
    for i in range(1, len(table_data) + 1):
        if i - 1 == best_r2_idx:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#E2EFDA')
    
    ax.set_title('Figure 5: Regression Performance Metrics Summary', 
                fontsize=14, fontweight='bold', pad=20)
    
    output_path = f'{OUTPUT_PATH}fig5_metrics_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Fig.5 saved: {output_path}')
    
    return output_path, metrics_df


# ============================================================================
# Main
# ============================================================================
def main():
    print("="*60)
    print("Depeg Detection - Visualization Pipeline")
    print("="*60)
    
    # 加载数据
    print("\n[1/5] Loading predictions...")
    all_data = load_all_predictions()
    print(f"    Loaded {len(all_data)} horizons")
    for h, df in all_data.items():
        print(f"    - {h}: {len(df)} samples")
    
    # 生成图表
    print("\n[2/5] Generating Fig.1: ROC + PR Curves...")
    plot_roc_pr_curves(all_data)
    
    print("\n[3/5] Generating Fig.2: Tolerance Curve...")
    plot_tolerance_curves(all_data)
    
    print("\n[4/5] Generating Fig.3: Generalization Heatmap...")
    plot_generalization_heatmap(all_data)
    
    print("\n[5/5] Generating Fig.4 & Fig.5...")
    plot_feature_analysis(all_data)
    plot_metrics_table(all_data)
    
    print("\n" + "="*60)
    print("All figures saved to:", OUTPUT_PATH)
    print("="*60)


if __name__ == '__main__':
    main()
