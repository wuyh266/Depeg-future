# DepegDetector - 稳定币脱锚检测框架

基于多维数据指标的稳定资产实时脱锚检测框架。

## 项目概述

本项目实现了一个用于实时检测稳定币脱锚（Depeg）的机器学习框架，主要功能包括：

1. **AMM 信号提取** - 从 Uniswap 等 AMM 池中提取流动性失衡指数（LII）、香农熵、净流量等核心指标
2. **BOCD 变点检测** - 使用贝叶斯在线变点检测算法识别指标异常
3. **分层预警模型** - 融合多指标给出脱锚风险评分

## 环境配置

### 依赖要求

- Python 3.8+
- 虚拟环境：`.venv`（项目已配置）

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- pandas, numpy
- scikit-learn, xgboost
- ccxt（交易所数据获取）
- requests, yaml

## 项目结构

```
DepegDetector/
├── config.yaml              # 配置文件
├── src/
│   ├── main.py              # 核心主程序（数据获取/预测/监控）
│   ├── fetch.py             # TheGraph 数据获取
│   ├── fetch_price.py       # 价格数据获取
│   ├── fetch_ohlcv.py       # K线数据获取
│   ├── fetch_curve.py       # Curve 池数据获取
│   ├── entropyV2.py         # 熵值计算
│   ├── netflow.py           # 净流量计算
│   ├── markout.py           # 退出影响计算
│   ├── bocd.py              # 贝叶斯变点检测
│   ├── bocd_extended.py     # 扩展BOCD
│   ├── depeg_predictor.py   # 脱锚预测器
│   ├── ml_depeg_predictor.py # ML脱锚预测
│   ├── validate_model.py    # 模型验证
│   ├── feature_engineering.py # 特征工程
│   └── experiments/         # 实验代码
├── train_improved.py        # 模型训练
├── train_improved_v2.py     # 模型训练v2
├── evaluate_depeg.py        # 评估脚本
└── doc/                     # 文档资料
```

## 配置说明

`config.yaml` 主要配置项：

| 配置项 | 说明 |
|--------|------|
| `subgraph` | Uniswap V2 TheGraph 端点 |
| `subgraph_uniswap_v3` | Uniswap V3 TheGraph 端点 |
| `proxy` | 网络代理设置 |
| `stablecoins` | 监控的稳定币列表及时间范围 |
| `curve_pools` | Curve 稳定币池配置 |
| `markout_window` | 退出影响窗口（秒） |
| `bocd.hazard_rate` | BOCD 风险率参数 |
| `model` | 机器学习模型参数 |

## 使用方法

### 1. 数据获取

```bash
# 运行主程序获取数据（指定币种）
python src/main.py --coin USDC

# 获取所有币种数据
python src/main.py --all

# 获取 OHLCV 数据
python src/main.py --ohlcv --coin USDC

# 获取 Curve 池数据
python src/main.py --curve --coin USDC

# 实时监控模式
python src/main.py --monitor --coin USDC
```

### 2. 训练模型

```bash
# 改进版训练
python train_improved_v2.py

# 或使用交叉验证训练
python improved_cross_validation.py
```

### 3. 脱锚预测

```bash
# 使用训练好的模型进行预测
python src/main.py --coin USDC
```

### 4. 模型评估

```bash
# 评估脱锚检测性能
python evaluate_depeg.py

# 评估ML指标
python evaluate_ml_metrics.py

# 评估ML预测
python evaluate_ml_predictions.py
```

### 5. 运行实验

```bash
# RQ1: 指标测试
python src/experiments/rq1_indicator_test.py

# RQ2: 组合测试
python src/experiments/rq2_combination_test.py

# RQ3: 模型评估
python src/experiments/rq3_model_eval.py

# RQ4: 预警提前量
python src/experiments/rq4_early_warning.py
```

## 核心指标

| 指标 | 说明 |
|------|------|
| `entropy` | 香农熵 - 衡量流动性分布不确定性 |
| `netflow` | 净流量 - 代币流入流出差值 |
| `markout` | 退出影响 - 模拟代币退出的价格影响 |
| `volume` | 交易量 |

## 支持的稳定币

项目支持多种稳定币的脱锚检测：USDC, USDT, DAI, UST, FRAX, TUSD, sUSDE, USDe, xUSD, IRON, USDN, LUSD, FEI, sUSD, USDS, GUSD, BUSD, MIM 等。

## 评估指标

- AUROC / F1 / Precision / Recall
- 误报率（False Alarm Rate）
- 预警提前量

## 注意事项

1. 首次运行需配置有效的 TheGraph API Key
2. 如遇网络问题需配置代理（见 `config.yaml`）
3. 模型训练需要足够的历史数据

---

## 10. 各文件的示例命令与命令格式

### 10.1 数据获取

#### `src/main.py` — 核心主程序（数据获取/指标计算/模型预测）
整合数据获取、指标计算（熵/净流量/退出影响）、BOCD 变点检测的全流程管道。支持多币种、多版本 Uniswap（V2/V3/V4）及 Curve 池数据获取。
```bash
# 获取单个币种数据
python src/main.py --coin USDC

# 获取所有币种
python src/main.py --all

# 获取分钟级 OHLCV 数据
python src/main.py --ohlcv --coin USDC

# 获取 Curve 池数据
python src/main.py --curve --coin UST

# 实时监控模式
python src/main.py --monitor --curve --coin USDC

# 指定最近天数（增量更新）
python src/main.py --coin USDC --days 7
```
**核心函数**: `main()`, `process_single_coin()`, `compute_pool_bocd_features()`

#### `src/fetch.py` — TheGraph 数据获取
通过 GraphQL 从 Uniswap V2/V3/V4 subgraph 查询代币池列表（`get_token_pools`）、Swap 数据、池小时数据。支持分页查询和增量更新。**类**: `Fetcher`（全局数据获取）和 `PoolFetcher`（单池细粒度查询，含 Positions/Mints/Burns 事件）。
```bash
# 在代码中使用
from src.fetch import Fetcher, get_token_pools
pools = get_token_pools("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", api_key, min_tvl=10000)

# 或 as 脚本运行
python src/fetch.py
```
**核心函数**: `get_token_pools()`, `get_token_pools_at_time()`, `Fetcher.fetch_all_for_coin()`

#### `src/fetch_price.py` — 多源价格数据获取
从 CoinGecko（REST API）、Binance（K线 API）、Coinbase（蜡烛图 API）、TheGraph（TokenDayDatas）四路数据源获取稳定币历史价格。自动故障切换，支持增量更新。
```bash
# 获取所有币种价格
python src/fetch_price.py

# 在代码中使用
from src.fetch_price import PriceFetcher
fetcher = PriceFetcher()
df = fetcher.fetch_historical('USDC')
```
**核心函数**: `PriceFetcher.fetch_historical()`, `fetch_from_binance()`, `fetch_from_coingecko()`, `fetch_from_thegraph()`

#### `src/fetch_ohlcv.py` — 分钟级 K 线数据获取
通过 `ccxt` 库从 Bybit/Coinbase/Binance/OKX/KuCoin 等交易所获取 1 分钟级 OHLCV 数据。支持多交易所故障切换、增量更新（仅获取缺失区间）。
```bash
# 命令行获取
python src/fetch_ohlcv.py USDC 2023-02-01 2023-04-01

# 在代码中使用
from src.fetch_ohlcv import OHLCVFetcher
fetcher = OHLCVFetcher()
fetcher.fetch('USDC', '2023-02-01', '2023-04-01')
```
**核心函数**: `OHLCVFetcher.fetch()`, `_fetch_from_exchange()`

#### `src/fetch_curve.py` — Curve 池数据获取
从 Curve subgraph 获取池小时数据（virtualPrice, A 参数, 交易量）、流动性事件（Add/Remove Liquidity）、兑换事件。支持离线处理和 WebSocket 实时监控两种模式。**类**: `CurveFetcher`（批量获取）, `CurveWebsocketMonitor`（实时监控）。
```bash
# 获取单个 Curve 池数据
python src/fetch_curve.py --coin UST

# 获取所有 Curve 池
python src/fetch_curve.py --all

# 实时监控
python src/fetch_curve.py --coin UST --monitor
```
**核心函数**: `fetch_pool_hour_data()`, `fetch_liquidity_events()`, `compute_curve_features()`, `CurveWebsocketMonitor.start()`

#### `src/client.py` — GraphQL 客户端
封装 HTTP POST 请求到 TheGraph subgraph 的客户端。支持 API Key 认证、请求重试、分页查询（`paged_query`）、代理配置。
```bash
from src.client import GraphClient
client = GraphClient("https://...subgraph_url...", "api_key")
data = client.paged_query(graphql_template, "entity_name")
```
**核心类**: `GraphClient.__init__()`, `query()`, `paged_query()`

---

### 10.2 指标计算

#### `src/entropyV2.py` — 香农熵计算
计算 AMM 池流动性分布的香农熵。输入为两种代币的储备量（V2 的 `reserve0/reserve1`）或价格（V3 的 `token0Price/token1Price`），输出 0-1 之间的熵值（1 表示完全均匀分布）。
```bash
from src.entropyV2 import compute_entropy_v2
df = compute_entropy_v2(pool_hours_df)  # 添加 'entropy' 列
```
**核心函数**: `entropy_from_reserve(r)`, `compute_entropy_v2(df)`

#### `src/netflow.py` — 净流量计算
从 Swap 数据中计算每小时代币净流入/流出量（`amount0In - amount0Out`），按小时聚合。
```bash
from src.netflow import compute_netflow
netflow_df = compute_netflow(swaps_df, base="USDC")  # 返回 hour, flow 两列
```

#### `src/markout.py` — 退出影响计算
模拟代币退出的价格影响。支持两种模式：`compute_markout`（小时级，使用价格映射计算 PnL）和 `compute_markout_5min`（5 分钟级，含价格均值/标准差/偏度/峰度/VWAP 等详细统计）。
```bash
from src.markout import compute_markout, compute_markout_5min
markout_hourly = compute_markout(swaps, prices, h=3600)
markout_5min = compute_markout_5min(swaps, pool_hour_data, h=300)
```

---

### 10.3 变点检测

#### `src/bocd.py` — 贝叶斯在线变点检测（基础版）
实现 Adams & MacKay 2007 的 BOCD 算法。使用 `ConstantHazard` 风险函数 + `StudentTModel`（Normal-Gamma 共轭先验）作为预测模型。维护运行长度后验分布 `log_R`，输出变点概率。支持有效截断（`truncate_threshold`）。
```bash
from src.bocd import BOCD, ConstantHazard, StudentTModel
hazard = ConstantHazard(lam=200)
model = StudentTModel(mu0=0, kappa0=1, alpha0=1, beta0=1)
bocd = BOCD(hazard, model)
bocd.update(0.01)  # 输入新观测值
cp_prob = bocd.get_cp_probs()[-1]  # 当前变点概率
```

#### `src/bocd_extended.py` — BOCD 扩展版
在基础 BOCD 上增加偏差分数（`deviation` = z-score of prediction error）、惊异度（`surprise` = -log P(r=0)）、预测误差等实用特征。**类**: `ExtendedBOCD`（封装 BOCD，提供 `get_latest_features()` 接口）。被 `main.py` 和 `add_bocd_features.py` 调用。
```bash
from src.bocd_extended import ExtendedBOCD
bocd = ExtendedBOCD(hazard_rate=200)
for x in returns:
    bocd.update(x)
features = bocd.get_latest_features()  # {cp_prob, deviation, surprise, ...}
```
**核心函数**: `ExtendedBOCD.update()`, `get_latest_features()`, `get_features()`

---

### 10.4 脱锚预测模型

#### `src/depeg_predictor.py` — 基础脱锚预测器
基于波动率和偏离度的轻量级概率预测模型。从 OHLCV 数据计算滚动窗口特征（偏离度、波动率、偏离趋势），通过指数映射转换为 30/60/120 分钟不脱锚概率。输出 SAFE/CAUTION/WARNING/CRITICAL/DEPEG 五个状态。
```bash
# 运行模拟预测
python src/depeg_predictor.py --coin USDC

# 所有币种
python src/depeg_predictor.py --all
```
**核心类**: `DepegPredictor` — `predict()`, `run_simulation()`, `compute_risk_score()`

#### `src/ml_depeg_predictor.py` — XGBoost 脱锚预测器
机器学习版脱锚预测。提取滚动窗口特征（波动率、偏度、峰度、收益率等）+ BOCD 特征，使用 XGBoost 分类器，在 UST 脱锚事件上训练，预测 3h/6h/12h 内是否脱锚。支持 `TimeSeriesSplit` 交叉验证和 `StandardScaler` 标准化。
```bash
# 在代码中使用
from src.ml_depeg_predictor import MLDepegPredictor
predictor = MLDepegPredictor()
features = predictor.extract_features(data_dir='data/raw data')
labels = predictor.generate_labels(features)
predictor.train(features, labels)
```
**核心函数**: `MLDepegPredictor.extract_features()`, `generate_labels()`, `train()`, `predict()`, `evaluate()`

#### `src/lstm_depeg_predictor.py` — LSTM 脱锚预测器
使用 PyTorch 双向 LSTM（2 层，隐藏层 64）的秒级脱锚预测模型。自定义 `DepegDataset` 处理时间序列，`DepegLSTM` 定义网络结构。作为 XGBoost 的对比模型，使用相同特征集。
```bash
from src.lstm_depeg_predictor import LSTMDepegPredictor
predictor = LSTMDepegPredictor()
predictor.train(features, labels)
preds = predictor.predict(test_features)
```
**核心类**: `LSTMDepegPredictor`, `DepegLSTM`（nn.Module）

#### `src/dg_final.py` — 域泛化（DG）最终模型
论文核心模型。使用 5 维特征（价格、log 收益率、偏离度、滚动均值、滚动标准差）+ Transformer + Mixture of Experts（MoE）+ VREx 损失实现域不变脱锚预测。**关键创新**: 反坍缩正则化（Anti-collapse Regularization），防止所有专家退化为同一模式。在 USDC/DAI/UST 等训练，泛化到 EURC/sUSD/TUSD 等多币种。
```bash
# 训练和评估 DG 模型
python src/dg_final.py

# 或导入使用
from src.dg_final import build_features_v3, train_model, evaluate_model
```
**核心函数**: `build_features_v3()`, `train_model()`, `evaluate_model()`, `main()`

#### `src/train_predict.py` — 三层层次化脱锚预测器
层次化预测架构：第一层 BOCD 异常检测 → 第二层 AMM 池级 Random Forest → 第三层币种级全局 Random Forest。同时支持分类（是否脱锚）和回归（距脱锚时间）。
```bash
python src/train_predict.py
```
**核心类**: `HierarchicalDepegPredictor`

---

### 10.5 训练与评估

#### `train_improved.py` — 改进版训练（V1）
在 `ml_depeg_predictor.py` 基础上增加 SMOTE 过采样、XGBoost `scale_pos_weight` 调整、`TimeSeriesSplit` 交叉验证。解决正样本严重不平衡问题。
```bash
python train_improved.py
```

#### `train_improved_v2.py` — 改进版训练（V2）
针对正样本全部集中在数据开头（索引 0-10797）的问题：使用 `StratifiedKFold` 替代固定划分，在正样本富集的分折上训练，SMOTE + 类别权重。
```bash
python train_improved_v2.py
```

#### `improved_cross_validation.py` — 改进交叉验证
在正样本区域内进行时间序列划分，确保每个测试分折都包含正样本。主要评估指标为 AUC-PR（更适合不平衡数据）。
```bash
python improved_cross_validation.py
```

#### `evaluate_depeg.py` — 模型可靠性评估
评估 RandomForest 和 XGBoost 在 3h/6h/12h/24h 不同时间范围的表现。使用 `DepegModelEvaluator` 类进行时间序列交叉验证、混淆矩阵分析。
```bash
python evaluate_depeg.py
```
**核心类**: `DepegModelEvaluator`

#### `evaluate_ml_metrics.py` — ML 预测指标评估
从保存的 metrics CSV 加载特征和标签，计算 3h/6h/12h 的准确率、召回率、F1、AUC-ROC、AUC-PR。
```bash
python evaluate_ml_metrics.py
```

#### `evaluate_ml_predictions.py` — ML 预测结果评估
类似 `evaluate_ml_metrics.py`，但额外需要预测结果文件，在预测输出 vs 真实标签上计算指标。
```bash
python evaluate_ml_predictions.py
```

#### `evaluation_report.py` — 完整评估报告生成
聚合交叉验证、Bootstrap 交叉验证和最终模型评估结果，生成完整报告。
```bash
python evaluation_report.py
```

---

### 10.6 特征工程

#### `src/feature_engineering.py` — 特征工程中心
构建用于 ML 模型的多时间窗口（6/12/24/48/72 小时）滚动统计特征（波动率、偏度、峰度、偏离度、收益率），融合 BOCD 特征。按预测目标（6/24/168 分钟）组织特征集。**类**: `FeatureEngineer`。
```bash
from src.feature_engineering import FeatureEngineer
engineer = FeatureEngineer()
features = engineer.compute_features(metrics_df)
```

#### `src/depeg_label.py` — 脱锚事件标注
基于价格与锚定价格（默认 1.0）的绝对偏离度阈值（默认 5%）标注脱锚标签。支持最小持续时间过滤避免噪声误标。**类**: `DepegLabeler`。
```bash
from src.depeg_label import DepegLabeler
labeler = DepegLabeler(threshold_pct=5, min_duration_hours=6)
labeled_df = labeler.label_timeseries(price_df)
events = labeler.detect_events(price_df)
```
**核心函数**: `label_timeseries()`, `detect_events()`, `get_event_intervals()`

#### `src/generate_soft_labels.py` — 软标签生成
生成 0-1 连续软标签（软标签 = 未来窗口内价格偏离超过阈值的时间比例），支持 3h/6h/12h 预测窗口。
```bash
from src.generate_soft_labels import generate_soft_labels
df_soft = generate_soft_labels(df, horizon_minutes=360, threshold=0.05)
```

#### `src/add_bocd_features.py` — 添加 BOCD 特征
为现有特征 CSV 文件批量添加 BOCD 特征（对收益率、偏离度、交易量、价格极差运行 `ExtendedBOCD`，输出 deviation/cp_prob/surprise）。
```bash
python src/add_bocd_features.py
```
**核心函数**: `extract_bocd_features(df, hazard_rate)`

---

### 10.7 实验（Research Questions）

#### `src/experiments/rq1_indicator_test.py` — RQ1: 指标有效性测试
```bash
python src/experiments/rq1_indicator_test.py
```

#### `src/experiments/rq2_combination_test.py` — RQ2: 指标组合测试
```bash
python src/experiments/rq2_combination_test.py
```

#### `src/experiments/rq3_model_eval.py` — RQ3: 模型评估
```bash
python src/experiments/rq3_model_eval.py
```

#### `src/experiments/rq4_early_warning.py` — RQ4: 预警提前量
```bash
python src/experiments/rq4_early_warning.py
```

#### `src/fair_comparison.py` — 公平对比实验（Transformer）
固定 Transformer 架构，对比"全部特征 vs 仅偏离度 vs 无偏离度"三组特征的 AUC 差异。验证偏离度特征对模型性能的真实贡献。定义 `PriceTransformer`（含位置编码）。
```bash
python src/fair_comparison.py
```

#### `src/lstm_comparison.py` — 公平对比实验（LSTM）
固定 LSTM 架构的公平对比，与 `fair_comparison.py` 相同的特征分组逻辑。定义 `SimpleLSTM`。
```bash
python src/lstm_comparison.py
```

#### `src/killer_test.py` — 压力测试
验证高 AUC 是否源于数据泄露或捷径学习。运行对抗性/置换测试检验模型鲁棒性。使用 Transformer + 随机标签/置换特征等测试。
```bash
python src/killer_test.py
```

#### `src/ablation_experiment.py` — 消融实验
通过去掉偏离度特征运行 `dg_final.py`（不带 `--include-deviation` 参数），测量 AUC 下降幅度。
```bash
python src/ablation_experiment.py
```

#### `src/quick_ablution.py` — 快速消融（Logistic Regression）
轻量级逻辑回归消融基线，快速检验不同特征子集的效果（仅价格、仅偏离度、仅波动率、无偏离度、全部）。
```bash
python src/quick_ablution.py
```

---

### 10.8 模型分析与解释

#### `src/dg_interpretation_core.py` — DG 模型解释核心
对 DG 模型进行潜空间分析（PCA/t-SNE）、路由矩阵分析（MoE 专家激活模式）、特征归因（Ridge 逼近）。生成论文中的解释性图表。
```bash
python src/dg_interpretation_core.py
```

#### `src/analyze_dg_final_vs_experiments.py` — DG 对比其他实验
加载 DG Final 和其他实验（fair_comparison, lstm_comparison, quick_ablation, killer_test）的预测 CSV，生成对比指标表。
```bash
python src/analyze_dg_final_vs_experiments.py
```

#### `src/analyze_dg_routing_matrix.py` — MoE 路由矩阵分析
分析专家-币种路由矩阵，可视化各专家对不同币种的激活模式。使用 PCA 对潜空间降维。
```bash
python src/analyze_dg_routing_matrix.py
```

#### `src/recompute_results_from_predictions.py` — 从预测结果重算指标
从已保存的预测 CSV 文件重新计算评估指标，无需重训模型。
```bash
python src/recompute_results_from_predictions.py
```

#### `src/event_level_metrics.py` — 事件级指标计算
计算事件粒度（而非时间点粒度）的评估指标。加载多个实验源（killer_test, fair_comparison, lstm_comparison, quick_ablation）的预测结果。
```bash
python src/event_level_metrics.py
```

#### `src/compare_models.py` — 模型对比
XGBoost vs LSTM 横向对比。加载已保存模型，在测试集上评估，生成对比报告 CSV（AUC, AP, Precision, Recall, F1）。
```bash
python src/compare_models.py
```

#### `src/export_predictions.py` — 导出 DG 模型预测
为每个币种导出 DG Final 模型的预测结果（含多级标签：硬二值/软连续），保存为 CSV。
```bash
python src/export_predictions.py
```

---

### 10.9 工具与辅助

#### `src/align.py` — 时间戳对齐
将任意时间戳列对齐到小时边界：`ts // 3600 * 3600`。
```bash
from src.align import align_hour
df_aligned = align_hour(df, 'timestamp')
```

#### `src/preprocess.py` — 对数差分预处理
`logdiff(x) = log(1 + pct_change(x))`，用于价格/收益率特征变换。
```bash
from src.preprocess import logdiff
df['price_ld'] = logdiff(df['price'])
```

#### `src/tools.py` — 共享工具函数
CSV 文件备份保存（`save_csv_with_backup`）、时间戳解析（`parse_datetime_to_timestamp`）、格式转换等。
```bash
from src.tools import save_csv_with_backup, parse_datetime_to_timestamp
backup_path, saved_path = save_csv_with_backup(df, 'output.csv')
ts = parse_datetime_to_timestamp('2023-03-11')
```

#### `src/validate_model.py` — 模型可靠性验证
验证 ML 预测模型在已知历史脱锚事件（UST, USDC, DAI）上的可靠性。生成校准曲线、ROC/PR 曲线、不同阈值下的精确率/召回率。**类**: `ModelValidator`。
```bash
python src/validate_model.py
```

#### `src/monitor.py` — 实时监控系统
生产级实时脱锚监控系统。周期获取池指标，运行 XGBoost 预测模型，检查预警阈值。可触发买卖信号。持续循环运行。**类**: `DepegMonitor`。
```bash
python src/monitor.py
```

#### `src/visualization.py` — 论文级可视化
生成 4 大核心论文图表：ROC 曲线、PR 曲线、特征重要性柱状图、混淆矩阵。支持多币种、多时间跨度的聚合指标表。
```bash
python src/visualization.py
```
**核心函数**: `plot_roc_curves()`, `plot_precision_recall_curves()`, `plot_feature_importance()`

#### `src/draw_model_frameworks.py` — 模型架构图绘制
使用 matplotlib 绘制 LSTM、Transformer、DG-Transformer 三种模型架构的出版级框架图。
```bash
python src/draw_model_frameworks.py
```

#### `src/make_paper_figures.py` — 论文全图生成
生成论文中所有比较图表（柱状图、热力图、表格），对比三组模型族（Pure Transformer, LSTM, DG Final）在三组特征（Deviation Only, No Deviation, All Features）和多时间跨度的表现。输出中文标注的出版级图表。
```bash
python src/make_paper_figures.py
```
