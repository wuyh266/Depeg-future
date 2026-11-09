# DepegDetector Pipeline

基于多维数据指标的稳定资产实时脱锚检测框架 — **独立可运行管道**。

可独立复制到任意机器，安装依赖后直接运行，不依赖项目其他目录。

## 本仓库的本地配置与数据

首次使用时，先在 PowerShell 中复制配置示例：

```powershell
Copy-Item config.example.yaml config.yaml
```

然后编辑 `config.yaml`，填写自己的 API 密钥、RPC 地址和可选代理。`config.example.yaml` 中的密钥为空，Infura 项目 ID 使用占位符。

原始行情数据和运行结果保留在本地，本仓库不包含 `data/`。训练和评估前，请按下文的数据获取步骤生成数据，或将已有的数据文件放入 `data/`。本地的 `config.yaml`、环境变量文件、虚拟环境、模型权重和缓存均已加入忽略规则。

## 环境配置

```bash
pip install -r requirements.txt
```

Python 3.8+，推荐 3.10。

## 项目结构

```
pipeline/
├── config.yaml               # 配置文件（API key / 币种 / 模型参数）
├── requirements.txt          # Python 依赖
├── README.md                 # 本文件
├── README_original.md        # 原始项目 README（备份）
├── docs/
│   ├── paper_outline_aligned_utf8.md     # 论文目录（与代码一致）
│   ├── paper_outline_formal_utf8.md      # 论文目录（正式版）
│   └── 框架设计说明.txt                   # 框架设计文档
├── data/                     # 运行时生成的数据存放目录
└── src/
    ├── main.py               # [入口] 核心主程序：数据获取→指标计算→BOCD→预测
    ├── monitor.py             # [入口] 实时监控系统（持续循环运行）
    ├── dg_final.py            # [入口] DG Transformer 模型训练与评估
    ├── fair_comparison.py     # [入口] Transformer 公平对比实验
    ├── lstm_comparison.py     # [入口] LSTM 公平对比实验
    │
    ├── fetch.py               # TheGraph 数据获取（Uniswap V2/V3/V4）
    ├── fetch_price.py         # 多源价格数据（CoinGecko / Binance / Coinbase）
    ├── fetch_ohlcv.py         # 分钟级 K 线（ccxt + 多交易所）
    ├── fetch_curve.py         # Curve 池数据获取
    ├── client.py              # GraphQL 客户端（分页/重试/代理）
    │
    ├── entropyV2.py           # 香农熵计算
    ├── netflow.py             # 净流量计算
    ├── markout.py             # 退出影响计算
    ├── bocd_extended.py       # BOCD 变点检测（偏差分数/惊异度）
    │
    ├── depeg_label.py         # 脱锚事件标注
    ├── feature_engineering.py # 多时间窗口特征工程
    ├── experiment_utils.py    # 共享常量与评估工具
    ├── align.py               # 时间戳对齐
    ├── preprocess.py          # 对数差分预处理
    ├── tools.py               # CSV 备份 / 时间解析等工具函数
    │
    ├── make_paper_figures.py  # [分析] 论文全图与对比表格生成
    ├── visualization.py       # [分析] ROC / PR / 特征重要性 / 混淆矩阵
    ├── draw_model_frameworks.py # [分析] 模型架构图绘制
    ├── dg_interpretation_core.py # [分析] DG 潜空间 / 路由矩阵分析
    ├── analyze_dg_final_vs_experiments.py # [分析] DG vs 基线对比
    ├── event_level_metrics.py # [分析] 事件级指标计算
    ├── export_predictions.py  # [分析] 导出 DG 模型预测结果
    └── __init__.py            # 包初始化（自动添加 src/ 到 sys.path）
```

## 配置说明

编辑 `config.yaml`：

| 配置项 | 说明 |
|--------|------|
| `THE_GRAPH_API_KEY` / `thegraph.api_key` | TheGraph API Key |
| `proxy.http` / `proxy.https` | HTTP 代理（如需） |
| `stablecoins` | 监控的稳定币列表及时间范围 |
| `curve_pools` | Curve 稳定币池配置 |
| `bocd.hazard_rate` | BOCD 风险率参数（默认 200） |
| `model` | 机器学习模型参数 |

## 快速开始

### 1. 数据获取 → 指标计算

```bash
# 获取 USDC 所有数据（熵/净流量/退出影响/价格/BOCD）
python src/main.py --coin USDC

# 获取所有币种
python src/main.py --all

# 仅获取 OHLCV 数据
python src/main.py --ohlcv --coin USDC

# 获取 Curve 池数据
python src/main.py --curve --coin UST

# 增量更新最近 N 天
python src/main.py --coin USDC --days 7
```

### 2. 训练与评估（3 个实验入口）

```bash
# DG Transformer 模型
python src/dg_final.py

# Transformer 公平对比（全部特征 / 仅偏离度 / 无偏离度）
python src/fair_comparison.py

# LSTM 公平对比
python src/lstm_comparison.py
```

三个实验都会在 `data/` 下生成预测结果 CSV。

### 3. 论文图表生成

```bash
# 生成全部论文图表与对比表格
python src/make_paper_figures.py

# ROC / PR 曲线
python src/visualization.py

# 模型架构图
python src/draw_model_frameworks.py

# DG 分析
python src/dg_interpretation_core.py
python src/analyze_dg_final_vs_experiments.py

# 导出预测结果
python src/export_predictions.py
```

### 4. 实时监控

```bash
python src/monitor.py
```

## 支持的稳定币

USDC, USDT, DAI, UST, FRAX, TUSD, sUSDE, USDe, xUSD, IRON, USDN, LUSD, FEI, sUSD, USDS, GUSD, BUSD, MIM, EURC

## 注意事项

1. 首次运行需在 `config.yaml` 中配置有效的 TheGraph API Key（免费注册 https://thegraph.com/studio/）
2. 如需代理请配置 `config.yaml` 中的 `proxy` 项
3. 三个实验入口（`dg_final.py`, `fair_comparison.py`, `lstm_comparison.py`）需要先通过 `main.py` 获取特征数据
4. 原项目完整版（含训练/验证/消融等脚本）见 `README_original.md`
