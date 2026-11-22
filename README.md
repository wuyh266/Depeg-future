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

2026-11-21：新增可选本地 gRPC 服务 `src/grpc_server.py`，依赖见 `requirements-grpc.txt`。它复用离线推理、常驻加载模型；协议与 Go 调用方法见相邻 Depeg 仓库的 `docs/grpc-inference.md`。原有 CSV/JSON 入口继续可用，真实权重仍需交付。

DG Transformer 独立推理、交付打包和 Go 联调见 [INFERENCE.md](INFERENCE.md)。推理以 `src/dg_final.py` 为准；旧 `export_predictions.py` 不作为部署入口。真实权重和配套 scaler 仍需模型方交付。

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
    ├── collect_ohlcv.py       # OKX USDC/USDT 的独立分钟采集入口
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

#### 单独试跑 OKX 分钟采集

在仓库目录打开 PowerShell。若尚未安装项目依赖，试跑只需安装 CCXT：

```powershell
python -m pip install ccxt==4.5.39
python src/collect_ohlcv.py --once
```

这会抓取最近 10 个已结束的分钟，把 OKX 确认收盘的数据保存到 `data/collector-preview/`，不需要 API 密钥、`config.yaml` 或数据库。如果要补抓指定时间段，或使用本机 HTTP 代理：

```powershell
python src/collect_ohlcv.py --once --start 2026-11-07T08:35:00Z --end 2026-11-07T08:45:00Z
python src/collect_ohlcv.py --once --proxy http://127.0.0.1:7890
```

时间区间包含起点、不包含终点，`Z` 表示 UTC。价格按 USDT 报价，成交量以 USDC 为单位。缺少的分钟会记在日志里；采集输出与原有训练 CSV 分开放置。

接入 MySQL 时，`--import-bin` 接收新版 Go 导入工具的可执行文件路径。给 `--loop` 指定该路径后，采集器先补写 `--pending-dir` 中的批次，再根据数据库记录继续采集。数据库没有这条行情时，需用 `--bootstrap-from` 指定首次采集起点。指定 `--once --start ... --end ... --import-bin ...` 可走同一待处理目录补录历史。

Go 工具需支持 `-latest-open-time -asset-id ccxt.usdc-usdt -source ccxt:okx -interval 1m`，标准输出为最后一根 K 线的 RFC3339 开盘时间；没有记录时成功退出且标准输出为空。写入时需支持 `-anchor=` 表示不提供锚定价，`-source ccxt` 与 CSV 的 `exchange=okx` 组合为 `ccxt:okx`。已有旧版工具不能用于自动入库。

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
