# DG Transformer 推理交付与验收

本入口以 `src/dg_final.py` 的 `DGModelFinal`、`build_features_v3` 和 `predict_batch` 为唯一实现，不使用旧 `export_predictions.py`，不计算未来真实标签，不训练、不联网拉行情。

## 当前交付状态

- 已实现：独立离线入口、单窗口交付包打包、标准化参数 JSON 导出、错误 JSON、多窗口合并、Go 读取校验。
- 未交付：真实训练权重、与每份权重严格配套的训练 scaler、训练数据口径核对结果、真实样例参考预测。当前仓库没有这些文件，不能据此宣布模型上线。
- `data/inference-smoke/` 如存在，全部是合成行情和未训练随机权重，只验证软件接口。其版本固定为 `UNTRAINED-SMOKE-ONLY`，质量为 `integration_only`，不代表模型效果。

## 输入约定

首批限定 `asset_id=ccxt.usdc-usdt`、`symbol=USDC/USDT`、`source=ccxt:okx`、现货 `1m`。不得将任意其他交易对 CSV 改名后传入；当前采集 CSV 没有 symbol 列，调用者必须保证它来自该采集器。

CSV 必须包含 `timestamp,datetime,open,high,low,close,volume,exchange`，可以带其他列。

| 字段 | 约定 |
| --- | --- |
| timestamp | 整数 Unix 秒，对应分钟开盘时刻；不是毫秒 |
| datetime | 带明确时区、整分钟的 ISO 8601 时间；与 timestamp 一致，输出统一 UTC |
| open/high/low/close | 正有限数，USDT/USDC 报价，必须满足 OHLC 大小关系 |
| volume | 非负有限数，以 USDC 计；即使某权重不使用成交量，输入仍保留采集器字段 |
| exchange | 固定 okx |

输入必须逐分钟升序、无缺口、无重复且已经收盘。入口检查时间已过收盘点，但无法凭 CSV 独立证明交易所 confirm 标志；请使用 `collect_ohlcv.py` 的已确认收盘输出。入口不补缺口、不重排、不用零替代无效特征。全零成交量窗口导致 volume_ratio 无效时明确报错。

当前标签直接比较 `abs(close - 1.0) >= 0.05`，因此对这批输入表示 USDC 相对 USDT 的偏离。它不能自动解释成相对美元的绝对脱锚。正式交付前，模型方需核对训练交易所、价格方向、报价币、成交量单位、分钟频率和清洗方法，并在 manifest 中填写 `data_compatibility_note`。未核对时 quality 必须保留 `integration_only`。

## 序列长度和时间对齐

window 固定为训练函数的默认值 10；最长完整滚动特征需要 60 个 log_return，首个完整特征位于原始第 61 根 K 线（零基索引 60）。

训练使用 `X[i-seq_len:i]` 对应 `y[i]`。本入口原样保留：参考分钟为 t，模型最后使用 t−1 分钟的特征，未来标签从 t+1 分钟开始。参考分钟 K 线必须已经收盘，但不进入模型序列。

因此连续原始 K 线最少数量 = **60 根预热 + seq_len 根特征输入 + 1 根参考分钟**。默认 seq_len=120 时为 **181 根**；权重实际 seq_len=60 时为 121 根。数量由 checkpoint 导出到 manifest，入口不猜测或写死 120。

例：参考 K 线开盘为 03:00，收盘截止 `observed_at=03:01`，模型最后输入开盘为 02:59。0–3h 预测覆盖开盘分钟 03:01 至 06:00，时间区间为 `[03:01,06:01)`。保留这一间隔是为了与当前训练一致；如未来调整训练对齐方式，应发布新的 alignment/模型版本并重新验收，不能只改推理端。

## 窗口与权重

一个交付目录对应一份 checkpoint 和一个窗口，多个窗口通过重复 `--model-dir` 合并。打包工具从 checkpoint 读取 horizon 并与训练的 HORIZON_MAP 核对，不把所有可选窗口宣称成已有权重。

| 训练 horizon | 相对参考分钟 t 的标签开盘偏移（含两端） | 相对 observed_at 的预测时间区间 |
| --- | --- | --- |
| 0-3h | 1…180 | [0,180) 分钟 |
| 3-6h | 181…360 | [180,360) 分钟 |
| 6-9h | 361…540 | [360,540) 分钟 |
| 6-12h | 361…720 | [360,720) 分钟 |
| 12-18h | 721…1080 | [720,1080) 分钟 |
| 18-24h | 1081…1440 | [1080,1440) 分钟 |
| 24-30h | 1441…1800 | [1440,1800) 分钟 |
| 48-54h | 2881…3240 | [2880,3240) 分钟 |

最终可用窗口表由实际交付目录中的 manifest 构成。每次训练当前会把 scaler 写入相同文件名；交付方必须保留每次训练自己的两个 scaler，不能拿后一次训练覆盖后的 scaler 搭配旧权重。打包程序能检查结构和维度，无法证明同维度 scaler 的训练来源。

## 模型方打包

使用 Python 3.11，在项目中安装 `requirements-inference.txt`。CPU 环境可先从 PyTorch 官方 CPU 索引安装 `torch==2.5.1`，再安装该文件。已验证环境的完整版本记录见 `requirements-inference-tested.txt`。

下面参数中的文件必须由模型方提供；不自动下载或生成真实训练权重。只读取可信交付方提供的 pkl，打包后推理使用 JSON 标准化参数，不再加载 pkl。

```powershell
python src/package_dg.py --checkpoint trained/dg_final_0-3h_d128_k5.pt --scaler trained/dg_final_scaler.pkl --stats-scaler trained/dg_final_stats_scaler.pkl --model-version dg-release-001 --output-dir delivery/0-3h --sample-input real_sample.csv
```

默认 include_deviation/include_volume 均为 true。对应训练关闭时显式加入 `--no-include-deviation` 或 `--no-include-volume`。必须使用实际训练配置。

确认训练与 OKX 数据口径相容后，加入 `--verified --data-compatibility-note "实际核对依据"`。这个标识由模型交付方声明，打包工具不会代替科学验证。

交付目录必须为空。程序会导出：`model.pt`、`manifest.json`、`preprocessing.json`、`predict.py`、`dg_final.py`、依赖文件和 README；提供 sample-input 时还会复制 `sample_input.csv` 并实际计算 `reference_output.json`。因此复制该目录到另一台电脑即可安装依赖和离线运行。

## 一条命令推理

在交付目录安装依赖后：

```powershell
python predict.py --input sample_input.csv --model-dir . --output prediction.json
```

在仓库中可运行多个已交付窗口：

```powershell
python src/predict.py --input real_sample.csv --model-dir delivery/0-3h --model-dir delivery/3-6h --output prediction.json
```

默认 CPU；显式 `--device cuda` 在无 CUDA 时返回错误。stdout 只输出一个 JSON，成功退出码 0，业务失败退出码 2。`--output` 同步保存成功或失败 JSON，避免旧成功文件被误用；写入失败通过 stdout 返回 OUTPUT_WRITE_FAILED。参数拼写等 argparse 使用错误退出码 2 并输出使用说明。

## 输出约定 dg-inference/v1

成功信封包含 `schema_version`、`ok=true`、`generated_at`、`predictions` 数组。每条预测包含：

| 字段 | 语义 |
| --- | --- |
| asset_id / symbol / source | 资产、交易对、来源 |
| observed_at | 最后一根已收盘参考 K 线的收盘边界，UTC |
| reference_open_time / last_input_open_time | 参考 K 线及实际最后模型输入 K 线的开盘时间 |
| horizon | checkpoint 中的窗口名 |
| horizon_start_minutes / horizon_end_minutes | 相对 observed_at 的半开区间偏移 |
| prediction_start / prediction_end | 预测区间 [start,end)，UTC |
| depeg_minute_ratio | [0,1] 的显著脱锚分钟占比估计；0.2 显示“预测脱锚分钟占比 20%” |
| model_version / feature_version / label_version | 可追溯的模型、特征和标签版本 |
| quality | integration_only 或 verified；前者只能用于联调 |

该输出不表示“至少发生一次脱锚的概率”，也不表示价格预测。缺失/失败不能填为 0；0 是合法预测值。没有实际参考输出时，不编造模型结果。

错误示例：

```json
{"schema_version":"dg-inference/v1","ok":false,"error":{"code":"INSUFFICIENT_DATA","message":"至少需要 181 根连续已收盘 K 线，收到 10 根"}}
```

错误码包括 INPUT_NOT_FOUND、INVALID_INPUT、NON_CONTIGUOUS_DATA、UNCLOSED_CANDLE、INSUFFICIENT_DATA、MODEL_FILE_MISSING、INVALID_MANIFEST、MODEL_INCOMPATIBLE、INVALID_FEATURES、DEPENDENCY_MISSING、DEVICE_UNAVAILABLE、INVALID_PREDICTION、INFERENCE_FAILED、OUTPUT_WRITE_FAILED。Go 可按 code 提示原因，不将错误转为零值。

## 验证与 Go 联调

```powershell
python -m unittest discover -s tests -p test_predict.py -v
python tests/make_smoke_bundle.py
python src/predict.py --input data/inference-smoke/sample_input.csv --model-dir data/inference-smoke/bundle --output data/inference-smoke/prediction.json
```

上述 smoke 步骤全部使用合成输入及随机未训练权重，参考值仅检查代码。CPU 同环境参考预测建议绝对误差 ≤1e-6；generated_at 不参与数值一致性比较。

在相邻 Depeg 目录读取输出：

```powershell
go run ./cmd/check-prediction -input ../Depeg-future/data/inference-smoke/prediction.json
```

Go 检查 schema、窗口、比值、时间、来源和质量，保留真实零值及 integration_only 标识。该命令只读文件，不写数据库。网站风险服务仍保持未上线状态；真实模型完成验收后，才能继续接入持久化、查询及网页展示。
