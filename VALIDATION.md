# DG 推理入口验证记录

验证日期：2026-11-11。环境：Python 3.11.16、PyTorch 2.5.1+cpu，完整依赖版本见 requirements-inference-tested.txt。

## 已通过

- Python unittest：8 项通过。覆盖训练代码与离线推理数值一致性、时间边界、181 根默认输入要求、缺文件、缺分钟、逆序、重复行、错误时间戳单位、错误来源、未收盘、无效滚动特征、manifest/权重参数不一致、其他特征配置与 6–9h 窗口、独立目录运行、失败覆盖旧输出。
- Go：`go test ./internal/adapter/pipeline/python ./cmd/check-prediction` 通过。覆盖真实零值、结构化错误、缺失/null 比值、越界、来源、窗口、时间和质量校验。
- 打包命令：成功从测试 checkpoint 及配套 scaler 生成独立交付目录，并生成样例参考输出。
- 独立运行：在 `data/inference-smoke/standalone-delivery/` 中执行其复制的 predict.py，成功生成 prediction.json；其 predictions 与 reference_output.json 完全一致，generated_at 单独生成。
- 跨语言读取：Go check-prediction 成功读取上述 Python 输出，并保留 integration_only。

## 验证边界

所有模型运行测试均使用合成输入和未训练随机测试权重，版本为 UNTRAINED-SMOKE-ONLY。测试模型 d_model=16、num_env=2，以减小测试资源开销；实现仍直接复用训练模型类，正式维度由真实 checkpoint 决定。

真实训练权重、其各自配套 scaler、实际支持窗口与训练数据口径尚未提供，因此没有验证真实预测质量或 OKX 数据与训练数据的相容性，也没有启用网站风险服务。本次没有训练模型、改动历史训练/导出脚本、写入数据库或进行哈希值校验。
