# 个人坐姿分类与姿势提醒

《神经网络与深度学习》课程第 05 题，截止 **2026 年 12 月 20 日**。完整安排见 [项目计划](坐姿分类项目计划.md)。

使用冻结的 **YOLO11n-pose** 提取人体关键点，通过 **PyTorch MLP** 识别端正、低头、上身前倾和侧倾四类坐姿，再结合概率平滑与持续时间规则提供可关闭的姿势提醒。

目前已实现数据清单校验、分组划分、关键点归一化、规则基线、MLP 训练与评估、时序提醒和视频演示入口。公开数据审计、真实 YOLO MPS 推理和特征缓存已跑通，见 [首阶段验证报告](docs/首阶段验证报告.md)。**四类人工标签仍待复核，正式训练、分类指标和摄像头实测尚未完成。** 模拟数据只用于程序检查。

## 获取与安装

以下命令从克隆仓库开始执行，需要先安装 [uv](https://docs.astral.sh/uv/)。固定环境名为 **`sitting-posture-py312`**，目录 `.venv-sitting-posture/`：

```bash
git clone https://github.com/Yukikazct/sitting-posture-classification.git
cd sitting-posture-classification
uv venv --python 3.12.2 --prompt sitting-posture-py312 .venv-sitting-posture
uv pip sync --python .venv-sitting-posture/bin/python requirements-macos-py312.lock.txt
source .venv-sitting-posture/bin/activate
python -m posture check
python -m unittest discover -s tests -v
```

依赖快照来自 macOS Apple Silicon / Python 3.12.2，其他平台的安装与运行尚未验证。后续安装、测试、数据处理和推理统一使用该环境。

## 公开仓库内容

仓库包含坐姿分类代码、配置、测试、项目计划、数据清单模板和审计文档。数据集、人物图片及视频、模型权重、虚拟环境、完整运行产物和课程原始文件不随代码上传。文档中指向 `runs/stage1/`、`data/raw/`、`data/processed/` 及模型文件的历史证据属于本地产物，克隆仓库不会包含这些文件；正式复现需要按来源说明准备数据并重新生成。

YOLO 权重及数据按各自来源许可使用，详见 [模型说明](docs/相关工作与模型说明.md) 和 [数据可用性审计](docs/数据可用性审计.md)。公开代码不替代数据、图片或第三方模型的授权。

## 目录

```text
configs/default.json       四类标签、姿态模型、训练与提醒参数
data/manifest.csv          人工标注样本清单模板（当前仅表头）
data/events.csv            连续视频事件标注模板（当前仅表头）
data/raw/                  原始公开数据和获同意的自采素材
data/processed/            特征缓存和提取元数据
models/                    本地 YOLO11n-pose 权重
posture/                   数据、特征、训练、评估、提醒与演示代码
tests/                     分组泄漏、归一化、拒识与时间状态测试
docs/                      标注规范、实验记录模板、首阶段任务
reports/                   英文 IEEE 论文说明
runs/                      每次运行的独立产物目录
```

## 环境准备

本项目固定使用 **`sitting-posture-py312`** 环境，目录为 `.venv-sitting-posture/`，已配置 Python 3.12.2。下面命令均从项目根目录运行：

```bash
source .venv-sitting-posture/bin/activate
python -m posture check
```

后续安装、测试和推理统一使用这个环境。未激活时可直接用 `.venv-sitting-posture/bin/python` 替代示例中的 `python3`；激活后 `python` / `python3` 均指向该环境。

需要重建时使用依赖快照：

```bash
uv venv --python 3.12.2 --prompt sitting-posture-py312 .venv-sitting-posture
uv pip sync --python .venv-sitting-posture/bin/python requirements-macos-py312.lock.txt
```

常规功能依赖声明在 `pyproject.toml`，可选视觉和绘图依赖已安装。VS Code 默认解释器也指向该项目环境。

`auto` 设备顺序为 CUDA → MPS → CPU。在 Apple Silicon Mac 上可显式指定 `--device mps`。`check` 仅报告当前环境、可用设备和本地权重状态。

## 先验证框架

```bash
python3 -m unittest discover -s tests -v
python3 -m posture smoke --output runs/smoke-cpu --device cpu
python3 -m posture smoke --output runs/smoke-mps --device mps
```

`smoke` 生成明确标记的随机特征，训练 2 个 Epoch，检查验证集拒识、权重保存、测试评估和提醒计时，并生成 `SMOKE_ONLY.json`。它不下载模型、不采集摄像头、不使用真实数据。MPS 命令仅适用于 `check` 确认可用的设备。

每次运行请指定新的输出目录，避免覆盖已有实验。失败后的目录保留用于诊断；修复后使用新目录重试。

## 数据与特征流程

先按 [标注规范](docs/标注规范.md) 完成真实标签和来源记录，再将样本填入 `data/manifest.csv`。

固定标签及编号：`0 upright`（端正）、`1 head_down`（低头）、`2 forward_lean`（上身前倾）、`3 side_lean`（侧倾）。未知或复合动作不作为第五类训练标签。

清单字段：

```text
sample_id,image_path,label,split,subject_id,video_id,group_id,view,source,timestamp_sec
```

- `image_path` 相对于清单文件所在目录；例如清单在 `data/` 时，图片路径可写 `raw/public/image001.jpg`。
- `split` 为 `train`、`val`、`test`，未划分时留空。已有官方划分应先审计再保留。
- 人员、视频和拍摄组字段用于阻止已知数据泄漏；同一原图裁剪的不同样本也必须共享 `group_id`。
- 没有身份信息时保留空值并记录限制，不能编造人员 ID。路径检查不能替代视觉去重。

```bash
python3 -m posture audit --manifest data/manifest.csv
python3 -m posture split --manifest data/manifest.csv --output data/processed/manifest_split.csv --seed 42
python3 -m posture audit --manifest data/processed/manifest_split.csv --require-split
```

`split` 会把共享人员、视频、拍摄组或图片的样本合并为连通组后整体划分，比例近似 70%/15%/15%；需至少 3 个独立组。类别不平衡和缺失身份会在结果中披露；不会为精确比例拆散同一组，也不会静默重新划分已有 split 的清单。

从官方渠道准备 `models/yolo11n-pose.pt` 后运行：

```bash
python3 -m posture extract --config configs/default.json --manifest data/processed/manifest_split.csv --output data/processed/features.npz --device mps
```

预训练提取器使用固定权重，不参与 MLP 训练。框架要求本地权重已存在。每张图像需只有一个可用人物；无人、多人、必需关键点不足等情况保留为拒识行。多人公开图片应先经人工确认目标人物，生成单人裁剪并保留原图分组信息。

缓存为无 pickle 的 NPZ：`features(N,25)`、`labels(N)`、`splits(N)`、`sample_ids(N)`、`views(N)`。25 维由鼻、双耳、双肩、双髋的归一化 x/y/置信度及 4 个几何特征组成；缺耳可用掩码表示，鼻、肩和髋必须可信。缓存旁的 `.metadata.json` 保存来源摘要、权重摘要、特征定义和拒识原因。

## 训练与评估

```bash
python3 -m posture train --config configs/default.json --cache data/processed/features.npz --output runs/mlp-seed42 --seed 42 --device mps
python3 -m posture evaluate --cache data/processed/features.npz --checkpoint runs/mlp-seed42/checkpoint.pt --output runs/eval-mlp-seed42 --split test --device mps
python3 -m posture evaluate --config configs/default.json --cache data/processed/features.npz --method rules --output runs/eval-rules --split test
python3 -m posture plot --history runs/mlp-seed42/history.csv --output runs/mlp-seed42/curves.png
python3 -m posture profile --checkpoint runs/mlp-seed42/checkpoint.pt --device mps
```

MLP 为 `25 → 64 → 32 → 4`，默认 Adam、学习率 0.001、Batch Size 64、最多 100 个 Epoch、验证集 Macro-F1 早停 patience 10。标准化仅拟合有效训练样本；提取失败的训练行不用于梯度更新，验证和测试行则保留并按拒识计入指标。

训练保存完整配置、权重与标准化参数、`history.csv`、`summary.json` 和缓存摘要，不读取测试集表现来选择模型。正式实验另用种子 43、44 重复，并汇总均值与标准差。规则阈值只能用验证集校准，默认阈值只是启发式起点。

评估输出 `metrics.json`、`predictions.csv`、`confusion_matrix.csv`，包含每视角统计。静态分类使用 argmax，提取失败计为 unknown；提醒阶段另使用分类置信度阈值和时序逻辑。混淆矩阵有四行五列，最后一列为拒识。

`profile` 统计分类器参数、线性层 MACs/FLOPs 和 batch=1 前向延迟，1 MAC 按 2 FLOPs 计算，不包括 bias、激活、关键点提取、标准化与图像 I/O。完整姿态模型复杂度和端到端 FPS 仍须在真实模型运行后补测，不能用分类器耗时代替。

## 视频与摄像头演示

```bash
python3 -m posture demo --config configs/default.json --checkpoint runs/mlp-seed42/checkpoint.pt --source 0 --device mps
python3 -m posture demo --checkpoint runs/mlp-seed42/checkpoint.pt --source data/raw/test.mp4 --save runs/demo-smoothed.mp4 --device mps
python3 -m posture demo --checkpoint runs/mlp-seed42/checkpoint.pt --source data/raw/test.mp4 --save runs/demo-raw.mp4 --raw --device mps
```

窗口中按 `q` 退出，`r` 切换提醒。OpenCV 叠加文字使用英文标签。默认 EMA 系数 0.3、分类置信度 0.6；异常持续 3 秒提醒，正常持续 1 秒解除，同一连续异常每隔至少 10 秒可再次提醒。无可靠观测时重置提醒状态。

视频计时使用帧号/FPS，摄像头使用单调时钟；提醒持续时长不按处理帧数计算。源视频需有可靠固定帧率。可在 JSON 配置增加 `"demo": {"display": false}`，对已有视频关闭 GUI。

这两个演示入口支持 raw/EMA 对照，但当前只输出运行摘要和可选叠加视频；按视频事件统计误提醒/小时、事件召回率和提醒延迟的离线工具仍待后续实现。事件真值模板见 `data/events.csv`。

## 下一步

按 [首阶段任务](docs/首阶段任务.md) 完成公开数据可用性审计、人工标签试标、官方模型真实单次推理和资源测试，再开始正式训练。使用 [实验记录模板](docs/实验记录模板.md) 记录真实结果。当前未产生正式坐姿实验指标或可用于交付的训练权重。
