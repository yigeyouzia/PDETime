# PDETime：保留原模型语义的 TSLib 风格训练入口与独立脚本

> 更新时间：2026-10-21；审计起点：`b7d74ec978362a2821679f5d2ed370df87dc7347`。
> 来源：[PDETime arXiv:2402.16913v1](https://arxiv.org/abs/2402.16913v1)、原 `train.py`、`src/PDETime/`、前两版 runner，以及用户本地 Time-Series-Library 的参数/脚本格式。
> 状态：原模型一致性检查、独立脚本检查和真实 ETTm1 CPU 冒烟通过；未完成 GPU 全量训练或论文主表复现。

## 1. 本轮范围与版本选择

按用户要求，**只增加参数化训练入口和独立 shell，不按 TSLib 重构模型、层或 exp/data_provider 目录。**

- 新入口：根目录 `train_reproduce.py`。
- 直接 import 原 `PDETime`、`PDETimeLoss` 和 `PDETimeDataset`。
- `src/PDETime/models/`、`src/PDETime/preprocess/` 和原 `train.py` 本轮没有改动。
- 新建七份数据集/模型 `.sh`，各自显式列出四种 H 的完整命令，没有数据集总控循环。
- 原 A5000 runner、上一版推导模型 runner 及结果保留，避免混淆历史。它们不是本轮“保留源码训练语义”的运行入口。

**论文与原代码有实质冲突。本轮保留原代码的模型/loss/初值/参数覆盖，已明确的实验参数可按论文配置；不将代码冲突自动改写成个人推导。** 因此可以称“源码语义保留的复现入口”，不能称已经严格 1:1 复现论文方法或准确率。

原项目来源为 [rohithm8/PDETime](https://github.com/rohithm8/PDETime)，用户 fork 为 [yigeyouzia/PDETime](https://github.com/yigeyouzia/PDETime)。论文作者官方身份仍未核实；PDF 文件名或页眉模板不能单独证明 ICML 录用。

## 2. 审计：原代码、前两版与本轮的区别

| 项目 | 原 train.py + src | reproduce_a5000.py | 上一版 reproduce_paper.py | 本轮 train_reproduce.py |
|---|---|---|---|---|
| 模型类 | 原 PDETime | 原 PDETime | 新增 PaperPDETime | **原 PDETime，未包装重写 forward** |
| 聚合层注册 | Python list，未纳入 model.parameters/state_dict | 转为 ModuleList | 转为 ModuleList | 保留 Python list，优化器覆盖不变 |
| N>1 聚合层 | 重复引用同一模块 | 仍共享，但默认 N1 | 默认独立初始化 | 保留共享；N 仍可由 CLI 配置 |
| Solver MLP | 代码中间 Linear 存在共享 | 沿用 | 可选共享/独立 | 沿用原 Solver |
| ridge 目标 | 历史原始值 X | 同原代码 | X 减历史真实末值 | 同原代码 |
| 未来解码偏移 | 最后历史重建输出 | 同原代码 | 历史真实末值 | 同原代码 |
| loss | 原 Lr + Lp + Lf；没有 Lc | 同原代码 | 新实现 Lp + Lc + Lf | **直接使用原 PDETimeLoss** |
| 时间特征 | loader 自动选择；完整 ETTm1 为 7 项 | 同原代码 | 固定论文 5/4 项 | paper 协议 5/4 项；trainpy 协议自动选择 |
| ETT 切分 | 7:1:2 | 7:1:2 | 6:2:2 | paper 协议 6:2:2；trainpy 协议 7:1:2 |
| val/test 历史上下文 | 各子集孤立建窗 | 同原代码 | 默认携带切分前历史 | **保留孤立建窗** |
| 目标数值尺度 | 原始值 | 原始值 | 默认原始值，可 standard | 原始值，不额外加入 scaler |
| checkpoint | 原脚本仅注释保存 | 保存注册后的模型 | 保存重建模型 | 原 state_dict 加未注册聚合层的辅助状态 |
| shell | 没有完整实验脚本 | 单数据集 runner | 跨数据集总控脚本 | 七份独立脚本，每份四条明确命令 |

从原项目提交 `ce8ea86` 到本轮审计起点，`src` 中此前已有 decoder 的设备修复与 loader 的常量时间字段修复。本轮保留这些已存在的修改，没有进一步重写模型或数据源码。

### 为什么连未注册聚合层也不顺手修复

原 Encoder 将聚合层放在普通 list 中，因此 `Adam(model.parameters())` 不更新它们；N>1 又使用列表乘法共享模块。这些行为影响训练，不能在“改入口格式”时悄悄改变。

本轮保留其原行为，单独保存/恢复这些层的参数，解决“state_dict 不包含它们，重载预测会变化”的运行产物问题。辅助保存不将它们加入优化器，不取消共享，不改变 forward。设备分配在模型构造时通过 PyTorch device context 完成，避免为迁移设备而修改模型源码。

在默认 ETTm1/paper 时间特征配置下，实际记录到注册/优化器参数 137248，另有未注册聚合参数 287008；这些是源码行为的事实，不是本轮新方法的设计或改进。

## 3. 参数对照与默认值来源

| 参数 | 论文 | 原 train.py / 源码 | 本轮设置 |
|---|---|---|---|
| 数据集 | 七个基准 | demo 只用 ETTm1 | 七个数据集分别有脚本 |
| H / pred_len | 96、192、336、720 | 96 | 四种 H 显式命令；CLI 默认 96 |
| L / seq_len | L=μH，μ∈{1,3,5,7,9}，validation 选择 | 5H | CLI 未传时 5H；脚本逐条给 L，见下一表 |
| lr / learning_rate | {1e-4,1e-3,5e-3} | 1e-4 | 起跑 1e-4，可调；不声称已完成选优 |
| batch_size | 32 | 32 | 32 |
| train_epochs | 未明确 | 10 | 10，标记为原入口回退值 |
| seed / itr | seed2024，重复3次 | 未显式固定 seed | 默认 2024/3，每次重置同一 seed；非独立种子实验 |
| optimizer / scheduler | Adam；scheduler 未明确 | Adam；无 scheduler/early stopping | 相同，不套 TSLib 的 lradj/patience |
| k / inr_layers | 默认5 | 5 | 5，保留原分支计数方式 |
| N / aggregation_layers | 候选1、2、3、5 | 1 | 起跑1，可调，原共享语义保留 |
| S / patch_length | 有分析，主表具体值未核实 | 32 | 32，可调；要求 (L+H)%S=0 |
| hidden / latent / temporal latent | 数值未核实 | 64 / 64 / 32 | 相同，可调 |
| Solver MLP 深度/宽度 | 数值未核实 | 3 / 64 | 相同，可调；不替换原类 |
| s_cff / SIREN ω | 完整数值配置未核实 | 1 / 默认30 | s_cff 可调；SIREN 沿用原模型默认 |
| ridge λ | 数值未核实 | 模型默认0.1 | ridge_lambda=0.1，可调 |
| 基础 loss | Smooth L1 | 默认 beta1、mean | 相同；smooth_l1_beta 可调 |
| 复合 loss / 初值 | 论文式12—16与代码不同 | 原 PDETimeLoss / 重建末值 | **冲突保留并报告，不按个人推导修正** |
| ETT 切分 | 6:2:2 | 7:1:2 | protocol=paper 为6:2:2；trainpy 为7:1:2 |
| 其他切分 | 7:1:2 | 原 demo 7:1:2 | 相同 |
| 时间特征 | ETTm/Weather5项，其余4项 | 自动选择 | paper 按附录；trainpy 使用自动选择 |
| 数值/时间缩放 | 真实值尺度未完整交代；时间特征[0,1] | 仅时间字段及τ按split MinMax | 沿用原 scaler，目标不标准化 |
| 建窗/切分取整 | 细节未完整交代 | 各split独立；源码整数表达式 | 沿用，实际行号和时间范围落盘 |
| 验证选优 | validation loss | 原demo没有验证 | 新增原 PDETimeLoss 的 val loss 选checkpoint；可显式改MSE |

`--protocol paper` 对齐论文明确的数据设置，不代表 loss/模型冲突已解决。`--protocol trainpy` 用于核对原 demo 的 7:1:2 与自动日历特征。两个协议的模型实现都是同一个原类。

### 独立脚本采用的起跑 L

| 数据集 | H96 | H192 | H336 | H720 |
|---|---|---|---|---|
| ETTh1 / ETTh2 | 480 | 960 | 1680 | 720 |
| ETTm1 / ETTm2 | 480 | 960 | 1680 | 3600 |
| Electricity | 480 | 960 | 1680 | 720 |
| Traffic | 480 | 960 | 1680 | 3600 |
| Weather | 480 | 960 | 1008 | 720 |

多数采用原入口 μ=5；例外均在脚本旁注释。按 PDF Table5 的长度和孤立建窗，部分 val 子集无法容纳 L=5H 与长 H。例如 Weather 的 val 约1754行，L1680+H336 已放不下。因此 Weather/H336 显式采用 μ=3，相关 H720 采用 μ=1；ETTh 的 μ=1 长 H 也与论文 Table3 的最佳报告对应。

**这些命令是可解释的起跑配置，不是全数据集已核实的最优值。** 没有暗中借用前一个split历史、截断 H 或减少 batch。实际 CSV 更短仍会明确报错；论文完整候选搜索遇到这种窗口限制需单独决定数据协议，不能悄悄改变后再称相同设置。

## 4. 文件和数据位置

```text
PDETime/
├── train.py                              # 原入口，本轮未改
├── train_reproduce.py                    # 本轮参数化入口
├── src/PDETime/                          # 原模型、loss、数据，本轮未改
├── scripts/long_term_forecast/
│   ├── ETT_script/
│   │   ├── PDETime_ETTh1.sh
│   │   ├── PDETime_ETTh2.sh
│   │   ├── PDETime_ETTm1.sh
│   │   └── PDETime_ETTm2.sh
│   ├── ECL_script/PDETime.sh
│   ├── Traffic_script/PDETime.sh
│   └── Weather_script/PDETime.sh
└── tests/test_tslib_entry.py
```

没有新增 models/、layers/、exp/ 或 data_provider/ 层级。每份 shell 中使用 `python -u train_reproduce.py`、下划线参数和显式配置；`model_name` 仅是清晰的局部变量，没有数据集矩阵或总控函数。

脚本默认数据目录：

```text
data/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
data/electricity/electricity.csv
data/traffic/traffic.csv
data/weather/weather.csv
```

若服务器 ETTm1 在 `data/ETTm1.csv`，将对应脚本的 `--root_path` 改为 `./data/`，或直接使用下面的 Python 命令。CSV 必须含有效、唯一、递增的 date 及对应真实变量。ETT 严格为 date + HUFL/HULL/MUFL/MULL/LUFL/LULL/OT，不把日历字段加入目标。非 ETT 检查变量数321/862/21，不自动填值、裁剪或转换原始数据格式。

## 5. 推荐运行顺序与命令

复用已经跑过 A5000 的环境；需要 torch、numpy、pandas、scikit-learn，绘图需要 matplotlib。原环境说明见 [01_a5000_reproduction.md](./01_a5000_reproduction.md)。

### 先看参数，不加载数据或 GPU

```bash
python train_reproduce.py --help
python train_reproduce.py --data ETTm1 --seq_len 480 --pred_len 96 --dry_run
```

### 先做 ETTm1/H96 少量 batch

```bash
python -u train_reproduce.py \
  --task_name long_term_forecast --is_training 1 \
  --model PDETime --model_id ETTm1_smoke_480_96 \
  --data ETTm1 --root_path ./data/ETT-small/ --data_path ETTm1.csv \
  --seq_len 480 --pred_len 96 --protocol paper \
  --train_epochs 1 --batch_size 2 --itr 1 --seed 2024 \
  --max_train_batches 2 --max_eval_batches 2 \
  --device cuda --checkpoints ./runs/tslib_smoke
```

在本地 CPU 检查时改成 `--device cpu`。任何 batch 限额都会产生 `partial=true`，不作为论文准确率结果。

### 再做单个完整配置

```bash
python -u train_reproduce.py \
  --model PDETime --model_id ETTm1_480_96_first \
  --data ETTm1 --root_path ./data/ETT-small/ --data_path ETTm1.csv \
  --seq_len 480 --pred_len 96 --protocol paper \
  --train_epochs 10 --batch_size 32 --learning_rate 1e-4 \
  --inr_layers 5 --aggregation_layers 1 --patch_length 32 \
  --seed 2024 --itr 1 --device cuda \
  --checkpoints ./runs/tslib_first
```

epoch10 是原脚本预算，不保证收敛。初步检查曲线与时间/显存后再运行每份完整 shell：

```bash
bash scripts/long_term_forecast/ETT_script/PDETime_ETTh1.sh
bash scripts/long_term_forecast/ETT_script/PDETime_ETTh2.sh
bash scripts/long_term_forecast/ETT_script/PDETime_ETTm1.sh
bash scripts/long_term_forecast/ETT_script/PDETime_ETTm2.sh
bash scripts/long_term_forecast/ECL_script/PDETime.sh
bash scripts/long_term_forecast/Traffic_script/PDETime.sh
bash scripts/long_term_forecast/Weather_script/PDETime.sh
```

每份独立运行对应数据集的4种H，每种重复3次。用户选择要运行的文件，不需要调用跨数据集总控脚本。默认同一 seed 重复，不能把重复所得小标准差当作独立随机初始化证据。可明确以另一个 `--seed` 运行补充实验。

### 超参数试验仅看 validation

修改单条命令的 `--seq_len`、`--learning_rate`、`--aggregation_layers` 等即可调参。论文候选集合已在参数来源中记录；本轮没有新增隐藏的自动网格调度。试验阶段使用：

```bash
python -u train_reproduce.py \
  --data ETTm1 --root_path ./data/ETT-small/ --data_path ETTm1.csv \
  --seq_len 288 --pred_len 96 --learning_rate 1e-3 --aggregation_layers 2 \
  --evaluate_test 0 --itr 1 --device cuda \
  --checkpoints ./runs/tslib_validation_trials
```

这会保存验证最优checkpoint，`test` 为 null。按validation选定配置后，再单独测试该checkpoint：

```bash
python -u train_reproduce.py \
  --is_training 0 \
  --checkpoint ./runs/tslib_validation_trials/实际实验目录/repeat0/best.pt \
  --device cuda --batch_size 32 \
  --checkpoints ./runs/tslib_selected_test
```

目录名包含真实配置/数据/源码指纹，请使用程序打印的实际路径。评估恢复训练配置，不静默接受新的网络/数据/scaler设置；换机器可显式指定新的 `--root_path`、`--data_path`，文件哈希仍须一致。旧A5000或PaperPDETime checkpoint不能直接加载成本轮完整checkpoint。

## 6. 结果与身份

每个配置保存到 `checkpoints/model_id_配置指纹/`，每次重复为 `repeat0/1/2`。已存在的目录拒绝覆盖；需要重跑时更换 `--checkpoints` 或实验标识。

```text
repeat0/
├── config.json                    # 完整实际参数
├── parameter_sources.json         # 论文、原入口回退、保留语义与新增工程
├── data_manifest.json             # 哈希、字段、切分、窗口和时间scaler
├── provenance.json                # 源码哈希、Git SHA、版本/设备与优化器覆盖
├── history.csv                    # train_loss、val_loss、val_mse、val_mae
├── best.pt                        # 原模型state、未注册聚合层辅助state、optimizer
├── prediction.png                 # 明确列名的原始尺度预测图
└── summary.json                   # 真实测试、partial、时间/训练峰值显存
```

`mse/mae` 为未来 H 步、真实 C 通道的原始尺度聚合，同时保存逐通道指标。`train_loss` 沿用原脚本的batch均值记录，`val_loss`按验证样本数加权。validation最佳checkpoint属于本轮新增的评估工程；不改训练每步所用的原 loss/Adam。

真实值不额外标准化，论文目标尺度仍未完整核实，因此不要与Table1直接计算性能差距或通过缩放汇总MSE凑数。原论文与代码的连续性损失、初值和重建语义差异也仍在，格式规范不能解决这些科学复现问题。

## 7. 实际验证记录

2026-10-21，本地 Python3.8.20、PyTorch2.0.0 CPU、NumPy1.23.5、pandas1.5.3、scikit-learn1.2.2：

| 检查 | 实际结果 |
|---|---|
| 模型/loss/一步更新保真 | 原train.py模型配置，7个时间特征；同seed、同batch，原类直接构造与新入口预测/loss/Adam后参数逐位相同 |
| 原共享和优化器覆盖 | N>1聚合仍共享；Solver中间Linear仍共享；聚合参数仍不在原optimizer中 |
| 完整checkpoint | 另一次随机初始化后恢复注册与未注册权重，同输入预测逐位一致，别名共享仍保留 |
| 数据保真 | trainpy协议的数组与原split/数据类一致；paper协议6:2:2、5个ETTm时间特征明确检查 |
| 独立shell | 七份文件经bash语法检查；用无训练stub实际执行，捕获28条命令，全部由真实parser接受，每份仅一个数据集 |
| 起跑窗口可用性 | 按PDF Table5行数检查28组 L+H 均可装入各自孤立split；实际文件还会再次检查 |
| 真实ETTm1 CPU冒烟 | 69680行，L480/H96，原默认宽度，训练/验证/测试各最多2 batch，1epoch；训练、保存、绘图成功 |
| 独立评估 | 重载上述checkpoint，原始尺度与逐通道指标与训练结束时完全一致 |
| validation-only | 真实ETTm1少量batch运行，test=null，没有做测试评估 |
| GPU/正式准确率 | 本轮未运行；不能据以上记录宣称已复现论文主表 |

新增7项 unittest，另有前版13项测试保留。原始 `train.py` 和整个 `src/PDETime/` 的Git diff保持为空。运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest discover -s tests -p test_tslib_entry.py -v
```

本轮的正确完成标准是“入口与独立命令规范化、原模型及训练语义保持、参数来源和验证记录透明”。后续如果要解决论文与源码的实质冲突，应作为独立、明确限定的方法修正实验，不与本轮保真基线混合。
