# PDETime 复现实验 01：Linux + RTX A5000 / ETTm1

> 维护日期：2026-10-20；仓库：[yigeyouzia/PDETime](https://github.com/yigeyouzia/PDETime)；原项目：[rohithm8/PDETime](https://github.com/rohithm8/PDETime)  
> **状态：复现脚本已提交，尚未在目标 A5000 上实测。** 本文提供可执行的步骤、检查点和排错方法。**跑通 ≠ 已复现论文表格**。

## 0. 目标与已核实的代码

这份指南从空白 Linux 环境开始，到数据下载、依赖安装、GPU 冒烟测试、完整训练、验证、测试与结果落盘。

当前仓库结构（2026-10-20 检查）：

```text
PDETime/
├── README.md
├── train.py                         # 原始演示入口：10 epoch，仅绘图
├── src/PDETime/
│   ├── preprocess/dataloader.py     # 数据集、时间特征、70/10/20 切分
│   └── models/
│       ├── encoder.py               # SIREN / Fourier / attention
│       ├── solver.py                # patch + 积分
│       ├── decoder.py               # ridge regression
│       └── pdetime_model.py         # PDETime + 3 项 loss
├── scripts/reproduce_a5000.py       # 本次增加：可执行训练/评估入口
└── docs/experiments/01_a5000_reproduction.md
```

原始 `train.py` 的默认参数：

| 配置 | 值 |
| --- | --- |
| 数据 | ETTm1.csv（date + 7 个变量，含 OT） |
| horizon | 96 |
| lookback | 480 (= 5 × horizon) |
| batch_size | 32 |
| epochs | 10 |
| lr | 1e-4 / Adam |
| 时序隐藏特征 | 32 |
| latent_features / hidden_features | 64 / 64 |
| INR_layers / aggregation_layers | 5 / 1 |
| patch_length | 32 |
| 损失 | PDETimeLoss(SmoothL1Loss)：历史重建 + 未来残差 + 一阶差分 |

**重要：** `scripts/reproduce_a5000.py` 沿用源模型及损失函数，但为了让训练可执行/可评估，注册聚合层、移除逐样本打印、修正运行设备与绘图方式、添加 val/test 指标及保存检查点。因此它是一个**带工程修复的可复现基线**，与未改动的原始演示代码**不是严格等价实现**。

## 1. GPU 和驱动预检查

RTX A5000 通常为 **24 GB VRAM**。实际可用容量取决于服务器是否共享 GPU。不要仅凭型号假定显存空闲或 CUDA 版本。

```bash
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv
which python3
df -h .
```

应看到 `NVIDIA RTX A5000` 和 GPU 驱动。若 `nvidia-smi` 不可用，应先让管理员安装/恢复驱动，而非只安装 CUDA Toolkit。以下环境优先使用 PyTorch 官方 **CUDA 12.1 wheel**（前提是驱动支持），不要求先安装完整 CUDA Toolkit。如果 `nvidia-smi` 显示旧驱动不兼容，应按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择对应 wheel/升级驱动，而不要盲装。

## 2. 获取代码与隔离环境

```bash
# 在你希望放置代码的工作目录运行
git clone https://github.com/yigeyouzia/PDETime.git
cd PDETime
git rev-parse HEAD
git status --short

# 若服务器已有 Conda，可直接使用：
conda create -n pdetime python=3.10 -y
conda activate pdetime

python -m pip install --upgrade pip
# 固定一组便于记录的推荐版本；联网安装需要访问 PyPI/PyTorch 下载服务器
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install "numpy==1.26.4" "pandas==2.2.2" "scikit-learn==1.5.1" "matplotlib==3.9.2" "tqdm==4.66.5"
```

如果 `torch==2.3.1+cu121` 不适配服务器驱动或 Python，请从官方页面选择与当前驱动兼容的安装命令，记录最终版本。不要混用系统 Python 的 torch 和 Conda 的包。

环境核对：

```bash
python - <<'PY'
import sys, torch, numpy, pandas, sklearn, matplotlib
print("python", sys.version)
print("torch", torch.__version__, "torch CUDA", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
print("numpy", numpy.__version__, "pandas", pandas.__version__,
      "sklearn", sklearn.__version__, "matplotlib", matplotlib.__version__)
assert torch.cuda.is_available(), "CUDA 不可用，请先排查驱动/torch wheel"
PY
```

### GLIBC 2.28 报错时的兼容环境替代方案

如果执行 `conda create -n pdetime python=3.10 -y` 后，激活环境运行 `python --version` 提示：

```text
python: /lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.28' not found
```

这是 **Conda 下载的 Python 二进制与 Linux 系统 GLIBC 版本不兼容**，与 A5000 或 CUDA 无关。先收集系统版本：

```bash
cat /etc/os-release
ldd --version | head -n 1
conda list -n pdetime python
```

优先**保留旧环境**，从 conda-forge 单独创建兼容环境，严格只从该渠道选择包，避免默认 Anaconda 渠道选到依赖较新 GLIBC 的 Python：

```bash
conda deactivate
conda create -n pdetime_cf --override-channels -c conda-forge python=3.10 pip -y
conda activate pdetime_cf
which python
python --version
python -m pip --version
```

`conda-forge` 官方默认 Linux GLIBC 基线为 2.17，但个别包可能要求更高版本，因此最终仍需以 `ldd --version` 和实际 `python --version` 是否成功为准。如果 Python 可启动，继续运行本节上面的 PyTorch / NumPy / Pandas 等依赖安装命令；若仍报 GLIBC，暂时停止安装并核对系统、Conda 包来源和 `LD_LIBRARY_PATH`。**不要自行覆盖系统 libc.so.6，也不要通过 CONDA_OVERRIDE_GLIBC 伪装系统版本**。

参考：[conda-forge GLIBC 兼容性](https://conda-forge.org/docs/maintainer/knowledge_base/) · [conda-forge 渠道隔离](https://conda-forge.org/docs/user/transitioning_from_defaults/)。

## 3. 下载 ETTm1 数据集

**权威数据来源**：[zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)。当前训练入口使用的是 **ETTm1**，不是 ETTh1，也不是任意 PDE 模拟数据；ETTm1 记录了约两年、每 15 分钟一条的变压器数据，官方文件 70,080 行，CSV 约 10 MB。

```bash
mkdir -p data
curl -fL --retry 3 \
  https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv \
  -o data/ETTm1.csv

# 如果 curl 失败，也可以：
# wget -O data/ETTm1.csv https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv
ls -lh data/ETTm1.csv
sha256sum data/ETTm1.csv
python - <<'PY'
import pandas as pd
p = "data/ETTm1.csv"
df = pd.read_csv(p)
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print(df.head(3).to_string(index=False))
print("nulls:", df.isna().sum().to_dict())
assert len(df) == 70080, "数据集记录数异常，请检查下载/版本"
assert df.columns.tolist() == ["date","HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]
assert not df.isna().any().any(), "存在空值，请先核对数据"
PY
```

**安全提示：** `data/` 本地数据与 `runs/` 实验产物不要推送仓库；下面给出了仅需在服务器运行的 `.git/info/exclude` 规则，不会改动共享 `.gitignore`。

```bash
printf '\n/data/\n/runs/\n' >> .git/info/exclude
```

## 4. 运行前理解三个工程问题

1. **聚合层未注册**：`src/PDETime/models/encoder.py` 使用普通 Python list 存储 `agg_layers`，导致 `model.parameters()`、`.to(cuda)` 和 `state_dict()` 均不涵盖该层。新入口在模型构建后用 `nn.ModuleList` 包装它，**不改原模型源码**。若直接运行原始 `train.py`，该缺陷在 GPU 上可能造成设备不一致。
2. **原始逐样本打印**：`PDETimeDataset.__getitem__` 中有 `print(idx, len(self))`，导致数万次日志输出。新入口提供行为等价但不打印的子类。
3. **原始入口未保存/评估**：`train.py` 没有完整 val/test 流程，也未输出 MSE/MAE。新入口按验证集未来区间 MSE 选最好 checkpoint，最后仅在测试集评估一次；可画 `OT` 示例。

此外，默认 `lookback + horizon = 576`，可以被 `patch_length=32` 整除；更换预测长度时，要保证两者之和能被 32 整除。数据集内部采用时序 **70% train / 10% validation / 20% test**（与某些 ETT 论文的标准划分未必相同），且时间特征由各分片分别 MinMax 缩放；数值目标列不经过这个 scaler。**切分、时间特征处理和评估定义必须明确记录，否则不要直接比较论文榜单**。

## 5. 先跑 GPU 冒烟测试

从项目根目录执行（不要 `cd scripts` 后运行）：

```bash
mkdir -p runs
python scripts/reproduce_a5000.py \
  --device cuda \
  --epochs 1 \
  --batch-size 2 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --run-dir runs/smoke_a5000 \
  2>&1 | tee runs/smoke_a5000_console.log
```

说明：
- 只训练 2 个 batch 且只评估各 2 个 batch；**冒烟数字不具备论文比较意义**。
- 预期启动时打印 GPU、数据行数、变量维度及 train/val/test window 数；随后输出一条 `train_loss/val_mse/val_mae` 和测试 MSE/MAE。
- 检查 `runs/smoke_a5000/` 下是否出现 `best.pt`、`history.csv`、`summary.json`、`prediction_ot.png`。出现这些文件意味着本次脚本流程执行完成，但不证明预测质量。
- 若显存不足，先结束其它占用显存的任务，再把 `--batch-size` 改为 1。

## 6. 完整 ETTm1 / H96 训练与测试

冒烟测试通过后，使用原始演示配置（10 epochs、batch size 32）进行一次完整基线：

```bash
python scripts/reproduce_a5000.py \
  --data data/ETTm1.csv \
  --device cuda \
  --epochs 10 \
  --batch-size 32 \
  --horizon 96 \
  --lookback 480 \
  --lr 0.0001 \
  --seed 2026 \
  --num-workers 0 \
  --run-dir runs/ETTm1_H96_L480_B32_seed2026 \
  2>&1 | tee runs/ETTm1_H96_L480_B32_seed2026_console.log
```

训练通常包含数万滑动窗口，会明显慢于冒烟测试。**这里没有在目标 GPU 上跑过，不能提供可信耗时、峰值 VRAM、最终 MSE 或 MAE**。第一次先不要擅自提升 batch 或换 30/50 epochs；先跑通基线并记录，再设计单变量实验。

服务器断开 SSH 也要保留任务时，建议使用 `tmux`：

```bash
tmux new -s pdetime
# 粘贴上面的完整训练命令
# 按 Ctrl+B 再按 D 脱离会话
tmux attach -t pdetime
```

另开终端查看：

```bash
nvidia-smi
tail -f runs/ETTm1_H96_L480_B32_seed2026_console.log
```

**注意：** 使用 `cmd | tee file` 时 shell 默认可能只返回 tee 的退出码。正式脚本若需要可靠退出码，请先 `set -o pipefail`，或改用 `python ... > run.log 2>&1`。新 runner 只在 epoch 边界保存最佳模型，**没有从 checkpoint 续训功能**。

## 7. 产物与指标解释

```text
runs/ETTm1_H96_L480_B32_seed2026/
├── best.pt                 # 最佳验证集 MSE 的模型/优化器状态及训练配置
├── history.csv             # 每轮 train_loss、val_mse、val_mae
├── summary.json            # 数据窗口数、最好 epoch、测试指标、配置
└── prediction_ot.png       # 测试集首个窗口最后一个变量 OT 的预测曲线
runs/ETTm1_H96_L480_B32_seed2026_console.log
```

```bash
python - <<'PY'
import json, pandas as pd
run="runs/ETTm1_H96_L480_B32_seed2026"
with open(f"{run}/summary.json") as f:
    s=json.load(f)
print("best epoch", s["best_epoch"], "test", s["test"])
print(pd.read_csv(f"{run}/history.csv").to_string(index=False))
PY
```

- `train_loss`：原仓库的三项 SmoothL1 组合损失，不是 MSE。
- `val_mse, val_mae, test.mse, test.mae`：对 **未来 96 步全部 7 个变量**计算的逐元素均值；以目标数值原单位衡量（原始数据加载器没有缩放这 7 个预测值）。
- 预测构造遵循原始 `train.py` 的可视化语义：未来残差输出加上模型在 `lookback-1` 的重建值。
- `prediction_ot.png` 是一个示例窗口的最后一列 OT 曲线，不代表整体测试性能。
- 如启用 `--max-eval-batches`，`summary.json` 只覆盖部分评估数据，不可标注为“全测试集”。
- 由于注册聚合层、与论文标准训练/划分流程可能不同，这些数字**不能直接声称复现论文 SOTA**。

## 8. 显存调优：A5000 24 GB

请按以下顺序做，且每次只改一个参数：

| 场景 | 处理建议 | 影响 |
| --- | --- | --- |
| CUDA OOM | batch 32 → 16 → 8 → 4 → 2 | 有效 batch 改变，指标不再严格等同 |
| GPU 利用率低 | 在不 OOM 前提下测试 batch 64 | 更大显存与不同优化轨迹 |
| CUDA 有空闲显存 | 不一定需要全部用满 | 利用率、吞吐和训练稳定性更重要 |
| 数据加载较慢 | 试 `--num-workers 2` | 增加 CPU 内存/进程开销 |
| 不稳定或 loss 为 NaN | 先降低 lr、核查 CSV | 改变实验配置，需要记录 |
| 想比较多个 seed | 分别设置 `--seed 2026/2027/2028` 和不同 `--run-dir` | 汇报 mean/std |

暂时不要开启 AMP 混合精度：本模型的 Decoder 有 `torch.linalg.solve`，低精度/矩阵条件数可能带来数值问题；要做混合精度应单独验证 solve 的浮点精度与梯度稳定性。

## 9. 排错对照

| 异常 | 优先检查 |
| --- | --- |
| `No module named src` | 更新至包含入口路径修复的最新版脚本，并从仓库根目录执行 |
| `torch.cuda.is_available() == False` | 驱动、PyTorch wheel / Conda 环境、CUDA_VISIBLE_DEVICES |
| `FileNotFoundError: data/ETTm1.csv` | 下载失败、当前工作目录不在仓库根目录 |
| 形状不匹配 | `lookback+horizon` 必须能被 32 整除；确认读的是 ETTm1 |
| CUDA OOM | `nvidia-smi` 检查其它任务，先将 batch 降至 8 或 2 |
| `Expected all tensors to be on the same device` | 使用新 runner，勿直接照搬未修复的 `train.py` |
| CPU 很慢 | 确认 `--device cuda`、GPU 确实可见 |
| 误认为测试指标很差 | 对照历史窗口、未来残差和未归一化目标定义；检查三个 loss 的定义 |
| 大量重复输出 `idx, len` | 原始 dataloader 有 print，改用新 runner |

## 10. 后续研究和可比较性

本仓库更接近 **PDE 视角的长序列预测**，而非典型的“明确写出 Navier–Stokes/PDE 残差、通过自动微分约束”的 PINN。其 Encoder → Solver → Decoder 路线可作为“时间序列 + 动力学归纳偏置”的研究参考，但**不能把 Solver 名字直接等同于显式 PDE 物理约束**。

跑完 H96 基线后，再考虑：
1. 用不同 seed 重复 3 次，保存 MSE/MAE 的 mean ± std；
2. 先确认文献采用的 train/val/test、归一化和评估定义，再做 H192/H336/H720 及其他 ETT 子集；
3. 对 Encoder（INR/CFF/SIREN）、Solver、3 项 loss 分组消融；
4. 对比 DLinear、PatchTST、iTransformer、PINNsFormer 等，统一窗口、训练预算和评估划分；
5. 探索将显式 PDE 约束、观测稀疏性、时空场重建引入现有预测管线。

## 11. 实验记录模板

```text
日期 / 服务器：
Git commit：git rev-parse HEAD
GPU：nvidia-smi
Python / torch / CUDA：
数据集 SHA256：
命令行：
seed / batch / horizon / lookback / epochs / lr：
训练完成：是 / 否
最佳 epoch：
验证 MSE / MAE：
测试 MSE / MAE：
最大显存 / 运行时长（实测）：
问题与修复：
备注：带工程修复的基线；不宣称与原论文严格一致
```

参考代码：
- [原始训练入口](../../train.py)
- [完整训练与评估脚本](../../scripts/reproduce_a5000.py)
- [PDETimeDataset](../../src/PDETime/preprocess/dataloader.py)
- [PDETime 模型与损失](../../src/PDETime/models/pdetime_model.py)
- [ETT 官方数据](https://github.com/zhouhaoyi/ETDataset/tree/main/ETT-small)
