# PDETime 论文参数复现：Python 入口、推荐配置与七数据集脚本

> **2026-10-21：历史重建版本。** 本文对应 `PaperPDETime`、新目标函数及总控脚本；与保留原模型/训练语义的任务不同。当前推荐入口及审计见 [03_tslib_entry.md](./03_tslib_entry.md)。原实现与历史记录保留，不将此版视为原代码保真复现。

> 来源：PDETime [arXiv:2402.16913v1](https://arxiv.org/abs/2402.16913v1)，第 5—6 页、Table 1/5、Appendix A.2/A.4；本仓库 `train.py`。
> 更新时间：2026-10-20。实现起点：`1fe610564fcc22e6ff10aa1aea1a30382a4db68b`。
> 状态：参数化入口、shell 批量命令、机制测试与真实 ETTm1 CPU 冒烟测试已完成；A5000/3090 全量实验和论文表格指标尚未复现。

## 1. 这一版采用什么规则

**论文有明确数值时按论文设置；论文没交代时尽量继承 `train.py`，并保存来源。** 原 `train.py` 与 `scripts/reproduce_a5000.py` 保留。新入口为：

```text
scripts/reproduce_paper.py                # train / search / evaluate / dry-run
scripts/reproduce_paper.sh                # smoke / fixed / search，七数据集四种 H
tests/test_paper_reproduction.py         # unittest，无需下载数据或 GPU
```

这里的“复现”是按论文公开内容进行的重建。原仓库来自 [rohithm8/PDETime](https://github.com/rohithm8/PDETime)，尚未确认它与论文作者的官方对应关系；未公开的设置不能仅因 demo 使用了某值就认定为论文设置。

| 参数/协议 | 本版默认或候选值 | 来源与解释 |
|---|---|---|
| 数据集 | ETTh1、ETTh2、ETTm1、ETTm2、Electricity、Traffic、Weather | 论文 |
| H | 96、192、336、720；单次训练默认 96 | 论文；shell 的 fixed/search 覆盖四种 |
| L | 默认 5H；搜索 μ={1,3,5,7,9} | 公式及候选来自论文；单配置 μ=5 来自 train.py |
| 学习率 | 默认 1e-4；搜索 {1e-4,1e-3,5e-3} | 论文候选；单配置沿用 train.py |
| batch | 32 | 论文与 train.py 一致；smoke 特意减到 2 |
| 优化器/基础损失 | Adam / SmoothL1Loss，beta=1，mean reduction | 类型来自论文；其他细节采用 PyTorch/原代码默认并显式实现 |
| ETT 切分 | 6:2:2，按时间顺序 | 论文；不使用原 demo 的 7:1:2 |
| 其他切分 | 7:1:2，按时间顺序 | 论文 |
| epoch | 10，无 scheduler/early stopping | train.py 回退值；不是已核实论文值 |
| seed | 2024 | 论文 |
| 重复次数 | Python 默认 1；shell fixed/search 默认 3 | 论文写重复 3 次；先以 1 次检查流程更省资源 |
| 重复种子 | `fixed`：每次均为 2024 | 贴近“固定 seed 重复 3 次”的文字；不是独立种子统计 |
| INR k | 5 | 论文；保留原代码各分支的层数计数约定 |
| 聚合 N | 默认 1；搜索 {1,2,3,5} | 论文候选；默认来自 train.py |
| patch S | 32，要求 (L+H) 能被 S 整除 | train.py；论文主表 S 未核实 |
| hidden / latent / temporal latent | 64 / 64 / 32 | train.py，论文未明确这些数值 |
| Solver MLP 深度/宽度 | 3 / 64 | train.py |
| CFF / SIREN | scales=1；沿用 SIREN 默认 ω=30 | train.py / 源类默认 |
| ridge λ | 0.1；保持原 decoder 对含 bias 的正规矩阵加正则 | 原模型默认；论文未明确数值及 bias 约定 |
| 目标值标准化 | `none` | 继承 train.py；另支持 `standard` 作为明确记录的对照 |
| 时间特征 | ETTm/Weather 5 项；其他 4 项 | Appendix A.2；不自动加入 year/quarter |
| 时间特征缩放 | `split`，每个子集映射到 [0,1] | 继承 loader 的方式；另可选 canonical 固定日历范围 |
| 时间索引 τ | `split`，每个子集的时间戳缩放到 [0,1] | 继承 loader；也提供 train/window 两种坐标对照 |
| 边界历史 | `carry` | 重建选择：val/test 可带前 L 个已知历史点，未来标签仍属于当前子集；支持长 L 网格 |
| checkpoint 与超参数选优 | validation `prediction_loss` 最小 | 论文说 validation loss；此处明确采用未来残差 Smooth L1，可改 loss/mse |
| 聚合层权重 | 正确注册、N 层独立初始化 | 重建选择；原列表乘法共享问题不作为默认 N 层实现 |
| Solver 隐藏层权重 | `trainpy`：保留原代码中间 Linear 共享 | 未公开细节尽量继承；提供 independent 作为消融，不能悄悄改变 |

精确整数边界采用 `train_end=int(n*train_ratio)`、`val_end=train_end+int(n*val_ratio)`，余下为 test。这里没有擅自改为固定月份截取。CSV 行数与 PDF Table 5 不同时会打印并保存备注，不静默裁剪数据；可用 `--strict-paper-shape` 严格拒绝。

## 2. 模型方面做了哪些必要对齐

不能仅替换 H、lr、seed 而沿用旧损失：

1. **初值来自观测。** `initial = x[:, -1:, :]`，不使用历史重建值替代。
2. **统一残差。** ridge 仅用历史 `x-initial` 拟合；历史与未来均输出相对于同一初值的残差，预测加回 initial。
3. **外层目标。** 对照论文式 12—16，使用 `Lp + Lc + Lf`；`Lr` 记录为历史重建诊断，不再额外加入外层。ridge 的内层仍采用原 decoder 的平方正则闭式解，允许训练梯度通过它回传。
4. **一阶差分。** 残差序列在历史/未来边界具有统一含义，首个未来点也参与差分损失。
5. **连续性。** 沿用 Appendix A.4/原 Solver 的 patch 尾部锚点反向积分，增加相邻 patch 锚点的一致性约束。在最终 Solver MLP 前计算 Lc；该具体离散化是公开公式的重建选择。
6. **测试适应。** 测试时 ridge 只看历史观测，不利用未来测试标签更新模型。

令每个 patch 尾部锚点为 $a_j$，下一个 patch 内的离散导数为 $v_{j+1,s}$，默认 $\Delta\tau=1$，本版连续性为：

$$
L_c=\operatorname{SmoothL1}\left(a_{j+1},a_j+\sum_{s=1}^{S}v_{j+1,s}\right).
$$

当整段只有一个 patch 时没有跨 patch 边界，Lc=0。所有项按参与元素取 mean，不在已经取 mean 的 Smooth L1 上再次盲目除 H/L。`--continuity-weight` 与 `--difference-weight` 默认均为 1；改权重属于消融。

保留 Solver 中间权重共享、τ/日历每 split 缩放是为了遵循当前“未给出时结合 train.py”的策略，不代表已经验证这些是作者真实实现。需要比较时，单独记录 `--solver-weight-sharing independent`、`--calendar-scaling canonical`、`--time-index-mode window` 等对照。

## 3. 数据放置与环境

在已经跑过 A5000 的 Python 环境中运行即可；需要 `torch`、`numpy`、`pandas`，绘图需要 `matplotlib`。默认 `--num-workers 0`。先检查：

```bash
python -c "import sys,torch,numpy,pandas; print(sys.version); print(torch.__version__,torch.cuda.is_available())"
python scripts/reproduce_paper.py --help
```

原有环境/GLIBC 排错见 [01_a5000_reproduction.md](./01_a5000_reproduction.md)。新代码在 Python 3.8.20、PyTorch 2.0.0 CPU 上实际测试；这不是论文指定软件版本。服务器如使用其他 CUDA/torch 版本，脚本会保存实际版本。

所有 CSV 应含 `date` 和真实数值变量。ETT 必须恰好为以下列；不会把派生日历列加入目标：

```text
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
```

推荐目录：

```text
data/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── electricity/electricity.csv      # date + 321 列
├── traffic/traffic.csv              # date + 862 列
└── weather/weather.csv              # date + 21 列
```

也支持直接放 `data/ETTm1.csv`、`data/electricity.csv` 等。Python 可通过 `--data` 指定任意路径，shell 可用 `DATA_ROOT=/actual/root`。日期必须有效、唯一且严格递增，目标数值须有限；不自动填充缺失值或转换原始 UCI/PEMS 格式。

数据来源：ETT 为 [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)；论文 Appendix A.1 对其余三类数据给出了 [UCI Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)、[PEMS Traffic](https://pems.dot.ca.gov/)、[Jena Weather](https://www.bgc-jena.mpg.de/wetter/) 来源。脚本使用已整理的多变量 CSV，并保存 SHA-256。不同预处理版本不因数据集同名就自动等价，尤其应核对 PDF Table 5 的行数。

**不提交原始数据和训练产物。** 本仓库忽略 `data/`、`runs/`、`.pt` 等；脚本不会自动下载或上传数据。

## 4. 推荐先运行什么

所有命令在仓库根目录执行。输出目录必须不存在；重复实验应更换 `RUN_ROOT`/`--run-dir`，脚本不会覆盖已有结果。

### 4.1 第一步：A5000 冒烟测试

先打印命令，确认数据根目录与输出：

```bash
bash scripts/reproduce_paper.sh smoke --dry-run
```

实际运行，默认 ETTm1/H96、L480、batch2、1 epoch、训练/评估各 2 batch：

```bash
DATA_ROOT=/path/to/data \
RUN_ROOT=runs/paper_smoke \
bash scripts/reproduce_paper.sh smoke
```

等价 Python 命令：

```bash
python scripts/reproduce_paper.py \
  --mode train --dataset ETTm1 --data data/ETT-small/ETTm1.csv \
  --horizon 96 --lookback-multiplier 5 \
  --epochs 1 --batch-size 2 --lr 1e-4 --seed 2024 \
  --max-train-batches 2 --max-eval-batches 2 \
  --device cuda --run-dir runs/paper_smoke_python
```

任何 batch 限额都会在 summary 中标记 `partial=true`，不能用来填写论文主表。

### 4.2 第二步：ETTm1/H96 完整单配置，先确认训练曲线

```bash
DATASETS=ETTm1 HORIZONS=96 REPEATS=1 \
DATA_ROOT=/path/to/data RUN_ROOT=runs/paper_ettm1_h96 \
bash scripts/reproduce_paper.sh fixed
```

**推荐起跑参数：**H96、L480、Adam 1e-4、batch32、epoch10、seed2024、k5、N1、S32、hidden/latent64、temporal latent32、λ0.1。epoch10 是第一阶段预算，并不保证收敛。观察训练损失 Lp/Lc/Lf 和 validation 曲线后再决定是否增加预算，保持各候选的预算一致。

Python 可逐项调整：

```bash
python scripts/reproduce_paper.py \
  --mode train --dataset ETTm1 --data data/ETT-small/ETTm1.csv \
  --horizon 96 --lookback-multiplier 5 \
  --epochs 10 --batch-size 32 --lr 1e-4 --seed 2024 \
  --inr-layers 5 --aggregation-layers 1 --patch-length 32 \
  --hidden-features 64 --latent-features 64 --temporal-latent-features 32 \
  --mlp-hidden-layers 3 --mlp-hidden-features 64 \
  --ridge-lambda 0.1 --value-scaler none \
  --device cuda --run-dir runs/paper_ettm1_h96_python
```

### 4.3 第三步：ETTm1 四种 H 的固定配置

```bash
DATASETS=ETTm1 DATA_ROOT=/path/to/data \
RUN_ROOT=runs/paper_ettm1_four_horizons \
bash scripts/reproduce_paper.sh fixed
```

默认每种 H 重复 3 次，共 12 次训练。L=5H 分别为 480、960、1680、3600。它覆盖四种任务，但没有执行论文给出的全部超参数搜索。

### 4.4 第四步：从 ETTm1/H96 的验证搜索开始

```bash
DATASETS=ETTm1 HORIZONS=96 DATA_ROOT=/path/to/data \
RUN_ROOT=runs/paper_search_ettm1_h96 \
bash scripts/reproduce_paper.sh search
```

默认验证搜索 5×3×4=60 组：μ、lr、N 的候选均来自论文。所有 trial 只评估 validation；选定配置写入 `selected_H96.json` 后，重新训练 3 次并分别测试。共 63 次训练。

完整组合网格及“单次搜索、选定后重训三次”是明确的重建流程，论文没有完整交代联合搜索顺序。不要把当前脚本的搜索调度说成已核实作者原流程。

### 4.5 七个数据集、四种 H：全部命令

```bash
# 只检查展开的 28 条数据集/H 命令
bash scripts/reproduce_paper.sh fixed --dry-run
bash scripts/reproduce_paper.sh search --dry-run

# 固定配置：7×4×3 = 84 次训练
DATA_ROOT=/path/to/data RUN_ROOT=runs/paper_all_fixed \
bash scripts/reproduce_paper.sh fixed

# 完整验证网格：7×4×60=1680 个 trial，再加 7×4×3=84 次最终训练
DATA_ROOT=/path/to/data RUN_ROOT=runs/paper_all_search \
bash scripts/reproduce_paper.sh search
```

七个数据集对应的显式命令也可以逐一执行：

```bash
DATASETS=ETTh1       bash scripts/reproduce_paper.sh search
DATASETS=ETTh2       bash scripts/reproduce_paper.sh search
DATASETS=ETTm1       bash scripts/reproduce_paper.sh search
DATASETS=ETTm2       bash scripts/reproduce_paper.sh search
DATASETS=Electricity bash scripts/reproduce_paper.sh search
DATASETS=Traffic     bash scripts/reproduce_paper.sh search
DATASETS=Weather     bash scripts/reproduce_paper.sh search
```

每条默认包含 H=96/192/336/720，并使用不同目录。A5000 24GB 能否跑完所有大 L/N/通道组合需要实测，不能由小样本通过推断。搜索会记录 ValueError、非有限损失或 CUDA OOM 的失败 trial；不静默降低 batch 或模型宽度。若候选未全部成功，结果必须连同 `trials.json` 中的失败记录解释。

## 5. 标准化、重复及其他对照

### 5.1 标准化不要与继承设置混淆

默认 `none` 与 train.py 一致，`mse/mae` 就是原始尺度；不能与论文 Table 1 的数字直接判定优劣。提供训练集拟合的 StandardScaler 对照：

```bash
VALUE_SCALER=standard DATASETS=ETTm1 HORIZONS=96 REPEATS=1 \
DATA_ROOT=/path/to/data RUN_ROOT=runs/paper_ettm1_h96_standard \
bash scripts/reproduce_paper.sh fixed
```

这会同时保存标准化尺度 `mse/mae` 和反变换尺度 `raw_mse/raw_mae`。后者由逐通道缩放后的误差重新计算，不能对汇总 MSE 任意乘一个系数代替。`data_manifest.json` 保存训练区间均值和标准差；val/test 不参与真实值 scaler 拟合。

### 5.2 固定 seed 重复和多 seed 是两套协议

默认 `REPEAT_SEED_MODE=fixed` 下每次重置为 2024，CPU 确定性运行可能完全相同；汇总会明确 `independent_seeds=false`，标准差不能作为独立随机初始化证据。

需要独立种子稳健性检查时：

```bash
DATASETS=ETTm1 HORIZONS=96 REPEATS=3 REPEAT_SEED_MODE=offset \
DATA_ROOT=/path/to/data RUN_ROOT=runs/paper_ettm1_h96_multiseed \
bash scripts/reproduce_paper.sh fixed
```

使用 2024、2025、2026；这是补充多 seed 协议，不宣称论文原本就是这三个种子。可设 `DETERMINISTIC=1` 启用 PyTorch 确定性算法，具体硬件/算子不支持时会明确报错。

### 5.3 可选改动单独消融

```bash
SOLVER_WEIGHT_SHARING=independent DATASETS=ETTm1 HORIZONS=96 REPEATS=1 \
RUN_ROOT=runs/paper_solver_independent bash scripts/reproduce_paper.sh fixed

CALENDAR_SCALING=canonical TIME_INDEX_MODE=window \
DATASETS=ETTm1 HORIZONS=96 REPEATS=1 \
RUN_ROOT=runs/paper_coordinate_ablation bash scripts/reproduce_paper.sh fixed
```

`calendar_scaling=canonical` 使用年内日 (doy−1)/365、月 (month−1)/11、周内日/6、小时/23、分钟/59。`time_index_mode=window` 每个 L+H 窗口采用 [0,1]；`train` 使用训练区间确定的统一时间原点与跨度，未来 τ 可以大于 1。默认 split 包括已知历史上下文的时间范围，不用未来真实目标拟合预处理。

## 6. 检查配置与恢复评估

不加载数据、不要求 CUDA，仅查看参数及来源：

```bash
python scripts/reproduce_paper.py \
  --dataset ETTm1 --horizon 192 --lookback-multiplier 3 \
  --run-dir unused --dry-run
```

独立评估某个 checkpoint：

```bash
python scripts/reproduce_paper.py \
  --mode evaluate \
  --checkpoint runs/paper_ettm1_h96/fixed/ETTm1/H96_seed2024/repeat0/best.pt \
  --data data/ETT-small/ETTm1.csv \
  --eval-split test --device cuda --batch-size 32 \
  --run-dir runs/paper_ettm1_h96_recheck
```

模型/数据协议从 checkpoint 恢复，不允许通过评估命令悄悄修改网络与 scaler；数据 SHA-256 必须匹配。评估可调整设备、batch、workers、绘图列和 batch 限额。当前入口不实现中断续训；已有输出目录不会被覆盖，也不会自动跳过旧实验冒充完成。

## 7. 输出文件和指标含义

```text
run-root/
├── parameter_sources.json           # 根配置、论文来源/回退值/重建选择
├── repeat0/                         # train 模式；后续还有 repeat1/2
│   ├── config.json
│   ├── parameter_sources.json
│   ├── provenance.json              # Git SHA、源码文件哈希、数据哈希、版本/硬件
│   ├── data_manifest.json           # 字段、切分、窗口/坐标和 scaler 统计
│   ├── history.csv                  # 分项 train/val loss、MSE/MAE、累计时间
│   ├── best.pt                      # validation 选出的权重、优化器与完整配置
│   ├── prediction.png               # 按列名 OT 或第一个真实变量，原始尺度
│   └── summary.json                 # 测试、参数量、fit 时间、训练峰值显存
└── summary.json                     # 每次结果、种子、均值/标准差、partial 标记
```

search 模式另有 `trials.json`、`selected_H*.json`、`H*/mu*_lr*_N*/`、`H*/final_repeat*/` 和 `results_H*.json`。trial 的 summary 中 `test=null`，不会每个候选都看一次测试再选最优。

## 8. 已执行的验证与当前边界

| 验证 | 实际状态 |
|---|---|
| 13 项 unittest | 已通过：真实变量白名单、训练集 scaler、历史上下文与标签边界、常数序列、差分/重建语义、patch 积分与 Lc、聚合注册/共享、有限梯度、检查点恢复、搜索顺序、七适配器、四种 H、28 条 shell 命令 |
| 七个数据集适配 | 已用合成 CSV 检查通道数/时间特征/前向结构；并非七个真实数据集训练 |
| 真实 ETTm1 | 69680 行、7 个变量；L480/H96、原默认宽度、CPU 训练/评估各 2 batch、1 epoch，保存/重载/测试/绘图成功 |
| 缩小验证搜索 | 真实 ETTm1，H96，μ={1,3}，小网络与 batch 限额；validation 选择后重新训练两次并评估成功 |
| 独立 evaluate | 重载真实 ETTm1 冒烟 checkpoint，与训练末测试的 MSE/MAE 一致 |
| shell / Python | `bash -n`、Python AST、CLI help、参数展开通过 |
| A5000/3090 全量实验 | 本次未执行；耗时、显存与正式准确率待服务器实测 |

本地测试环境为 Python 3.8.20、PyTorch 2.0.0 CPU、NumPy 1.23.5、pandas 1.5.3。机制检查可复跑：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest discover -s tests -v
```

**下一步：先 A5000 smoke，再 ETTm1/H96 完整固定配置，确认曲线和资源后再搜索。** 论文未明确的数据尺度、最佳 S/隐藏维度/λ、训练停止预算、源码身份、重复实验含义仍需要进一步证据；默认回退值不会被包装成已确认论文值。
