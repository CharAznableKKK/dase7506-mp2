# DASE7506 MP2 — 选择可靠的 LLM 数学答案

官方测试准确率：**82.22698072805139%（384/467）**。

本仓库包含冻结后的完整方法、匹配的模型资产、官方数据与评测工具、报告，以及实验和复现证据。评测在 CPU 上离线运行，从每题四个已有候选中选择一个答案。复现提交成绩不需要重新训练。

## 报告与目录

- [报告正文](report/REPORT.md)：完整英文报告，第 7 节包含 AI 协助与复用说明。
- [分页报告](report/REPORT.html)：与 Markdown 正文内容一致的独立 HTML 版本，按 A4 排为六页，包含参考文献。下载仓库后用浏览器打开；GitHub 页面可能只显示 HTML 源码。无需联网或生成 PDF。如需打印，使用 A4、100% 缩放，并关闭浏览器页眉和页脚。页数核验针对这一固定版式，不适用于任意 Markdown 渲染方式。
- [实验协议](docs/RESEARCH_PROTOCOL.md)、[全部验证配置](docs/EXPERIMENTS.md)、[数据分析](docs/DATA_ANALYSIS.md)。
- [复现核验记录](records/release/verification.json)：数据完整性、接口、离线运行、资源和成绩复现检查。
- [文件清单](records/release/files.json)：发布文件的字节数与 SHA-256；清单本身和 Git 元数据不计入清单。

```text
assets/final/       最终配置 config.json 和权重 weights.npz，两者均需提供
code/               方法、训练、分析、测试及保持原样的官方工具
code/data/          完整官方数据、清单、许可证和归属说明
docs/               实验协议、实验记录与数据分析
guide/              保持原样的官方作业指南
report/             Markdown 报告及其 HTML 分页版本
records/            结果、成本、哈希、检查和复现证据
requirements.txt    项目依赖，同时引用 code/requirements.txt
```

本项目的推理入口为 **`code/selection.py`**。`code/submission.py` 是保留的官方空接口。`code/README.md` 介绍课程提供的初始代码包；本项目的安装、训练和评测命令如下。

## 安装与完整性检查

使用 Python **3.12**、NumPy **2.5.3** 和 threadpoolctl **3.6.0**。先克隆仓库：

```powershell
git clone https://github.com/CharAznableKKK/dase7506-mp2.git
cd dase7506-mp2
```

以下命令均从仓库根目录运行。Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe code\check_data.py
```

安装依赖时可能需要联网，预测和评分无需联网。仓库已包含完整数据。数据检查器读取参考答案仅用于完整性与对齐检查，不用于方法开发。请保留 `.gitattributes`，以便 Windows 上也能保持官方 LF 字节；不要修改官方数据、数值解析器、评分器或清单。

Linux/macOS 安装命令：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python code/check_data.py
```

在 Linux/macOS 上执行后续命令时，将 `.\.venv\Scripts\python.exe` 替换为 `.venv/bin/python`，路径中的反斜杠改为 `/`；线程变量使用 `export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4` 设置。本项目实际验证的平台为 Windows。

## 无需训练复现冻结成绩

```powershell
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$elapsed = [System.Diagnostics.Stopwatch]::StartNew()
.\.venv\Scripts\python.exe code\predict.py --method code\selection.py --checkpoint assets\final --split test --output code\runs\reproduce\test-predictions.json
if ($LASTEXITCODE -ne 0) { throw 'Prediction failed' }
.\.venv\Scripts\python.exe code\score_predictions.py --split test --predictions code\runs\reproduce\test-predictions.json --output code\runs\reproduce\test-score.json
if ($LASTEXITCODE -ne 0) { throw 'Scoring failed' }
$elapsed.Stop()
$elapsed.Elapsed.TotalSeconds
```

预期评分字段：`correct = 384`、`n = 467`、`accuracy = 82.22698072805139`，其中 `accuracy` 的单位为百分比。外部计时覆盖两个新进程的启动、导入、加载、预处理、预测和独立评分；不能用 JSON 内部计时之和替代。两个进程顺序执行，上述环境变量用于限制数值库线程；核验记录显示实际加载的单个 OpenBLAS 线程池为四线程。跨平台复现时，数值差异及其对选择结果的影响应单独记录。

原始完整流程在 Intel Core i9-12900H／Windows 11 上耗时 0.7545 秒；独立环境复现耗时 1.2876 秒，仓库独立目录核验耗时约 0.7921 秒。以上均为本机测量，**参考 Xeon Platinum 8457C 未实测**。

方法仅加载配置与权重。存档预测用于独立评分核对，选择方法不会读取它们。对于新题库，可使用官方运行器的 `--split audit --data-dir <audit-input-directory>` 参数，并沿用相同的方法与 checkpoint。

## 复现训练与检查

```powershell
.\.venv\Scripts\python.exe code\train_model.py --config assets\final\config.json --output code\runs\retrained-final --record code\runs\retrained-final-training.json
.\.venv\Scripts\python.exe -m unittest discover -s code -p test_project.py -v
.\.venv\Scripts\python.exe code\check_method.py --checkpoint assets\final --output code\runs\method-check.json
```

训练输出目录必须尚不存在。拟合与标准化只使用训练集。记录的种子为 **20261024**，拟合采用确定性的岭回归闭式求解。选中的配置为 `embedding_surface_pairwise_a0.1_none`。在原环境中，两个 checkpoint 文件均可逐字节复现。所有学习得到的均值、缩放量、权重和偏置均包含在 checkpoint 中。按官方规则排除已有数据与普通源代码后，**额外推理资产解压后总计 25,530 字节**。

若需复现完整的固定验证研究，请在可丢弃的新克隆副本中执行：

```powershell
.\.venv\Scripts\python.exe code\analyze_data.py
.\.venv\Scripts\python.exe code\run_experiments.py
.\.venv\Scripts\python.exe code\analyze_results.py
```

研究包含 15 次拟合、33 个验证配置。脚本拒绝覆盖 `code/runs/validation-study-v1/`；新克隆副本中没有该目录。上述分析命令会重新生成 `records/analysis/` 和 `records/experiments/`，因此应在单独副本中运行，以保留已发布的原始证据。原始搜索总耗时为 36.8005 秒，单模型重训约 1.74 秒。计时及历史配置从 CRLF 转为 LF 的差异见 `docs/EXPERIMENTS.md`。

## 版本、checkpoint 与证据

方法在**测试评分之前**冻结于开发提交 `cf8e1ca64c8d51ae0edb04e276acebbb38c56398`。该 SHA 用于记录开发来源；本仓库中的方法及 checkpoint 与冻结时的逐文件哈希一致，复现无需开发仓库历史。

- `records/final/freeze.json` 保存原始代码／资产哈希、冻结时间与选择规则。
- `records/release/source-mapping.json` 核对发布的方法、训练代码、评分器、依赖和 checkpoint 与冻结哈希一致。
- 当前检出版本可通过 `git rev-parse HEAD` 获取；对应的模型资产位于该版本的 `assets/final/`。
- `records/final/test-score.json` 是原始官方评分输出；`records/final/evaluation.json` 保存原始命令与外部计时。
- `records/reproduction/` 和 `records/robustness/` 保存复现与输入接口检查证据。
- 历史记录可能包含原机器的绝对路径和日期，它们用于说明当时的运行情况，不是安装路径要求。实际复现使用本 README 的相对路径命令。

独立目录复现检查位于 `records/release/`；历史环境、训练及评测记录保留各自的日期和硬件信息。测试结果公开后，没有修改模型参数或选择规则。

## 工作分工与来源说明

Codex协助完成了作业要求核对、环境搭建、实验设计、代码、测试和报告草拟，学生完成了主要的实验设计、代码审阅、测试审阅和报告完善。

官方工具与数据保持原样。题目和参考答案来源于 GSM8K，保留其 [MIT 许可证](code/data/LICENSE.txt)及[来源归属说明](code/data/NOTICE.txt)。候选解答与表示由课程使用 Qwen3-0.6B 生成。本项目未使用额外模型权重或外部数据集。

官方来源：[作业指南](guide/GUIDE.md)、[上游仓库](https://github.com/xudongwu-0/DASE7506-MP2)、[官方初始运行说明](code/README.md)。
