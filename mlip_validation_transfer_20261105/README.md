# MLIP 迁移工作区 / isolated MLIP continuation

本文件夹是独立研究分支的数据包，与仓库根目录的论文训练、模型、图表和固定参数结果分开。运行输出只写入本文件夹的 `runs/`。不要替换或覆盖上级目录资料。

## 现在要跑什么

Li₇Y₇ZrS₁₆（mp-754856）两条 31 原子、900 K 历史轨迹，各续至总长 50 ps。A 的冻结断点为 25.2 ps，B 为 20.0 ps；坐标、动量、晶胞、随机数状态及此前完整 20 fs 采样历史已包含。A/B 初始化历史不同，不能视为严格同协议独立种子。快照生成时间见 `snapshot.json`；原电脑仍可能继续计算，新电脑续算属于另一分支，不要拼接两台机器重叠时段。

| 项目 | 设置 |
|---|---|
| 势函数 | MACE-MPA-0 medium，权重 SHA256 校验 |
| 数值 | float32、1 fs、Langevin 900 K、摩擦 0.01/fs、FixCom |
| 晶胞 | 两条均固定 673.582180 Å³，原有热体积准备结果，非严格常压 |
| 保存 | 每 100 步断点，每 20 步轨迹，每 500 步分析 |
| 资源 | 顺序运行，默认 2 CPU 线程，上限 4；入场 4 GiB 空闲 RAM、运行保护 2 GiB |
| 时间 | 每次启动最长 8 小时硬上限；到时保留已完成断点，可再次运行续接 |

## 新电脑安装（Windows 或 Linux）

建议 Python 3.12。NVIDIA 显卡需兼容 CUDA 的 PyTorch；CPU 版也可使用。GPU 显存不足会记录错误并停止，不会暗中降级或丢失旧断点。

```bash
git clone https://github.com/LiuQuanneng/ASPHConductivity.git
cd ASPHConductivity/mlip_validation_transfer_20261105
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux: source .venv/bin/activate
python -m pip install --upgrade pip
# NVIDIA GPU（驱动兼容 CUDA 12.4 时）：
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# 无 NVIDIA GPU 时，用这一条替代上一条：
# python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/prepare_model.py
python scripts/verify.py
```

权重约 79 MB，通过官方地址下载，不纳入此 Git 文件夹。也可复制原电脑的 `mace-mpa-0-medium.model` 后执行 `python scripts/prepare_model.py --from-file /path/to/mace-mpa-0-medium.model`。脚本验证与原运行完全相同的权重哈希。无需 Materials Project API 密钥。

## 先检查，再续跑

```bash
# NVIDIA GPU：各跑 20 步测试，输出与正式运行隔离
python scripts/run.py --device cuda --smoke --hours 0.25
# 正式顺序续跑 A、B；时间上限可缩短
python scripts/run.py --device cuda --hours 8 --threads 2
# CPU：把 cuda 换成 cpu
# 单独跑一条：加 --case history_a 或 --case history_b
```

再次执行相同正式命令会从 `runs/production/` 的最新断点恢复；已完成的任务自动跳过。不要同时启动相同输出目录的两份队列。程序退出码非零时，先读该目录的日志及对应 `status.json`。`queue.json` 表示队列状态；`analysis.json` 提供 Li/框架多起点 MSD、逐 Li 分布及分块诊断，不生成正式电导率。

CPU→GPU 或软件环境变化不会保证逐位一致轨迹，虽然使用相同结构、动量及随机数状态。需要保留新环境与设备信息，并把迁移后的结果视为对应数值后端的续算；本包已在原电脑做 CPU 短测试，GPU/Linux 需在目标机器运行上述测试。

## 下一步研究顺序

1. 完成 A/B 至 50 ps，比较相同剔除时间和滞后窗口的 Li/框架 MSD、逐 Li 事件、压力、温度、分块稳定性。
2. 若仍为平台或由单个跳跃主导，记录这一结果，不拟合正式 D 或电导率。若持续扩散且框架稳定，再明确同协议独立种子、有限尺寸和温度检查；这些计算未在本包中自动排队。
3. 保留 124 原子和已知 Li₃YBr₆ 对照的既有诊断，见研究记录。模型力准确性、相/缺陷匹配仍需解决；两势差异不等于 DFT 误差。
4. 其他条件候选为 K₂Li₁₄Zr₃O₁₄、Na₂Li₃GaO₄、Na₃LiWO₅，原结构已附。先复核精确相的定量输运先例和结构通道，再决定是否做新筛查；已有分类预测，不能称首次预测。其余文献依据见 `research/additional_candidates.txt`。

## 内容与来源

- `inputs/`：可续跑冻结快照和完整历史；不要修改。
- `structures/`：MP 来源的原始 CIF；实际续算使用 inputs 中热体积结构。
- `research/`：已有结果、来源链接、MSD 源数据与图、候选复核和未解决问题。报告中旧的 PID、时限和本机路径仅是历史记录，不是新电脑启动设置。
- `provenance/`：原 Windows worker 和历史 CPU 环境清单，作为追溯资料，非新电脑启动入口。
- `snapshot.json`、`manifest.json`：冻结时间、断点来源和文件 SHA256。

排名仍以仓库 `publication_fixed_S1/data/candidate_predictions_fixed_S1.csv` 为唯一来源。这里的 MLIP 结果尚未验证高电导新材料，不替代论文回归预测。

权重来源：https://github.com/ACEsuit/mace-mp/releases/tag/mace_mpa_0 。结构来源：https://materialsproject.org/ 。本包不重新分发文献全文或模型权重；相关使用条件以原来源为准。
