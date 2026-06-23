# Binary Component Lennard-Jones MD Simulation (Kr-Xe)

## 项目简介
本项目实现了一个双组分（Kr-Xe）Lennard-Jones (LJ) 分子动力学模拟引擎，包含完整的物理法则模拟、热力学与动力学性质计算、大批量网格扫描调度以及相图构建。在此基础上，项目还通过 Atomic Simulation Environment (ASE) 接入了 M3GNet 深度学习势函数，实现了传统经验力场与现代神经网络力场在精度与性能上的比对。

---

## 环境配置

本项目需要 **Python 3.11** 及以上版本。由于依赖深度学习库及高性能计算包，建议使用虚拟环境进行管理。

### 依赖安装 (基于 `uv`)
```bash
# 创建虚拟环境
uv venv .venv --python=3.11.7
# 激活环境
source .venv/bin/activate
# 安装依赖
uv pip install numpy scipy pandas matplotlib torch ase matgl pymatgen
```

---

## 项目结构与功能模块

### 目录结构
```text
binary-lj-md/
├── src/                      # 核心代码库
│   ├── __init__.py
│   ├── md_engine.py          # 基础动力学积分与力场引擎
│   ├── analysis.py           # 数据分析工具集
│   ├── phase_diagram.py      # 参数网格遍历与相图判定模块
│   └── nnff_compare.py       # 神经网络力场推理与性能对比模块
├── tests/                    # 测试与任务启动脚本
│   ├── test_task1.py ... test_task9.py
├── data/                     # 运行生成的数据缓存
│   ├── trajectories/         # 坐标轨迹数据 (.npy)
│   ├── thermo/               # 热力学日志 (.csv)
│   └── phase_diagram/        # 相图记录表
├── figures/                  # 各任务生成的可视化图表
└── config.yaml               # 物理参数全局配置文件
```

### 核心接口说明 (API)

*   **`src.md_engine.run_md(config)`**:
    系统的核心模拟入口。接收一个 `dict` 形式的 `config`，控制系统原子数、温度、时长、系综 (NVE/NVT) 等。输出包含坐标轨迹和能量历史的字典。支持 Numpy 和 PyTorch 双计算后端。
*   **`src.analysis.compute_partial_rdf(...)`**:
    计算双组分系统的偏径向分布函数 $g_{AA}(r), g_{BB}(r), g_{AB}(r)$。
*   **`src.analysis.compute_warren_cowley(...)`**:
    计算混合物的 Warren-Cowley 短程有序度参数 $\alpha_1(t)$，用于判定系统是倾向于相分离（$\alpha_1 > 0$）还是长程有序（$\alpha_1 < 0$）。
*   **`src.analysis.compute_msd_diffusion(...)`**:
    计算均方位移 (MSD) 并拟合自扩散系数 $D$。
*   **`src.analysis.compute_structure_factor(...)`**:
    通过原子坐标傅里叶变换计算系统的静态结构因子 $S(q)$。

---

## 使用方法

1.  **参数配置**：修改根目录下的 `config.yaml` 改变全局的 Kr/Xe 分子质量、$\varepsilon$、$\sigma$ 参数。
2.  **执行单项任务**：激活虚拟环境后，进入根目录，执行特定的任务测试文件。例如运行 Task 2 热力学验证：
    ```bash
    python tests/test_task2.py
    ```

---

## 任务进度与输出清单

### Task 1: 构建核心 MD 引擎 (✅ 已完成)
*   **工作内容**：基于 Python 构建了 Velocity-Verlet 积分器及 Nosé-Hoover 温控。实现了 Lorentz-Berthelot 混合法则，支持 NumPy 向量化与 PyTorch 计算图两种后端。
*   **输出结果**：可运行的基础 `run_md` 模块。

### Task 2: 引擎严谨性验证 (✅ 已完成)
*   **工作内容**：测试纯 Kr 体系在 NVE 系综下的能量守恒性质及 NVT 系综下的温度弛豫曲线。
*   **输出结果**：`figures/task2_NVE_energy.png`, `figures/task2_NVT_energy.png`。

### Task 3: 生产级模拟与基准数据获取 (✅ 已完成)
*   **工作内容**：在不同相互作用强度（$\xi = 0.7, 1.0, 1.3$）下完成了长时生产系综计算，生成了供后续分析的轨迹序列。
*   **输出结果**：存放在 `data/trajectories/` 中的基准轨迹数据。

### Task 4: 偏径向分布函数与短程有序度 (✅ 已完成)
*   **工作内容**：从轨迹数据中计算出了偏 $g(r)$ 和短程有序度参数 $\alpha_1$。
*   **输出结果**：`figures/task4_rdf_comparison.png`, `figures/task4_warren_cowley_timeseries.png`。

### Task 5: 均方位移与自扩散系数 (✅ 已完成)
*   **工作内容**：分析原子轨迹，计算长时 MSD，并基于爱因斯坦关系式拟合自扩散系数。
*   **输出结果**：`figures/task5_msd_diffusion.png`。

### Task 6: 混合焓与热力学属性 (✅ 已完成)
*   **工作内容**：计算不同组分浓度下的混合焓 $\Delta H_{\text{mix}}$。
*   **输出结果**：`figures/task6_mixing_enthalpy.png`。

### Task 7: 结构因子 (✅ 已完成)
*   **工作内容**：将实空间坐标变换至倒空间，计算 $S(q)$ 观察晶格周期性及超晶格峰。
*   **输出结果**：`figures/task7_structure_factor.png`。

### Task 8: T-x 相图构建 (⏳ 计算中)
*   **工作内容**：遍历 $T \in [80, 200]\text{K}$, $x_B \in [0.1, 0.9]$ 网格，长时（10万步）计算以寻找玻璃相变点及相分离边界。
*   **运行状态**：当前测试脚本 `test_task8.py` 正在后台执行长时网格遍历任务。
*   **预期输出**：`data/phase_diagram/phase_grid_long.csv` 以及最终的可视化相图 `figures/task8_phase_diagram_Tx.png`。

### Task 9: M3GNet 神经网络力场对比 (✅ 已完成)
*   **工作内容**：引入 `M3GNet-PES-MatPES-PBE-2025.2` 通用模型，针对经典 LJ 势进行了单点能量计算精度、二聚体曲线、ASE 分子动力学试运行的评价。并进行了计算量随原子数规模扩展的 Benchmark。
*   **输出结果**：
    *   二聚体曲线对比：`figures/task9_pes_dimer_comparison.png`
    *   径向分布函数对比：`figures/task9_rdf_lj_vs_m3gnet.png`
    *   耗时 Benchmark（测试规模延伸至 N=5000）：`figures/task9_performance_benchmark.png`
