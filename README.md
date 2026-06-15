# 双组分 Lennard-Jones 分子动力学模拟项目 (Kr-Xe)

## 项目介绍
本项目实现了一个从头编写的**双组分（Kr-Xe）Lennard-Jones 分子动力学 (MD) 模拟引擎**。本项目的物理目标包括对氪（Kr）和氙（Xe）混合气体系统的结构、热力学和动力学性质进行系统分析，构建相图，并最终将该基于经验势的模拟结果与 **M3GNet** 普适神经网络力场（NNFF）的计算结果进行横向比较。

---

## 环境配置
本项目使用 Python 3.10 构建。推荐使用 `uv` 或 `conda` 进行虚拟环境的管理与依赖安装。

### 使用 `uv` 安装（推荐）
```bash
# 在项目根目录下创建虚拟环境
uv venv .venv --python=3.10
# 激活环境
source .venv/bin/activate
# 安装所有依赖包
uv pip install numpy scipy matplotlib torch ase matgl pymatgen
```

### 使用 `conda` 安装
```bash
conda create -n lj_md python=3.10 numpy scipy matplotlib pytorch ase matgl pymatgen -c conda-forge
conda activate lj_md
```

---

## 任务进度清单

### ✅ 已完成的 Tasks (Task 1 - 3)
*   **Task 1: 构建核心 MD 引擎**
    *   实现了参数设置及 Lorentz-Berthelot 混合规则。
    *   实现了 FCC 晶格初始化与基于 Maxwell-Boltzmann 分布的速度初始化（带质心动量修正）。
    *   支持 **PyTorch Autograd**（自动微分）和 **NumPy 向量化** 两种后端进行解析或自动求导受力计算。
    *   实现了 Velocity-Verlet 积分器（微正则系综 NVE）与可逆的 Nosé-Hoover 温控积分器（正则系综 NVT）。
    *   设计了外部文件 `config.yaml` 灵活加载物理常数与计算基准。
*   **Task 2: MD 引擎严谨性验证**
    *   **NVE 能量守恒测试**：测试了 `dt=1fs` 与 `dt=2fs` 下的能量漂移率（Drift < $10^{-6}$），完全符合物理守恒定律。
    *   **NVT 温度平衡测试**：验证了体系起始状态能迅速平衡至目标温度（$120\text{ K}$），稳态偏差 < 0.02%。
*   **Task 3: 双组分生产级模拟运行**
    *   执行了标准配置（$N=500, x_B=0.5$）下，三种不同相互作用强度修正参数（$\xi = 0.7, 1.0, 1.3$）的高负荷 MD 生产模拟（单次 $50,000$ 步）。
    *   利用多进程并发执行完成了 $\xi=0.7$ 时的**浓度梯度系列扫描**（$x_B \in [0.1, 0.9]$），产出了 9 组 30,000 步的长期轨迹数据，为相图的构建做好了数据储备。

### ⏳ 待完成的 Tasks (Task 4 - 10)
*   **Task 4**: 编写 `src/analysis.py`，计算并绘制偏径向分布函数 $g(r)$ 及 Warren-Cowley 短程有序度参数。
*   **Task 5**: 计算各组分的均方位移 (MSD) 以及提取自扩散系数 $D$。
*   **Task 6**: 热力学分析（混合焓 $\Delta H_{\text{mix}}$、比热容 $C_v$ 及其涨落）。
*   **Task 7**: 轨迹可视化（导出 Ovito/ASE 支持的动画，观察相分离或有序化相变现象）。
*   **Task 8**: 相图构建（基于 Task 3 的数据分析 Kr-Xe 的临界相变及固-液/液-液相界）。
*   **Task 9 - 10**: M3GNet 神经网络力场推理引入与计算对比。
*   撰写自动生成的 LaTeX / Markdown 结题报告。

---

## 当前代码结构、功能与使用方法

```text
binary-lj-md/
├── config.yaml               # 全局物理常数与系统参数配置文件（推荐在此修改材料属性）
├── src/
│   ├── __init__.py
│   └── md_engine.py          # 核心 MD 引擎（包含受力、积分与运行总控逻辑）
├── tests/
│   ├── test_task1.py         # 核心力学及基础运行逻辑验证脚本
│   ├── test_task2.py         # NVE 能量守恒与 NVT 温度平稳验证（作图）
│   └── test_task3.py         # 高并发执行标准参数与浓度扫描的生产代码
├── data/
│   ├── trajectories/         # 保存的大量 .npy 模拟坐标轨迹文件
│   └── thermo/               # 保存的 .csv 热力学日志（时间步、动能、势能、压强等）
├── figures/                  # 测试输出的验证图表存放目录
└── README.md                 # 当前使用说明文档
```

### 使用方法

1.  **调整基础物理常数与物质**：
    若需修改 Lennard-Jones 常数（如质量、$\epsilon$ 或 $\sigma$），请直接编辑项目根目录下的 `config.yaml`。引擎会在运行时自动加载最新配置并计算出折合单位的基准值。
2.  **执行核心引擎验证**：
    在虚拟环境下执行 Task 2 测试文件即可验证物理合理性与引擎稳定性，生成相应的图表：
    ```bash
    python tests/test_task2.py
    ```
3.  **运行特定的生产模拟**：
    可参考或修改 `tests/test_task3.py` 中的 `config` 字典，调用 `src.md_engine` 中的 `run_md(config)` 进行特定条件的长时间跑算：
    ```python
    from src.md_engine import run_md
    
    config = dict(
        N_A=250, N_B=250, rho_star=0.8,
        T_equil_K=200.0, T_prod_K=150.0,
        dt_fs=2.0,
        n_equil=5000, n_prod=50000, n_save=50,
        ensemble_prod='NVT',
        xi=1.0,
        use_torch=False  # 生产模式推荐保持为 False 以使用 NumPy 解析求导加速计算
    )
    run_md(config)
    ```
