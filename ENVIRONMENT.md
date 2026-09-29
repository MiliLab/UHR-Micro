# 环境安装说明

本仓库包含三类推理/评测，它们的依赖互不兼容（尤其是 torch / transformers /
vLLM 版本），因此拆分为**三个独立的 Python 环境**。冒烟测试验证通过的精确版本如下。

## 硬件与驱动

- 至少 1 块 NVIDIA GPU（推荐 ≥ 80 GB 显存，如 A100/H100）。
- CUDA 驱动版本 ≥ 13.0（本机为 580.65.06，CUDA 13.0）。
- 三个环境各自编译 torch 的 CUDA 版本不同：Qwen 用 cu128，InternVL/SAM2 用 cu130。

## 环境 A：`qwen3vl` —— Qwen3-VL direct + MAP 方法（Qwen 骨干）

覆盖 `direct/` 与 `map/qwen3vl/` 的全部 CPU + GPU 阶段。

```bash
conda create -n qwen3vl python=3.10 -y
conda activate qwen3vl
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/qwen3vl.txt
```

关键包：

| 包 | 版本 |
|---|---|
| torch | 2.10.0+cu128 |
| torchvision | 0.25.0 |
| transformers | 5.5.3 |
| vllm | 0.19.1 |
| numpy / pillow | 2.2.6 / 11.3.0 |
| accelerate | 1.13.0 |
| qwen-vl-utils | 0.0.14 |

## 环境 B：`internvl` —— MAP 方法（InternVL3.5-8B 骨干）

覆盖 `map/internvl/`。InternVL3.5 走 transformers 后端（`trust_remote_code=True`），
因此还需要模型的参考代码目录（见下文「模型与参考代码」）。

```bash
conda create -n internvl python=3.10 -y
conda activate internvl
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements/internvl.txt
```

关键包：

| 包 | 版本 |
|---|---|
| torch | 2.11.0+cu130 |
| torchvision | 0.26.0 |
| transformers | 5.13.1 |
| timm / einops / decord | 0.9.12 / 0.6.1 / 0.6.0 |

## 环境 C：`sam2` —— 分割评测（SAM2）

仅用于 `Understanding/ReferSegmentation` 与 `Understanding/ComponentSegmentation`
两类任务的 bbox→mask 评测。

```bash
conda create -n sam2 python=3.10 -y
conda activate sam2
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements/sam2.txt
# 克隆 SAM2 仓库并安装（评测运行时通过 MAP_SAM2_ROOT 指向它）
git clone https://github.com/facebookresearch/sam2.git
```

关键包：torch 2.11.0+cu130、torchvision 0.26.0、numpy 2.2.6、pillow 12.2.0、pycocotools 2.0.11。

## 模型 checkpoint

把模型权重放到仓库根目录的 `models/` 下（或用软链接指向实际位置），目录结构须与
`evaluation/configs/model_registry.json` 中 `artifact_root_rel_workspace` 一致：

```
models/
├── Qwen3-VL-main/
│   └── Qwen3-VL-8B-Instruct/
└── InternVL3_5-8B/
```

运行时也可用环境变量覆盖默认路径：

- `MAP_MODEL_PATH` —— Qwen3-VL-8B 权重目录
- `MAP_INTERNVL_MODEL_PATH` —— InternVL3.5-8B 权重目录

## 数据集

UHR-Bench v4.1 数据集（含 `Datasource/` 图像与 QA 标注）通过环境变量 `MAP_DATASET`
指定。仓库本身**不包含**数据集与答案，需单独获取。

## InternVL 参考代码

InternVL3.5-8B 的推理依赖 `trust_remote_code`，需要模型自带的 Python 参考代码。
将模型仓库里的推理代码目录通过 `MAP_INTERNVL_REF_DIR` 指定（transformers 加载时
会把它加入 `sys.path`）。

## 运行前自检

```bash
# 检查 Qwen 环境能加载模型
python -c "import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__)"

# direct 推理 dry-run（不加载模型，仅校验 manifest/配置）
python -m direct.cli run --input-manifest <manifest> --run-config <run_config> \
  --adapter-config direct/configs/qwen3_vl_8b.json --model-path $MAP_MODEL_PATH \
  --image-root $MAP_DATASET/Datasource --gpus 0 --dry-run
```

其余环境变量说明见 `README.md` 的「Environment variables」一节。
