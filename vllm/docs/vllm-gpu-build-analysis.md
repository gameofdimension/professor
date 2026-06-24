# vLLM GPU 构建过程分析报告

> 调查对象：本仓库（`vllm-sm120`）的 GPU（NVIDIA CUDA）构建过程。
>
> 基线：`vllm-project/vllm`，`origin` 指向官方仓库。
>
> 分析所基于的 commit：**`accaa434f36b37a35b3e68eede167415ecc83c51`**（短 `accaa434f`，`git describe = v0.23.1rc0-307-gaccaa434f`，日期 2026-06-23，"[Rust Frontend] Support echo for token-ID completion prompts (#46219)"）。下文所有 `file:line` 均对应该 commit 的工作树状态。
>
> 方法：全部结论均经源码 / 编译脚本 / CI 脚本 / Dockerfile 直接核对（附 `file:line` 证据），未依赖网络搜索，避免幻觉。

---

## 目录

1. [仓库与 sm_120 定位](#1-仓库与-sm_120-定位)
2. [GPU 构建整体流程](#2-gpu-构建整体流程)
3. [需要安装 / 编译的依赖](#3-需要安装--编译的依赖)
4. [vLLM 本身编译的 C++/CUDA 源码](#4-vllm-本身编译的-ccuda-源码)
5. [wheel 中包含的 Python source / 产物](#5-wheel-中包含的-python-source--产物)
6. [CI / Docker 的实际 GPU 构建命令](#6-ci--docker-的实际-gpu-构建命令)
7. [预编译机制详解：`VLLM_USE_PRECOMPILED`](#7-预编译机制详解vllm_use_precompiled)
8. [附录：关键证据汇总](#8-附录关键证据汇总)

---

## 1. 仓库与 sm_120 定位

- 本仓库 = 上游 `vllm-project/vllm`（`origin`），fork 远程 `gameofdimension/vllm`（`god`）。
- **`sm120/` 目录不在调查范围内**（仅含本地 `bench/` 草稿，git 未跟踪，非官方代码）。
- **关于 sm_120：仓库本身没有任何 fork 专属的构建改动。** sm_120 支持全部是上游代码，通过 CUDA 13 的 family 目标 `12.0f`（或 CUDA 12.8/12.9 的 `12.0a;12.1a`）实现。fork 的实际差异在模型 / 运行时代码（GLM-5.x、Rust 前端等），不在 CUDA 工具链。

---

## 2. GPU 构建整体流程

构建入口为 `setup.py`。所有 C++/CUDA 扩展声明为 `CMakeExtension`（`setup.py:184-188`），由自定义命令 `cmake_build_ext`（`setup.py:190-405`）委托 CMake / Ninja 完成。三步：

| 步骤 | 函数 | 实际命令 | 证据 |
|---|---|---|---|
| 1. configure | `configure()` | `cmake <src> -G Ninja -DVLLM_TARGET_DEVICE=cuda -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc -DFETCHCONTENT_BASE_DIR=.deps …` | `setup.py:239-322` |
| 2. build | `build_extensions()` | `cmake --build . -j --target=<每个扩展>`，per-file gencode | `setup.py:324-354` |
| 3. install | 同上 | `cmake --install . --prefix <outdir> --component <target>`，`.so` 落到 `vllm/` 包根 | `setup.py:356-381` |

**设备探测**（`setup.py:81-103`）：`torch.version.cuda` 非空 → `VLLM_TARGET_DEVICE=cuda`；`torch.version.hip` → `rocm`；否则 `cpu`。
`_is_cuda()` = cuda 目标 + 有 cuda + 非 tpu（`setup.py:927-929`）。

`CMakeLists.txt`（60KB）在 `find_package(Torch REQUIRED)` 后按 NVCC 版本决定支持的架构族（见 §6.3）。

---

## 3. 需要安装 / 编译的依赖

### 3.1 Python 构建期依赖（`pyproject.toml:3-13`，镜像于 `requirements/build/cuda.txt`）

- `cmake>=3.26.1`
- `ninja`
- `packaging>=24.2`
- `setuptools>=77.0.3,<81.0.0`
- `setuptools-scm>=8.0`（从 git tag 推导动态版本）
- **`setuptools-rust>=1.9.0`**（编译 Rust 扩展）
- **`torch==2.11.0`** —— torch 同时是构建依赖（setup.py 需 `torch.utils.cpp_extension` 定位 CUDA 与编译器标志），故 `pyproject.toml:191` 标记 `no-build-isolation-package`
- `wheel`、`jinja2`

### 3.2 Python 运行期依赖（动态加载，`setup.py:1047-1094` `get_requirements()`）

按目标设备读 `requirements/<device>.txt`，CUDA 读 `requirements/cuda.txt`（先 `-r common.txt`）。关键项：

- **`torch==2.11.0` / `torchaudio==2.11.0` / `torchvision==0.26.0`** 三者锁版本
- `transformers>=5.5.3`、`tokenizers`、`safetensors>=0.6.2`、`compressed-tensors==0.17.0`
- `flashinfer-python==0.6.12` + `flashinfer-cubin==0.6.12`
- `nvidia-cutlass-dsl[cu13]==4.5.2`、`quack-kernels>=0.3.3`、`humming-kernels[cu13]==0.1.6`
  - CUDA 12 时 `setup.py:1077-1081` 把 `[cu13]` 降级为 `[cu12]`/裸名
- 注意：**不直接依赖 `flash-attn` / `xformers`** —— vLLM 自带 flash-attention fork（见 §3.3）

### 3.3 CMake 拉取并编译的外部 C++/CUDA 工程（`cmake/external_projects/`）

| 工程 | 仓库 / 锁定版本 | 编译方式 | 证据 |
|---|---|---|---|
| **CUTLASS** | `nvidia/cutlass` **`v4.4.2`** | **header-only** | `CMakeLists.txt:389,405` |
| **vLLM flash-attention** (FA2/FA3) | `vllm-project/flash-attention` commit **`803020a8…`** | 编译为 `.so` | `vllm_flash_attn.cmake:41-42` |
| **DeepGEMM** | `deepseek-ai/DeepGEMM` commit **`891d57b4…`** | 编译（per-CPython `.so`，vendored 进 `vllm/third_party/deep_gemm`） | `deepgemm.cmake:31-32` |
| **QuTLASS** | `IST-DASLab/qutlass` commit **`830d2c45…`** | 编译（SM100/SM120） | `qutlass.cmake:24-25` |
| **FlashMLA** | `vllm-project/FlashMLA` commit **`a6ec2ba7…`** | 编译（Hopper/Blackwell） | `flashmla.cmake:21-22` |
| **fmha_sm100 (MSA)** | `vllm-project/MSA` commit **`544eee5e…`** | **纯 Python**（拷贝 `.py`） | `fmha_sm100.cmake:19-20` |
| **triton_kernels** | `triton-lang/triton` tag **`v3.5.1`** | **纯 Python** | `triton_kernels.cmake:3,21` |

每个 `.cmake` 支持 `<PROJ>_SRC_DIR` 环境变量覆盖为本地路径以便开发。下载缓存统一到 `$ROOT/.deps`（`FETCHCONTENT_BASE_DIR`，`setup.py:288-290`）。

### 3.4 系统 / 工具链（CI/Docker 安装，不在 requirements/ 里）

| 工具 | 安装方式 | 证据 |
|---|---|---|
| **CUDA toolkit**（devel） | 基础镜像 `nvidia/cuda:13.0.2-devel-ubuntu22.04` | `Dockerfile:25,40` |
| **GCC ≥ 11.3** | apt 装 `gcc-11/g++-11` + `update-alternatives` | `Dockerfile:155-157`，`CMakeLists.txt:41-47` |
| C++20 / CUDA 20 标准 | CMake 设置 | `CMakeLists.txt:33-39` |
| **uv** | `curl … astral.sh/uv/install.sh` | `Dockerfile:169` |
| ccache / sccache（可选） | apt / GitHub release | `Dockerfile:134,413-417` |
| **protoc** | `tools/install_protoc.sh`（Rust gRPC 用） | `Dockerfile:291-292` |
| **Rust 1.95** | `rust-toolchain.toml` + `requirements/build/rust.txt` | `rust-toolchain.toml` |

---

## 4. vLLM 本身编译的 C++/CUDA 源码

源码在 `csrc/`（约 90 个 `.cu` + 33 个 `.cpp` + 大量头文件），按 ABI 世界分两套：

### 4.1 主力：libtorch stable-ABI（`csrc/libtorch_stable/`，~151 文件）

所有主 CUDA kernel 编进两个扩展（`setup.py:1156-1158`，CUDA/HIP 都建）：

- **`vllm._C_stable_libtorch`** → `vllm/_C_stable_libtorch.abi3.so`
  - 入口 `csrc/libtorch_stable/torch_bindings.cpp`，用 `STABLE_TORCH_LIBRARY_FRAGMENT(_C, ...)` 注册 `torch.ops._C.*`
  - 含 activations、cache_ops、rmsnorm/quant、RoPE、CUTLASS scaled_mm、Marlin/Machete/GPTQ/AWQ、sampler/topk、custom all-reduce、paged_attention v1/v2、mamba、DeepSeek-v4/MiniMax-M3 融合 kernel 等 ~100+ 个 op
- **`vllm._moe_C_stable_libtorch`** → `vllm/_moe_C_stable_libtorch.abi3.so`
  - 入口 `csrc/libtorch_stable/moe/torch_bindings.cpp`，注册 `torch.ops._moe_C.*`（topk_softmax、grouped_topk、moe_align、wna16 等）

stable-ABI 扩展按 `TORCH_TARGET_VERSION=0x020B000000000000ULL`（PyTorch 2.11）编译，与未来 torch 版本解耦（`CMakeLists.txt:1049-1050`）。

### 4.2 旧 unstable-ABI `_C`（`csrc/torch_bindings.cpp`）

**CUDA 已不再构建 `_C`** —— `setup.py:1153-1155` 只在 `_is_hip()`（ROCm）时 `append CMakeExtension("vllm._C")`；CPU 另有 `_C/_C_AVX512/_C_AVX2`。CUDA 路径只走 stable 变体。

### 4.3 独立小扩展（CUDA 必建）

- `vllm.cumem_allocator`（`csrc/cumem_allocator.cpp`，NVIDIA `cuMem*` 虚拟内存 API）→ `vllm/cumem_allocator.abi3.so`
- `vllm.spinloop`（`csrc/spinloop.cpp`，MWAITX/UMONITOR 自旋等待，仅 py≥3.11）→ `vllm/spinloop.abi3.so`

### 4.4 外部工程产物（由 §3.3 编译，setup.py 按条件声明）

flash-attn `_vllm_fa2_C`/`_vllm_fa3_C`、FlashMLA `_flashmla_C`/`_flashmla_extension_C`（CUDA≥12.9）、QuTLASS `_qutlass_C`、DeepGEMM `_deep_gemm_C`（CUDA≥12.3）。

### 4.5 sm_120 专属 kernel（上游代码，编译时按 arch 启用）

在 `CMakeLists.txt` 里按架构族条件编译，CUDA≥13 用 family `12.0f`（一个 cubin 覆盖整个 SM12x 家族），CUDA 12.8/12.9 用 `12.0a;12.1a`：

- **SM120 scaled_mm**（w8a8 FP8 CUTLASS c3x）：源 `scaled_mm_c3x_sm120.cu` 等，定义 `-DENABLE_SCALED_MM_SM120=1`（`CMakeLists.txt:728-745`）
- **SM120 NVFP4 / MoE**：`nvfp4_scaled_mm_sm120_kernels.cu` 等，定义 `-DENABLE_NVFP4_SM120=1`、`-DENABLE_CUTLASS_MOE_SM120=1`（`CMakeLists.txt:902-919`）
- Marlin、DSV3 fused GEMM 等也含 `12.0f` 分支（`CMakeLists.txt:506-526,642,871,1126`）

### 4.6 Rust 前端（`rust/`，独立于 CUDA，用 setuptools-rust 构建）

`tools/build_rust.py` 定义两个 `RustExtension`（`setup.py:1224`）：

- `vllm.vllm-rs` → 独立可执行文件 `vllm/vllm-rs`（`Binding.Exec`，OpenAI 兼容 HTTP 服务的子进程，由 `vllm/v1/utils.py` 按路径拉起）
- `vllm._rust_tool_parser` → PyO3 扩展 `vllm/_rust_tool_parser.abi3.so`（tool-call 解析器）

工具链锁定 Rust 1.95（`rust-toolchain.toml`），由 `build_rust.sh` → `python tools/build_rust.py` 驱动（`Dockerfile:316`）。

---

## 5. wheel 中包含的 Python source / 产物

### 5.1 包发现

`pyproject.toml:41-43`：`[tool.setuptools.packages.find]` `include = ["vllm*"]`，即整个 `vllm/` 目录 + vendored 包。顶层子包（~30 个）：

`vllm/v1`、`vllm/model_executor`、`vllm/attention`、`vllm/distributed`、`vllm/platforms`、`vllm/worker`、`vllm/entrypoints`、`vllm/engine`、`vllm/compilation`、`vllm/lora`、`vllm/config`、`vllm/multimodal`、`vllm/transformers_utils`、`vllm/device_allocator`、`vllm/tool_parsers`、`vllm/tokenizers`、`vllm/kernels`、`vllm/vllm_flash_attn/`、`vllm/third_party/{deep_gemm,flashmla,triton_kernels}` 等。

### 5.2 编译产物 `.so`（全部装到 `vllm/` 包根，`.abi3` 后缀来自 Python stable ABI）

- `vllm/_C_stable_libtorch.abi3.so`
- `vllm/_moe_C_stable_libtorch.abi3.so`
- `vllm/cumem_allocator.abi3.so`
- `vllm/spinloop.abi3.so`
- `vllm/vllm_flash_attn/_vllm_fa{2,3}_C.abi3.so`
- （可选）`vllm/_flashmla_C.abi3.so`、`vllm/_qutlass_C.abi3.so`
- `vllm/_rust_tool_parser.abi3.so`（PyO3）
- `vllm/vllm-rs`（可执行文件）
- ROCm 多一个 `vllm/_rocm_C.abi3.so`

Python 侧通过副作用 import 触发 op 注册：`import vllm._C_stable_libtorch`（`vllm/platforms/cuda.py`）、`import vllm._moe_C_stable_libtorch`，再由 `vllm/_custom_ops.py` 统一调用 `torch.ops._C.*`。**无 `.pyi` 存根**。

### 5.3 非 Python 数据文件（`setup.py:1160-1176` `package_data`）

- `py.typed`
- `model_executor/layers/fused_moe/configs/*.json`（326 个 MoE 调优配置）
- `model_executor/layers/quantization/utils/configs/*.json`
- `entrypoints/serve/instrumentator/static/*.js`、`*.css`（swagger-ui）
- `distributed/kv_transfer/kv_connector/v1/hf3fs/utils/*.cpp`
- DeepGEMM JIT 头文件 `third_party/deep_gemm/include/**/*.{cuh,h,hpp}`
- fmha_sm100 helper `*.cu`
- `vllm/libs/*.so*`（CPU 版打包的 tcmalloc）

### 5.4 入口点（`pyproject.toml:43-48`）

- console script：`vllm = "vllm.entrypoints.cli.main:main"`
- plugin：`vllm.general_plugins` 组下的 LoRA resolver

### 5.5 `MANIFEST.in`（管 sdist，非 wheel）

含 `LICENSE`、`requirements/*.txt`、`CMakeLists.txt`、`tools/build_rust.py`，`recursive-include cmake *`、`recursive-include csrc *`（保证源码分发可从源重建）。

---

## 6. CI / Docker 的实际 GPU 构建命令

### 6.1 Dockerfile 多阶段（`docker/Dockerfile`）

- `csrc-build` 阶段做**真正的重编译**：
  ```
  python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38
  ```
  （`Dockerfile:430`）。`--py-limited-api=cp38` → 产物是 stable-ABI wheel（cp38+ abi3）。
- `build` 阶段**复用** csrc 产物而非重编：
  ```
  VLLM_USE_PRECOMPILED=1
  VLLM_PRECOMPILED_WHEEL_LOCATION=$(ls /precompiled-wheels/*.whl)
  python3 setup.py bdist_wheel …
  ```
  （`Dockerfile:549-552`）。`VLLM_USE_PRECOMPILED` 触发 `precompiled_build_ext`（`setup.py:449-457`，空操作）+ 从 wheel 抽取 `.so`/rust 打进新 wheel（`setup.py:736-852`）。
- `rust-build` 阶段：`bash build_rust.sh`（`Dockerfile:316`）。
- 基础镜像 `nvidia/cuda:13.0.2-devel-ubuntu22.04`（`Dockerfile:40`）。

### 6.2 本地开发（`AGENTS.md`）

- 跳过编译用官方预编译 nightly wheel：
  ```
  VLLM_USE_PRECOMPILED=1 uv pip install -e .
  ```
  （从 `wheels.vllm.ai` 按 commit + CUDA variant 下载）
- 从源编译：
  ```
  uv pip install -e . --torch-backend=auto
  ```

### 6.3 架构矩阵与 sm_120（`.buildkite/release-pipeline.yaml`）

CMake 按版本定义 `CUDA_SUPPORTED_ARCHS`（`CMakeLists.txt:107-118`）：

| NVCC 版本 | 支持架构 |
|---|---|
| ≥ 13.0 | `7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0`（用 family 目标 `10.0f`/`12.0f`） |
| ≥ 12.8 且 < 13 | `7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;12.0;12.1`（arch-specific `a` 后缀） |
| 更早 | `7.0;7.5;8.0;8.6;8.7;8.9;9.0` |

Release wheel 的实际 gencode 矩阵（`release-pipeline.yaml:10-18`）：

| Job | CUDA | arch 列表 |
|---|---|---|
| x86 release wheel | 13.0.2 | `7.5 8.0 8.6 8.9 9.0 10.0` **`12.0`** |
| x86 release wheel | 12.9.1 | `7.5 8.0 8.6 8.9 9.0 10.0` **`12.0`** |
| aarch64 release wheel | 13.0.2 | `8.0 8.7 8.9 9.0 10.0 11.0` **`12.0`** |
| arm64 CI image | 13.0.2 | `9.0` **`12.0`**（覆盖 GB10/sm_121） |

> **`12.0` 即 sm_120**（Blackwell）。注意 x86 wheel 列表**不含 11.0**（为控制 wheel 体积 <500MB，`Dockerfile:571` 用 `.buildkite/check-wheel-size.py` 检查）。架构经 `TORCH_CUDA_ARCH_LIST`/`torch_cuda_arch_list` 传入，CMake 在 `cmake/utils.cmake` 解析成 per-file gencode。

### 6.4 关于本仓库的 sm_120 性质

**无任何 fork 专属构建改动** —— 所有 `12.0f`/`12.0a`/`ENABLE_*_SM120` 都是上游 vLLM 代码。仓库的 sm_120 定位完全体现在"用上游标准 CUDA 13 family 目标 `12.0` 编译"上。运行时侧确有 sm_120 专属路径（如 FlashInfer sparse MLA / MoE 的 `launch_sm120_*`，见 `tests/.../test_flashinfer_b12x_moe.py`），但那是 Python 调度逻辑，不属本构建调查范畴。

---

## 7. 预编译机制详解：`VLLM_USE_PRECOMPILED`

**一句话**：设了它就跳过所有 C++/CUDA（和 Rust）本地编译，转而从官方预编译 wheel 中抽取现成的 `.so` 直接打包，把"几小时的 nvcc 编译"变成"下载 + 解压"。

### 7.1 定义（`vllm/envs.py:594-596`）

```python
"VLLM_USE_PRECOMPILED": lambda: (
    os.environ.get("VLLM_USE_PRECOMPILED", "").strip().lower() in ("1", "true")
    or bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION"))  # 注意这条
)
```

> 注意：**只设 `VLLM_PRECOMPILED_WHEEL_LOCATION`（即使不显式设 `=1`）也会隐式打开此开关**。
> 关联开关：`VLLM_USE_PRECOMPILED_RUST`（`envs.py:599-600`）、`VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX`（`envs.py:603-604`）。

### 7.2 触发的 4 个行为

| # | 行为 | 证据 |
|---|---|---|
| 1 | **跳过 C++/CUDA 编译**：`build_ext` 换成 `precompiled_build_ext`（空操作，仅打印 `"Skipping build_ext: using precompiled extensions."`） | `setup.py:1210-1213`、`setup.py:449-457` |
| 2 | **跳过 Rust 编译**：派生 `USE_PRECOMPILED_RUST_FRONTEND = VLLM_USE_PRECOMPILED or VLLM_USE_PRECOMPILED_RUST`，`build_rust` → `precompiled_build_rust`（产物齐全则跳过，否则回退本地编译） | `setup.py:52-54`、`setup.py:1216-1220`、`setup.py:460-487` |
| 3 | **下载预编译 wheel 并抽取产物**：`determine_wheel_url()` + `extract_precompiled_and_patch_package()`，把现成 `.so`/rust/vendored `.py` 解到源码树并 patch `package_data` | `setup.py:1186-1195`、`setup.py:737-852` |
| 4 | **版本号加 `+precompiled` 后缀**（除非 `VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1`） | `setup.py:1018-1019` |

> 补充：setup.py 里"按 CUDA 版本条件声明"的可选扩展（FA3 / FlashMLA / DeepGEMM / QuTLASS）判断为 `if USE_PRECOMPILED_EXTENSIONS or (CUDA_HOME and nvcc>=X)`（`setup.py:1113,1123,1133`）——预编译模式下它们恒被声明，但因 `build_ext` 是空操作，只参与抽取/打包逻辑，并不会被编译。

### 7.3 抽取哪些文件（`setup.py:762-832`）

抽取分**两条路径**：

**路径 A — 固定 `.so` 清单（`exact_members`，`setup.py:767-783`）**：从 wheel 根目录精确抽取这组 `.abi3.so`：

```
vllm/_C.abi3.so
vllm/_C_stable_libtorch.abi3.so
vllm/_moe_C_stable_libtorch.abi3.so
vllm/_qutlass_C.abi3.so
vllm/_flashmla_C.abi3.so / _flashmla_extension_C.abi3.so / _sparse_flashmla_C.abi3.so
vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so / _vllm_fa3_C.abi3.so
vllm/cumem_allocator.abi3.so
vllm/spinloop.abi3.so
vllm/_rocm_C.abi3.so   (ROCm)
```

> 注意：**DeepGEMM 不在此清单中**——它的 `.so` 不在包根目录，而是 vendored 在 `third_party/deep_gemm/` 下。

**路径 B — 正则匹配 vendored 目录（`setup.py:787-832`）**：按目录全量抽取，各目录匹配范围不同：

| 目录 | 正则 | 抽取范围 |
|---|---|---|
| `vllm/vllm_flash_attn/**` | 仅 `*.py`（跳过 `__init__.py`/`flash_attn_interface.py`） | 纯 Python（其 `.so` 走路径 A） |
| `vllm/third_party/triton_kernels/**` | 仅 `*.py` | 纯 Python |
| `vllm/third_party/flashmla/**` | 仅 `*.py` | 纯 Python（其 `.so` 走路径 A） |
| `vllm/third_party/deep_gemm/**` | **`.*` 全部文件** | **含编译产物 `_C.cpython-*.so`** + `.py/.cuh/.h/.hpp` |
| `vllm/third_party/fmha_sm100/**` | `.*` 全部文件 | `.py` + `.cu` helper（无 `.so`） |

**所以 DeepGEMM 的 `.so` 会被抽取**，只是走路径 B 的 `deep_gemm_regex`（`setup.py:802-803`，注释明确"extract all files (.py, .so, .cuh, .h, .hpp, etc.)"），而非根目录固定清单。DeepGEMM 的 `.so` 是 **per-CPython** 的 `_C.cpython-*.so`（非 abi3），因此以 vendored 包形式放在 `third_party/deep_gemm/`，并按 CPython 解释器版本分别构建（`Dockerfile:396-399`、`DEEPGEMM_PYTHON_INTERPRETERS`）。

> 抽取前提：路径 A 和路径 B 中除 Rust 外的部分，都受 `extract_extensions`（= `VLLM_USE_PRECOMPILED`）门控；Rust 产物受 `extract_rust_frontend` 门控。两者均在 `setup.py:806-832` 的抽取循环里判断。

加上 Rust 产物 `vllm/vllm-rs` + `vllm/_rust_*.so`。

### 7.4 wheel 来源解析顺序（`setup.py:637-734`）

| 优先级 | 来源 | 触发条件 |
|---|---|---|
| 1 | 本地路径或 URL | 显式设 `VLLM_PRECOMPILED_WHEEL_LOCATION` |
| 2 | ROCm 本地 wheel 或 AMD PyPI 索引 | ROCm 系统 |
| 3 | **官方 nightly 仓库** `https://wheels.vllm.ai/{commit}/{variant}/vllm/` | 默认（CUDA） |

第 3 条细节：

- **variant**（cu129/cu130）：`VLLM_PRECOMPILED_WHEEL_VARIANT` 或自动探测（`torch.version.cuda` → `nvidia-smi`，`setup.py:527-568`）
- **commit**：`VLLM_PRECOMPILED_WHEEL_COMMIT`，否则取上游 main HEAD 经 `git merge-base` 算出的基线 commit（`setup.py:858-920`）

### 7.5 典型用法

1. **本地开发免编译**（`AGENTS.md` 推荐）：
   ```
   VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
   ```
   下载匹配 commit+CUDA 变体的 nightly wheel，跳过 nvcc 全量编译。

2. **Dockerfile 两阶段复用**（`docker/Dockerfile:549-552`）：
   - `csrc-build` 阶段做真正的重编译 → 产出 `dist/*.whl`
   - `build` 阶段设 `VLLM_USE_PRECOMPILED=1` + `VLLM_PRECOMPILED_WHEEL_LOCATION=$(ls /precompiled-wheels/*.whl)`，**复用**前一阶段 `.so` 重新打包，不重编

### 7.6 ⚠️ 对本 fork 的注意点

- 预编译 wheel 的 `.so` 按**上游 commit**（或 `VLLM_PRECOMPILED_WHEEL_COMMIT`）构建。本仓库当前 HEAD `accaa434f` 虽是上游 commit，但属 v0.23.x 旧基线；fork 若在 C++/CUDA 侧有改动，下载的 nightly wheel 与本地源码会**不一致**，抽出的 `.so` 可能与改过的源码不匹配。
- 若 `wheels.vllm.ai` 上找不到对应 commit 的 wheel，解析会失败。此时最稳妥的做法：用 `VLLM_PRECOMPILED_WHEEL_LOCATION` 指向自己构建好的本地 wheel，或干脆走 `uv pip install -e .` 从源码编译。

---

## 8. 附录：关键证据汇总

### 外部工程 git 锁定版本（已逐一核对）

| 工程 | 仓库 | 版本/commit | 文件:行 |
|---|---|---|---|
| CUTLASS | nvidia/cutlass | `v4.4.2` | `CMakeLists.txt:389` |
| flash-attention | vllm-project/flash-attention | `803020a8fa15407871341d41eba4919ade2ee1ee` | `vllm_flash_attn.cmake:42` |
| DeepGEMM | deepseek-ai/DeepGEMM | `891d57b4db1071624b5c8fa0d1e51cb317fa709f` | `deepgemm.cmake:32` |
| QuTLASS | IST-DASLab/qutlass | `830d2c4537c7396e14a02a46fbddd18b5d107c65` | `qutlass.cmake:25` |
| FlashMLA | vllm-project/FlashMLA | `a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b` | `flashmla.cmake:22` |
| MSA (fmha_sm100) | vllm-project/MSA | `544eee5e09ae2dfa774d5b06739013f9b7402c57` | `fmha_sm100.cmake:20` |
| triton_kernels | triton-lang/triton | `v3.5.1` | `triton_kernels.cmake:3` |

### setup.py 声明的 CUDA 扩展（`setup.py:1097-1158`）

| 扩展名 | 产物 | 条件 |
|---|---|---|
| `vllm._C_stable_libtorch` | `_C_stable_libtorch.abi3.so` | CUDA/HIP |
| `vllm._moe_C_stable_libtorch` | `_moe_C_stable_libtorch.abi3.so` | CUDA/HIP |
| `vllm.cumem_allocator` | `cumem_allocator.abi3.so` | CUDA/HIP |
| `vllm.spinloop` | `spinloop.abi3.so` | py≥3.11 |
| `vllm.vllm_flash_attn._vllm_fa2_C` | `_vllm_fa2_C.abi3.so` | CUDA |
| `vllm.vllm_flash_attn._vllm_fa3_C` | `_vllm_fa3_C.abi3.so` | CUDA≥12.3 |
| `vllm._flashmla_C` / `_flashmla_extension_C` | `_flashmla_C.abi3.so` 等 | CUDA≥12.9 |
| `vllm._qutlass_C` | `_qutlass_C.abi3.so` | CUDA≥12.3 |
| `vllm._deep_gemm_C` | vendored 进 `third_party/deep_gemm` | CUDA≥12.3 |
| `vllm.triton_kernels` / `vllm.fmha_sm100` / `_vllm_fa4_cutedsl_C` | 仅拷贝 `.py`，无 `.so` | — |
| `vllm.vllm-rs` | 可执行文件 | Rust |
| `vllm._rust_tool_parser` | `_rust_tool_parser.abi3.so` | Rust |

### 一句话总结

> vLLM 的 GPU 构建是一个 **setuptools → CMake/Ninja** 管线：把 `csrc/` 下约 90 个 CUDA kernel（主力编进 stable-ABI 的 `_C_stable_libtorch` / `_moe_C_stable_libtorch`）加上 6 个外部 git 工程（CUTLASS header-only、flash-attn、DeepGEMM、QuTLASS、FlashMLA、triton_kernels）和 Rust 前端，按 `TORCH_CUDA_ARCH_LIST` 指定的架构族（含 sm_120 = `12.0`/`12.0f`）编译成一组 `.abi3.so`，连同整个 `vllm/` Python 包 + 数据文件打包成 stable-ABI wheel。CI 通过 Dockerfile 两阶段（csrc 重编 → build 复用 `VLLM_USE_PRECOMPILED`）产出 release wheel。
