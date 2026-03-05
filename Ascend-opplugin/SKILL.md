---
name: Ascend-opplugin
description: Installs op-plugin (torch_npu operator plugin) environment and guides custom NPU operator integration with PyTorch via two generic patterns (no workspace vs workspace+tiling), from kernel implementation through host registration, build, and test. Use when working with op-plugin, operator integration, torch_npu custom ops, Ascend C, NPU operators, cpp_extension, or running custom operators on NPU.
---

# op-plugin

Guides installing the op-plugin environment and **generic** custom-operator integration for torch_npu. Two reference examples (add, matmul_leakyrelu) define two patterns; all new operators follow one of these patterns. **Generalization is the priority.**

## 0. Quick pre-check and branch selection

- **Check if torch_npu is already installed and usable (preferred path):**
  - Run:
    - `python - << 'EOF'`
    - `import torch; import torch_npu`
    - `print("torch:", torch.__version__)`
    - `print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))`
    - `print("npu available:", torch.npu.is_available())`
    - `EOF`
  - If import succeeds and `npu available: True`, you can **skip building torch_npu from op-plugin** and directly go to **Section 2+** for custom op integration.
- **If torch_npu is missing/broken:** follow Section 1 to install it via op-plugin.
- Always run everything in the **same Python environment** (same venv/conda) for:
  - building torch_npu,
  - building custom operators,
  - installing the wheel,
  - running tests.

## 1. Install op-plugin environment

This step should be **idempotent**: 优先复用已有环境，只在缺失或不匹配时才重建。

- **1.1 Source CANN once per shell**
  - `source <CANN install path>/set_env.sh` (e.g. `/usr/local/Ascend/ascend-toolkit/set_env.sh`).

- **1.2 Check whether op-plugin repo & torch_npu already match**
  - If you already have an op-plugin checkout and a working `torch_npu` in the same Python env, you can usually **skip rebuilding**:
    - Example quick check:
      - `python - << 'EOF'`
      - `import torch, torch_npu`
      - `print("torch:", torch.__version__)`
      - `print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))`
      - `print("npu available:", torch.npu.is_available())`
      - `EOF`
    - If imports succeed, `npu available: True`，且版本满足你当前项目需求，就可以直接进入 **Section 2** 进行算子接入，无需再次用 op-plugin 编译 `torch_npu`。

- **1.3 如果本地没有合适的 torch_npu 或需要指定版本，则准备 / 切换 op-plugin 环境**
  - Clone op-plugin（分支与目标 torch_npu 版本匹配）：
    - `git clone --branch 7.3.0 https://gitcode.com/ascend/op-plugin.git && cd op-plugin`
  - Build:
    - `bash ci/build.sh --python=3.9 --pytorch=v2.7.1-7.3.0`
    - 其中 `--python` 和 `--pytorch` 根据你的实际 Python / PyTorch 目标版本调整，具体可查 [reference.md](reference.md) 中的版本矩阵。
  - Install:
    - `cd dist && pip install dist/torch_npu-*.whl`
    - 之后回到 Section 0 的自检命令，再次确认 `torch_npu` 可用且 `npu available: True`。

Dependencies: torch_npu, CANN. Prefer the torch_npu Docker image for build. Version matrix (op-plugin branch ↔ PyTorch/Python/GCC) is in [reference.md](reference.md).

## 2. 接入模式选择：是否已经有 CANN 算子

**New-operator flow（更新版）:** 先判断 **CANN / ops-nn 里是否已经有这个算子** → 能复用则优先复用（Pattern C，OpCommand 直调 CANN 图算子）→ 否则再选择 Pattern A / Pattern B 自己写 AscendC kernel。

- **如果 ops-nn 里已经有完整实现**（有 AscendC kernel、tiling、`op_graph/*.h`、`op_host/op_api/*.cpp` 等，例如 `norm/layer_norm_v3`、`layer_norm_v4`）：
  - 不必再编译新的 AscendC kernel；
  - 只需要在自定义扩展里写一层很薄的 host wrapper，用 `at_npu::native::OpCommand` 调用已有图算子名（如 `"LayerNormV3"` / `"LayerNormV4"`），即可暴露成 `torch.ops.npu.*`；
  - CMake 只链接 `torch_npu`，让它内部处理 ACL/CANN 依赖（避免你手动去找 `libascendcl.so`、`libtiling_api.so` 等库路径）。

- **如果 CANN 侧没有这个算子，只是你自己的 AscendC kernel：**
  - 继续使用 Pattern A / Pattern B 的经典方式：自己写 kernel + tiling + host 封装。

下面的 Pattern A / Pattern B / Pattern C 可以同时存在一个工程里，关键是：**能用系统已有的，就不要重复造轮子**。

### Pattern A — No workspace (reference: add)

- **Kernel:** Inputs and outputs only (optional scalars). No workspace/tiling. File: `csrc/kernel/{kernel_name}_custom.cpp`, Ascend C: CopyIn → Compute → CopyOut.
- **Host:** Allocate output only (e.g. `at::empty_like(x)` or `at::empty(...)`). Call `EXEC_KERNEL_CMD({kernel_name}, blockDim, input..., output[, scalars])`. Include `aclrtlaunch_{kernel_name}.h`.
- **CMake:** `ascendc_library(no_workspace_kernel STATIC csrc/kernel/{kernel_name}_custom.cpp)` (or a dedicated target per kernel). Link this library into the shared op-extension target.

### Pattern B — Workspace and/or tiling (reference: matmul_leakyrelu)

- **Kernel:** Uses workspace (and optionally tiling). File: `csrc/kernel/{kernel_name}_custom.cpp` with CopyIn → Compute → CopyOut. Build with HAVE_WORKSPACE (and HAVE_TILING if tiling is used).
- **Host:** Allocate output, workspace tensor (size from platform or user), and optionally tiling tensor; call tiling generator if needed. Call `EXEC_KERNEL_CMD({kernel_name}, blockDim, input..., output, workspace[, tiling])`. Include `aclrtlaunch_{kernel_name}.h`.
- **CMake:** `ascendc_library(workspace_kernel STATIC csrc/kernel/{kernel_name}_custom.cpp)` and `ascendc_compile_definitions(workspace_kernel PRIVATE -DHAVE_WORKSPACE -DHAVE_TILING)` (drop HAVE_TILING if not used). Add `csrc/host/tiling/*.cpp` to the host sources if tiling is implemented. Link workspace_kernel into the shared op-extension target.

### Pattern C — 复用已有 CANN 算子（OpCommand 模式）

当目标算子在 CANN / ops-nn 里已经完整实现时（例如 `ops-nn/norm/layer_norm_v3` / `layer_norm_v4`），推荐用 Pattern C：

- **思路**：直接通过 `OpCommand` 调用 CANN 图算子名，而不是再新增 AscendC kernel。

- **Host 层写法示例（LayerNormV3）：**
  - PyTorch API 设计：
    - `torch.ops.npu.layer_norm_v3(x, gamma, beta, begin_norm_axis, begin_params_axis, eps) -> (y, mean, rstd)`
  - 实现要点：
    - 检查 `x` 在 NPU 上：`x.device().type() == PrivateUse1`。
    - 构造 `y/mean/rstd` 输出：
      - `y`：`at::empty_like(x)`
      - `mean/rstd`：shape `[A1...Ai, 1...1]`，其中 `i = begin_norm_axis`。
    - 使用 `OpCommand`：
      - `.Name("LayerNormV3")`
      - `.Input(x).Input(gamma).Input(beta)`
      - `.Output(y).Output(mean).Output(rstd)`
      - `.Attr("begin_norm_axis", (int64_t)begin_norm_axis)`
      - `.Attr("begin_params_axis", (int64_t)begin_params_axis)`
      - `.Attr("epsilon", (float)eps)`
      - `.Run()`

- **Host 层写法示例（LayerNormV4）：**
  - PyTorch API 设计：
    - `torch.ops.npu.layer_norm_v4(x, int[] normalized_shape, Tensor? gamma=None, Tensor? beta=None, float eps=1e-5) -> (Tensor, Tensor, Tensor)`
  - 实现要点：
    - C++ 签名使用 `at::IntArrayRef normalized_shape`、`c10::optional<at::Tensor> gamma_opt/beta_opt`。
    - 输出：
      - `y = at::empty_like(x)`
      - `mean/rstd` 形状 `[A1...Ai, 1...1]`，`Ai` 为非归一化轴（前 `input.dim() - normalized_shape.size()` 个）。
    - OpCommand 调用：
      - `.Name("LayerNormV4")`
      - `.Input(x)`
      - `.Input(normalized_shape)`  // host int list，OpCommand 自动做 H2D
      - 对可选输入：
        - 若 `gamma_opt` 有值：`.Input(*gamma_opt)`，否则 `.Input()`（空输入对应 OPTIONAL_INPUT）。
        - 若 `beta_opt` 有值：`.Input(*beta_opt)`，否则 `.Input()`。
      - `.Output(y).Output(mean).Output(rstd)`
      - `.Attr("epsilon", (float)eps)`
      - `.Run()`

- **CMake 方面的简化：**
  - 不需要任何 `ascendc_library` / tiling 源文件，只保留 host 源码：
    - `file(GLOB _SRCS csrc/host/*.cpp)`
    - `add_library({pkg} SHARED ${_SRCS})`
  - 链接库只保留：
    - `target_link_libraries({pkg} PRIVATE torch_npu)`
  - 头文件目录补上：
    - `${TORCH_NPU_PATH}/include`
    - `${TORCH_PATH}/include` 以及 `torch/csrc/api/include`
    - `${ASCEND_CANN_PACKAGE_PATH}/include`（解决 `graph/types.h` 等依赖）

- **踩坑经验（重要）：**
  - `OpCommand::Attr` 的整型参数要用 `int64_t`，否则会在 `OpAttrMaker::Set` 上出现 `bool` / `int64_t` 重载歧义编译错误。
  - `normalized_shape` 在 Python 端必须传 `List[int]`，不能传 `Tensor`；C++ 端用 `IntArrayRef` 即可。
  - 可选 Tensor 最好用 `Tensor?` + `.Input()` 的空输入表示法，而不是随意造“占位 Tensor”。
  - 若已经有 `aclnn_layer_norm*` 一类 C 接口，通常也不必直接手写 `aclTensorDesc` / `aclDataBuffer` / `aclnn*GetWorkspaceSize`，而是先考虑能否通过已有图算子名 + `OpCommand` 完成。

这一模式的核心启发：**先找 CANN 和 op-plugin 里是否已经有算子实现，有的话只加一层 PyTorch Host 接入即可。**

## 3. Kernel implementation (generic)

- Add `csrc/kernel/{kernel_name}_custom.cpp`. Follow Ascend C: CopyIn → Compute → CopyOut. Pattern A: no workspace. Pattern B: use workspace/tiling as in [Ascend C docs](https://www.hiascend.com/ascend-c).
- The kernel entry must match the name used in host and CMake: the generated header is `aclrtlaunch_{kernel_name}.h`, and host calls `EXEC_KERNEL_CMD({kernel_name}, ...)`.
- In CMake, add an `ascendc_library` that compiles this file; for Pattern B add `ascendc_compile_definitions` with `-DHAVE_WORKSPACE` and optionally `-DHAVE_TILING`. Ensure the library is linked into the final shared library (e.g. `op_extension` / `lib{pkg}.so`).

## 4. PyTorch integration — Host (generic)

- Add `csrc/host/{op_name}.cpp` with:
  - **Aten IR definition:** `TORCH_LIBRARY_FRAGMENT(npu, m) { m.def("{op_name}(...) -> ..."); }`
  - **Implementation:** A function (e.g. `run_xxx`) that allocates output (and for Pattern B: workspace, tiling), includes `aclrtlaunch_{kernel_name}.h`, and calls `EXEC_KERNEL_CMD({kernel_name}, blockDim, ...)`.
  - **Registration:** `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) { m.impl("{op_name}", TORCH_FN(run_xxx)); }`
- Reuse `utils.h` from the cpp_extension example (EXEC_KERNEL_CMD, ConvertTypes). Pass scalars as **lvalues** (no rvalues in EXEC_KERNEL_CMD).
- Pattern B: In the host implementation, compute workspace size (e.g. from platform API) and call the tiling generator; pass workspace and tiling tensors into EXEC_KERNEL_CMD.

## 5. Running the custom operator (build and test)

1. **SOC_VERSION:** In CMakeLists.txt set `set(SOC_VERSION "Ascendxxxyy" ...)` to your chip. Get Chip Name with `npu-smi info`; use `Ascend` + Chip Name (e.g. Ascend910B).
2. **Build wheel:** `python setup.py bdist_wheel`
3. **Install (force overwrite old installation):**
   - `cd dist && pip install --force-reinstall *.whl`
   - This avoids accidentally reusing an older `{pkg}` already in site-packages (which would hide new operators).
4. **Run tests:** `cd test && python test.py`

## 6. Test writing (generic; same pattern for both examples)

- **Unified steps:** `import {pkg}` (loads the .so and registers ops); create input tensors (on CPU then `.npu()` or directly on NPU); `output = torch.ops.npu.{op_name}(...)`; compute CPU reference `cpu_ref` (existing PyTorch op or formula); `TestCase.assertRtolEqual(output, cpu_ref)` (or the project’s equivalent).
- **Pattern A–style (like add):** `cpu_ref = torch.add(x, y)` (or the equivalent PyTorch op).
- **Pattern B–style (like matmul_leakyrelu):** `cpu_ref = some_combination(e.g. LeakyReLU(matmul(a,b) + bias))`.
- For each new operator, add a new test method (e.g. `test_xxx`) in `test/test.py` following this pattern. No separate demo.py; use test as the single entry for running and validating.
- If you rely on `torch_npu.testing.testcase.TestCase`, ensure `expecttest` is installed once: `pip install expecttest`.

## 7. Necessary files and scripts (generic)

Placeholders: `{pkg}` = Python package name, `{kernel_name}` = kernel entry name, `{op_name}` = PyTorch API name（**直接使用你为算子起的名称即可**，例如 kernel 叫 `add_custom`，则 `{op_name}` 通常也叫 `add_custom`，保持一套命名贯通即可）。Pattern A does not require workspace/tiling dirs; Pattern B requires workspace and optionally `csrc/host/tiling/` with tiling sources in CMake.

```
<project_root>/
├── {pkg}/
│   ├── __init__.py       # call _load_opextension_so()
│   └── _load.py          # torch.ops.load_library(.../lib/lib{pkg}.so)
├── csrc/
│   ├── kernel/
│   │   └── {kernel_name}_custom.cpp
│   └── host/
│       ├── {op_name}.cpp
│       ├── utils.h
│       └── tiling/        # optional, Pattern B
│           └── *_tiling.cpp
├── CMakeLists.txt        # SOC_VERSION, ascendc_library, add_library, link torch_npu
├── setup.py              # NpuExtension, build_clib/build_ext, package name {pkg}
└── test/
    └── test.py           # import {pkg}; .npu(); torch.ops.npu.{op_name}(...); cpu_ref; assertRtolEqual
```

**Naming consistency:** `{kernel_name}` must match `aclrtlaunch_{kernel_name}.h`, `EXEC_KERNEL_CMD({kernel_name}, ...)`, and the kernel source file. Package name must match the .so name and setup.py.

**Multiple operators:** Add one `ascendc_library` per kernel (choose Pattern A or B and HAVE_WORKSPACE/HAVE_TILING accordingly), add one `csrc/host/{op_name}.cpp` per op (and tiling sources if needed), and add one `test_xxx` in test.py per op, keeping the same naming and test pattern as in the two reference examples.

## 8. End-to-end automation checklist (demo-style)

When you want a **fully automated, demo-style flow** (like the cpp_extension examples), follow this script in your project root:

1. **Environment & torch_npu (once per machine):**
   - `source <CANN install path>/set_env.sh`
   - Run the pre-check in Section 0.
   - If torch_npu is missing/broken, build and install it via Section 1.
2. **Project build (per change to kernels/host/CMake):**
   - Set `SOC_VERSION` in `CMakeLists.txt` to a supported chip string (see CANN `host_config.cmake` support list; e.g. `ascend910b2`).
   - `python setup.py bdist_wheel`
   - `cd dist && pip install --force-reinstall *.whl`
3. **Run demo tests (per operator change):**
   - In `test/test.py`, follow Section 6 to:
     - import `{pkg}` (this auto-loads `lib{pkg}.so`),
     - call `torch.ops.npu.{op_name}(...)` with **op_name matching your operator name**, not `my_*`,
     - compute CPU `cpu_ref`,
     - compare with `assertRtolEqual`.
   - Execute: `cd test && python test.py`
4. **Quick verification of registration (optional debug step):**
   - `python - << 'EOF'`
   - `import torch, {pkg}`  # noqa
   - `print([name for name in dir(torch.ops.npu) if "{op_name_hint}" in name])`
   - `EOF`
   - Use this when tests say "no attribute" to confirm whether your op is actually registered.


## Additional resources

- [reference.md](reference.md) — Version table, SOC_VERSION, links to op-plugin and cpp_extension README, Ascend C.
- [examples.md](examples.md) — Generic “add a new operator” checklist (choose pattern → kernel → host → CMake → test) without binding to a specific op name.
