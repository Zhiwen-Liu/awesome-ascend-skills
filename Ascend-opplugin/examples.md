# Generic steps to add a new operator

The two reference examples (add, matmul_leakyrelu) illustrate two patterns. Use this checklist for **any** new operator; replace placeholders with your operator names.

## 1. Choose pattern

- **Pattern A:** No workspace, only inputs/outputs (and optional scalars). Example: add.
- **Pattern B:** Needs workspace (and optionally tiling). Example: matmul_leakyrelu.

## 2. Add kernel

- Add `csrc/kernel/{kernel_name}_custom.cpp`.
- Implement CopyIn → Compute → CopyOut in Ascend C. Pattern B: use workspace/tiling as required.
- Ensure the global kernel entry name matches `{kernel_name}` (used in host and CMake).

## 3. Add host

- Add `csrc/host/{op_name}.cpp` with:
  - `TORCH_LIBRARY_FRAGMENT(npu, m)` and `m.def("{op_name}(...) -> ...")`
  - Implementation function: allocate output (and for Pattern B: workspace, tiling); include `aclrtlaunch_{kernel_name}.h`; call `EXEC_KERNEL_CMD({kernel_name}, blockDim, ...)`
  - `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` and `m.impl("{op_name}", TORCH_FN(run_xxx))`
- Reuse `utils.h`; pass only lvalues to EXEC_KERNEL_CMD.
- Pattern B with tiling: implement tiling in `csrc/host/tiling/` and include in CMake host sources.

## 4. Update CMakeLists.txt

- **Pattern A:** `ascendc_library(no_workspace_kernel STATIC csrc/kernel/{kernel_name}_custom.cpp)` (or a new target name). Link into the shared op-extension library.
- **Pattern B:** `ascendc_library(workspace_kernel STATIC csrc/kernel/{kernel_name}_custom.cpp)` and `ascendc_compile_definitions(workspace_kernel PRIVATE -DHAVE_WORKSPACE -DHAVE_TILING)`. Add tiling sources to host file list if used. Link workspace_kernel into the shared library.
- Ensure `csrc/host/*.cpp` (and `csrc/host/tiling/*.cpp` if any) are in the shared library sources.

## 5. Add test in test/test.py

- `import {pkg}` (loads .so).
- Create inputs (e.g. `torch.rand(...).npu()`).
- `output = torch.ops.npu.{op_name}(...)`
- Compute `cpu_ref` (equivalent PyTorch op or formula).
- `self.assertRtolEqual(output, cpu_ref)` (or project equivalent).
- Add as a new test method, e.g. `test_xxx`, following the same pattern as the add and matmul_leakyrelu tests.

No separate demo script; running and validation are done via test.
