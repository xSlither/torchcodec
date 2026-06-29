# Building TorchCodec for Windows against PyTorch 2.6 (with NVDEC)

This fork exists to produce a **Windows** TorchCodec wheel that is ABI-compatible
with **`torch==2.6.0+cu124`**, with the end goal of **NVDEC GPU decoding**.

This document is the engineering plan and build guide. It is the source of truth
for *why this branch is based where it is* and *how to build it*.

---

## 1. Why this is based on `v0.7.0`

TorchCodec is a compiled C++ `torch` extension. A prebuilt wheel is locked to the
exact `libtorch` it was built against (PyTorch does **not** promise C++ ABI
stability across minor versions), which is why no published wheel works with
torch 2.6 on Windows. Building **from source against your installed torch 2.6**
is what removes that lock — *provided the source still compiles against torch
2.6's C++ API.*

Walking the fork's own tags established the constraints:

| Era | versions | torch | C++ ABI | Win CPU | Win CUDA |
|---|---|---|---|---|---|
| classic, pre-Windows | 0.0.3–0.6 | 2.4–2.8 | classic C++17 | ❌ | ❌ |
| **classic, Windows-CPU** | **0.7 / 0.8 / 0.9** | 2.8 / 2.9 / 2.9 | **classic C++17** | ✅ | ❌ |
| stable-ABI | 0.11+ / main | ≥2.11 | `torch::stable`, C++20 | ✅ | ✅ |

Two hard facts fall out of this:

- **≥0.11 cannot compile against torch 2.6.** It uses the torch *stable ABI*
  (`torch::stable`, `StableABICompat.h`, `TORCH_TARGET_VERSION=0x020b…` = 2.11).
  Those headers/symbols don't exist in 2.6.
- **No version ever shipped classic-ABI *and* Windows-CUDA together.** Windows
  CUDA only appears in the stable-ABI era. So a torch-2.6 Windows NVDEC build is
  not a "download" — it's a port of the classic CUDA path onto Windows.

`v0.7.0` is therefore the right base: it is the **earliest Windows-capable,
classic-ABI** release, so it has the **smallest torch gap** (built for 2.8 vs our
2.6 — 2 minor versions) of any version that already solved Windows, and it still
uses the classic C++ API that exists in torch 2.6.

`v0.2.1` is the *exact* torch-2.6 match but has **zero** Windows support, so it
would mean re-porting all of 0.7's Windows enablement. We keep `v0.2.1` only as a
**reference**: if some torch API used by 0.7 turns out to be missing in 2.6, copy
the 2.6-correct equivalent from `v0.2.1`.

> The good news from reading `v0.7.0/CudaDeviceInterface.cpp`: the classic CUDA
> path is **NVDEC-decode-via-FFmpeg** (`av_hwdevice_ctx_create("cuda")`,
> `AV_PIX_FMT_CUDA`) + **NPP color conversion** (`nppiNV12ToRGB_*`). The torch
> CUDA glue it uses (`at::cuda::CUDAEvent/CUDAStream`, `c10::cuda::CUDAGuard`) is
> stable API present in torch 2.6. So the port is mostly **build-system
> plumbing**, not a rewrite of decode logic.

---

## 2. Division of labor

The actual MSVC/CUDA compile must happen on **Windows**. This repo (branch
`claude/torchcodec-windows-wheels-lqchhj`) holds the source/patches/CI.

- **Source, patches, CMake, CI** → done here, committed to the branch.
- **The compile** → on your Windows box (Milestones 1–2) and/or GitHub Actions
  (Milestone 3). You chose **both**: local first for fast iteration, then CI for
  clean reproducible wheels.

### ⚠️ Embedded-Python caveat (important for your setup)

You run a portable/embedded Python (`./python/python.exe`, ComfyUI-style).
Embeddable CPython usually **lacks `Python.h` and `pythonXY.lib`**, which
`find_package(Python3 ... COMPONENTS Development)` requires — so building
*directly* in it will likely fail at configure time.

**Recommended:** build the wheel in a **standard full CPython** of the *same
minor version* as your embedded Python (so the `cpXY` ABI tag matches), with
torch 2.6 installed, then `pip install` the resulting `.whl` into the embedded
Python. A wheel built under CPython 3.x works in embeddable 3.x of the same
minor version.

```bat
:: find the minor version you must match
.\python\python.exe --version
```

---

## 3. Milestone 1 — CPU wheel against torch 2.6 (validation milestone)

This proves the central hypothesis (v0.7.0 source compiles + imports against
torch 2.6) and exercises the whole toolchain. **No code changes needed** — v0.7.0
already builds CPU on Windows; the only variable is torch 2.6 vs 2.8.

### 3.1 Prerequisites (Windows)

- **MSVC** — Visual Studio 2022 Build Tools (v17.x). `vc_env_helper.bat` already
  targets VS 17–18.
- **CMake ≥ 3.18** and **Ninja** — you already have both (`cmake 3.30.5`,
  `ninja 1.11.x` in your env).
- **pybind11** — already installed (`2.13.6`). Expose its CMake config:
  `for /f %i in ('python -m pybind11 --cmakedir') do set pybind11_DIR=%i`
- **FFmpeg "shared" dev libraries.** Two options:
  - *Simple (single FFmpeg, pkg-config path):* download an FFmpeg **shared**
    build that includes `lib/pkgconfig/*.pc` + headers (e.g. BtbN
    `ffmpeg-*-win64-gpl-shared`), and set `PKG_CONFIG_PATH` to its `pkgconfig`
    dir. Requires `pkg-config` on PATH.
  - *Like upstream (multi-FFmpeg, no pkg-config):* set
    `BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1` and let CMake fetch Meta's prebuilt
    non-GPL FFmpeg (4–7). Needs network to their S3. **Note:** these non-GPL libs
    are build-time only and **do not provide NVDEC at runtime** (fine for CPU).

### 3.2 Build (in the full-CPython env with torch 2.6 installed)

```bat
:: from the repo root, inside a VS x64 dev prompt (or via vc_env_helper.bat)
set TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=ON
for /f %i in ('python -m pybind11 --cmakedir') do set pybind11_DIR=%i

:: pick ONE FFmpeg path:
set PKG_CONFIG_PATH=C:\path\to\ffmpeg-shared\lib\pkgconfig
:: ...or:  set BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1

:: editable dev build = fastest iteration; validates compile + import
python -m pip install -e . --no-build-isolation -v
```

### 3.3 Validate

```bat
python -c "import torch, torchcodec; print(torch.__version__, torchcodec.__version__)"
python -c "from torchcodec.decoders import VideoDecoder; d=VideoDecoder('test/resources/nasa_13013.mp4'); print(d.metadata); print(d[0].shape)"
pytest test/decoders/test_video_decoder_ops.py -q   :: or: pytest test -q
```

**Exit criteria:** import succeeds and a CPU decode returns a `[C,H,W]` uint8
tensor. If a compile error names a torch symbol, diff that snippet against
`v0.2.1` (the exact-2.6 reference) and backport it. Record any such patch in §6.

### 3.4 Produce a labeled wheel (optional at this stage)

```bat
set BUILD_VERSION=0.7.0+cpu.torch26
set BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1   :: required by setup.py for bdist_wheel
python -m build --wheel --no-isolation
```

---

## 4. Milestone 2 — NVDEC CUDA wheel against torch 2.6 (the goal)

Only start after Milestone 1 is green. The port is build-system plumbing plus
runtime dependency wiring. Apply these deltas **on this branch** (I'll do this in
the repo once CPU is validated):

### 4.1 Code/build deltas to apply

**A. `src/torchcodec/_core/CMakeLists.txt` — modern CUDA + NPP discovery.**
v0.7 uses the deprecated `FindCUDA` vars (`CUDA_nppi_LIBRARY`,
`CUDA_nppicc_LIBRARY`) which are unreliable on MSVC. Replace with:
```cmake
if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()
# ...
if(ENABLE_CUDA)
    list(APPEND core_library_dependencies
        CUDA::cudart
        CUDA::nppig    # nppiNV12ToRGB_* lives here
        CUDA::nppicc   # color-conversion helpers
        CUDA::nppc)    # NPP core
    target_include_directories(${core_library_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
```
`CudaDeviceInterface.cpp` is host-compiled (no `.cu` in 0.7), so
`enable_language(CUDA)`/nvcc is **not** required — it only needs the CUDA
include dir + `cudart` + NPP libs. (`CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS` is already
ON in 0.7 "for when we add Windows CUDA".)

**B. `setup.py` — Windows generator.** Port HEAD's win32 block so CMake uses
Ninja under MSVC instead of the default MSBuild generator:
```python
if sys.platform == "win32":
    cmake_args.append("-G Ninja")
```

**C. `packaging/vc_env_helper.bat` — CUDA variant.** Add for the CUDA build:
```bat
set ENABLE_CUDA=1
```
(or pass `ENABLE_CUDA=1` in the environment before the build).

### 4.2 Toolchain / runtime prerequisites

- **CUDA Toolkit 12.4** (matches your `torch 2.6.0+cu124`) — provides NPP
  (`nppig`, `nppicc`, `nppc`) + `cudart` import libs/headers for the build.
- **NVDEC-enabled FFmpeg "shared" build at runtime** — the S3/non-GPL libs do
  **not** decode via NVDEC. Use a GPL-shared FFmpeg that lists cuvid:
  ```bat
  ffmpeg -decoders | findstr /i nvidia   :: expect h264_cuvid, hevc_cuvid, ...
  ```
  Put its `bin\` on `PATH` so the DLLs load at runtime.
- **Runtime CUDA DLLs:** `cudart64_12.dll` and `nvrtc` ship with torch's cu124
  wheel; the **NPP DLLs** (`nppig64_12.dll`, `nppicc64_12.dll`, `nppc64_12.dll`)
  come from the CUDA Toolkit — either keep the Toolkit `bin\` on `PATH` or copy
  those DLLs next to the installed `torchcodec` libs (the Linux CUDA wheels bundle
  NPP; we may do the same in CI).

### 4.3 Build & validate

```bat
set ENABLE_CUDA=1
set TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=ON
for /f %i in ('python -m pybind11 --cmakedir') do set pybind11_DIR=%i
set PKG_CONFIG_PATH=C:\path\to\ffmpeg-nvdec-shared\lib\pkgconfig
python -m pip install -e . --no-build-isolation -v
```
```bat
python -c "from torchcodec.decoders import VideoDecoder; d=VideoDecoder('test/resources/nasa_13013.mp4', device='cuda'); f=d[0]; print(f.shape, f.device)"
```
**Exit criteria:** a frame decodes with `device='cuda'` and the tensor is on the
GPU (no host round-trip).

### 4.4 Known risks for this milestone

- NPP symbol/lib names differ slightly across CUDA versions; the first CUDA
  configure/link will surface the exact missing lib — adjust the `CUDA::npp*`
  list accordingly.
- 10-bit / non-NV12 video is unsupported by this NPP path (CPU fallback only) —
  same as upstream at 0.7.
- `find_package(Python3 COMPONENTS Development)` + embedded Python (see §2.x).

---

## 5. Milestone 3 — CI (GitHub Actions, both CPU and CUDA)

v0.7's `windows_wheel.yaml` points at a personal fork
(`nicolashug/test-infra@release28windows`) and `release/2.8`. For reproducible
torch-2.6 wheels:

- Repoint to upstream test-infra pinned to **`release/2.6`**:
  `generate_binary_build_matrix.yml@release/2.6` and the merged
  `build_wheels_windows.yml`.
- CPU job: `with-cuda: disable`, install `torch --index-url .../whl/cpu` (2.6).
- CUDA job (new `windows_cuda_wheel.yaml`): `with-cuda: enable` for cu124, set
  `ENABLE_CUDA=1` in `vc_env_helper.bat`, install
  `torch --index-url .../whl/cu124` (2.6), and install an NVDEC FFmpeg for tests.
- Upload wheels as artifacts; no PyPI publish.

---

## 6. Patch log (torch-2.6 deltas backported from v0.2.1)

_None yet — to be filled in as Milestone 1/2 surface any._

| File | Symbol / issue | Fix (source) |
|---|---|---|
| | | |

---

## 7. Status

- [x] Branch based on `v0.7.0`
- [ ] Milestone 1 — CPU build green against torch 2.6
- [ ] Milestone 2 — NVDEC CUDA build green against torch 2.6
- [ ] Milestone 3 — CI producing labeled wheels
