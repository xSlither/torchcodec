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

### Build environment: a `uv` venv (your setup)

You build in a fresh clone on Windows with a per-project **`uv` venv**, CUDA
**12.4** installed globally, and torch pinned to **2.6.0+cu124** — reproducing
the environment you target elsewhere.

- **Pick the venv's Python minor version to match wherever the wheel will run**
  (e.g. your ComfyUI Python). The wheel is ABI-tagged `cpXY`, so a wheel built
  under 3.12 installs into any 3.12 environment, including an embeddable one.
- **uv-managed CPython includes dev headers + import lib** (it uses
  python-build-standalone), so `find_package(Python3 ... COMPONENTS Development)`
  works — *unlike* a bare ComfyUI embeddable distribution. That's why we build in
  the venv and deploy the resulting wheel.
- Run the build from an **"x64 Native Tools Command Prompt for VS 2022"** (so
  `cl.exe` is on PATH) with the venv activated. CUDA 12.4's installer sets
  `CUDA_PATH`, so `find_package(CUDAToolkit)` locates NPP for Milestone 2 with no
  extra config.

---

## 3. Milestone 1 — CPU wheel against torch 2.6 (validation milestone)

This proves the central hypothesis (v0.7.0 source compiles + imports against
torch 2.6) and exercises the whole toolchain. **No code changes needed** — v0.7.0
already builds CPU on Windows; the only variable is torch 2.6 vs 2.8.

### 3.1 Prerequisites (Windows)

- **MSVC** — Visual Studio 2022 Build Tools (v17.x); build from the "x64 Native
  Tools Command Prompt for VS 2022".
- **`uv`** — creates the venv and installs everything into it (torch, cmake,
  ninja, pybind11, build).
- **A CUDA Toolkit + a matching MSVC toolset** — a working `nvcc` is required
  even for the *CPU* build, because torch `2.6.0+cu124`'s CMake forces
  `enable_language(CUDA)`. You do **not** need CUDA 12.4: building with an older
  CUDA (e.g. **12.3**) is runtime-compatible with cu124 torch wheels (CUDA 12.x
  minor-version compatibility; torch doesn't use NPP, so no conflict). The catch
  is the **MSVC toolset must be old enough for your CUDA** — recent MSVC STL
  (~14.40+) hard-blocks CUDA < 12.4 (`STL1002`). For CUDA 12.3, build under MSVC
  **14.39** (VS 17.9) via `vcvarsall -vcvars_ver=14.39` + the Ninja generator
  (see Troubleshooting §7).
- **FFmpeg "shared" dev libraries.** Two options:
  - *Simple (single FFmpeg, pkg-config path):* download an FFmpeg **shared**
    build that includes `lib/pkgconfig/*.pc` + headers (e.g. BtbN
    `ffmpeg-*-win64-gpl-shared`), and set `PKG_CONFIG_PATH` to its `pkgconfig`
    dir. Requires `pkg-config` on PATH.
  - *Like upstream (multi-FFmpeg, no pkg-config):* set
    `BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1` and let CMake fetch Meta's prebuilt
    non-GPL FFmpeg (4–7). Needs network to their S3. **Note:** these non-GPL libs
    are build-time only and **do not provide NVDEC at runtime** (fine for CPU).

### 3.2 Build (from the repo root, in an x64 VS dev prompt)

```bat
:: 0) get this branch (a fresh clone lands on main/0.15, not our v0.7.0 base)
git fetch origin
git checkout claude/torchcodec-windows-wheels-lqchhj

:: 1) venv via uv — match the Python minor version to your deploy target
uv venv --python 3.12 .venv
.venv\Scripts\activate

:: 2) torch 2.6 (cu124) + build deps into the venv
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install setuptools wheel build cmake ninja pybind11 numpy pytest pillow

:: 3) build config
set TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=ON
for /f %i in ('python -m pybind11 --cmakedir') do set pybind11_DIR=%i
:: pick ONE FFmpeg path:
set PKG_CONFIG_PATH=C:\path\to\ffmpeg-shared\lib\pkgconfig
:: ...or:  set BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1

:: 4) editable dev build = fastest iteration; validates compile + import
uv pip install -e . --no-build-isolation -v
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
:: From a clean cmd.exe, pin the CUDA-12.3-compatible toolset (as in Milestone 1):
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.39
cl              :: must say 19.39.x
nvcc --version  :: 12.3

cd /d C:\Users\chase\Documents\GitHub\torchcodec
call .venv\Scripts\activate.bat
set ENABLE_CUDA=1
set TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=ON
for /f "delims=" %i in ('python -m pybind11 --cmakedir') do set pybind11_DIR=%i
set BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1
uv pip install -e . --no-build-isolation -v --reinstall-package torchcodec
```
The CMake NPP/CUDA discovery (`find_package(CUDAToolkit)` → `CUDA::nppicc/nppig/nppc`
+ `cudart`) and the runtime CUDA/NPP DLL registration are already wired into the
fork. At runtime keep the FFmpeg 7.x shared `bin` on PATH **and** the CUDA 12.3
`bin` on PATH (the loader shim registers both for the DLL search).
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
- `find_package(Python3 COMPONENTS Development)` resolution: uv-managed CPython
  ships headers + import lib, so this should work; if it ever fails, pass
  `-DPython3_ROOT_DIR=<venv>` (or use a python.org interpreter for the venv).

---

## 4.5 Building a distributable wheel (for the ComfyUI / torch-2.6 env)

The editable install proves it works in this venv. To deploy into another
cp311 + torch-2.6 environment, build a real wheel (same activated toolset/venv):
```bat
set ENABLE_CUDA=1
set BUILD_VERSION=0.7.0+cu123.torch26   :: stamps the wheel name; no git-sha suffix
set BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1
python -m build --wheel --no-isolation
:: -> dist\torchcodec-0.7.0+cu123.torch26-cp311-cp311-win_amd64.whl
```
Install it into the target env: `pip install <that>.whl`.

**Runtime requirements at the deploy site:**
- An FFmpeg **7.x shared** build with NVDEC (`avcodec-61.dll`, `h264_cuvid`) —
  always user-provided (we don't bundle FFmpeg; it's GPL).
- The **NPP + CUDA-runtime DLLs** (`nppicc64_12.dll`, `nppig64_12.dll`,
  `nppc64_12.dll`, `cudart64_12.dll`). `cudart` ships inside torch's cu124 wheel;
  **NPP** comes from a CUDA toolkit — *unless you bundle it* (below).

**Self-contained wheel (bundle NPP) — for hosting/sharing:** set
`TORCHCODEC_BUNDLE_CUDA_DLLS=1` at build time. `setup.py` then copies the NPP
(+ cudart) DLLs from `%CUDA_PATH%\bin` into the wheel next to the core libs;
Windows always searches a DLL's own directory for its dependencies, so they
resolve with **no CUDA toolkit and nothing on PATH** at the deploy site. NPP is
an NVIDIA redistributable, so this is license-clean. FFmpeg still required.
```bat
set ENABLE_CUDA=1
set TORCHCODEC_BUNDLE_CUDA_DLLS=1
set BUILD_VERSION=0.7.0+cu124.torch26
set BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1
python -m build --wheel --no-isolation
```
Without bundling, the `ops.py` shim still finds NPP/FFmpeg via `CUDA_PATH*`/PATH
(works on *your* box where CUDA 12.3 is installed).

> Is it *really* NVDEC (not a CPU→GPU copy)? The cuda path tries hardware decode
> first and only falls back to CPU if NVDEC can't handle the codec; h264 is
> NVDEC-supported and the decode returned a `cuda:0` tensor without the NV12/
> hw_frames_ctx errors, so NVDEC engaged. To confirm with numbers, run
> `python benchmarks/decoders/gpu_benchmark.py --devices=cuda:0,cpu --resize_devices=none`
> or watch `nvidia-smi dmon` (the `dec` column) during a decode loop.

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
| — | **None needed.** v0.7.0 C++ (incl. `CudaDeviceInterface.cpp`) compiles clean against torch 2.6 (warnings only: C4244/C4267/C4702/C4458/C4245). torch-2.6 source compatibility confirmed for both CPU and CUDA paths. | n/a |

---

## 7. Troubleshooting log

### `STL1002: Unexpected compiler version, expected CUDA 12.4 or newer`
Configure fails inside *PyTorch's* CMake (`TorchConfig → Caffe2 → cuda.cmake →
enable_language(CUDA)`), not in torchcodec — torch `+cu124` forces the CUDA
language on, so an `nvcc` + host-compiler pair must work even for the CPU build.
The error is an **MSVC STL guard**: recent MSVC toolsets (~14.40+) hard-block
CUDA < 12.4. It is **not** a real torch/CUDA incompatibility, and you do **not**
need to install CUDA 12.4 — building with your installed CUDA (e.g. 12.3) is
runtime-compatible with cu124 torch wheels. Fix = build with an MSVC toolset from
that CUDA's era:

1. **Ninja generator** (already wired into `setup.py` for win32). This is
   essential: the Visual Studio/MSBuild generator ignores the shell environment
   and always uses the newest installed toolset (e.g. 14.44), re-triggering the
   guard. Ninja uses whatever `cl.exe` is active in the shell. Requires `ninja`
   on PATH (it's in the build venv).
2. **Pin an older toolset** before building (add "MSVC v143 14.39" in the VS
   Installer if absent), keeping nvcc 12.3:
   ```bat
   "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.39
   cl              :: should report 19.39.x
   nvcc --version  :: still 12.3 -- fine
   ```
If 14.39 still trips the guard, step down to 14.38. If CMake's Ninja+CUDA pairing
can't find the host compiler, set `CUDAHOSTCXX` to the active `cl.exe`.

### `target "torch::nvtoolsext" contains CUDA::nvToolsExt but the target was not found`
Another *PyTorch* CMake issue (not torchcodec). CUDA 12.x removed the legacy
`nvToolsExt` library (NVTX is now header-only NVTX3); torch 2.6's
`Caffe2/public/cuda.cmake` fails to find NVTX3 on Windows and falls back to the
missing `CUDA::nvToolsExt` target, so `find_package(Torch)` dies at generate
time. NVTX is an unused profiling shim here. Fixed in our fork's
`src/torchcodec/_core/CMakeLists.txt`: before `find_package(Torch)` we
`find_package(CUDAToolkit QUIET)` and define an empty `CUDA::nvToolsExt`
stand-in if it's absent (WIN32-only, so Linux/macOS CUDA builds are untouched).

### `import torchcodec` fails: "Could not load libtorchcodec_coreN" / FFmpeg not found
The build links against the S3 FFmpeg (build-time only); at **runtime** torchcodec
dlopens the FFmpeg **shared** DLLs and loads the `libtorchcodec_coreN` matching the
FFmpeg major version it finds. Two ways this fails:
- A **non-shared** FFmpeg build (just `ffmpeg.exe`, e.g. BtbN `...-win64-gpl`) has
  no DLLs, so all cores fail to load.
- A build that is **too new**: v0.7.0 supports only **FFmpeg 4–7** (libavcodec
  58–61) and `core7` links the `avcodec-61` soname. BtbN/gyan **`master`/latest is
  now FFmpeg 8** (libavcodec 62/63) and will **not** load.

Fix: install an FFmpeg **7.x shared** build (libavcodec **61**) — e.g. BtbN
`ffmpeg-n7.1-latest-win64-gpl-shared.zip` (the `n7.x` branch, *not* master) or a
gyan.dev `7.1 ...-shared` archive — confirm `bin\avcodec-61.dll` exists, and put
its `bin\` on `PATH`. The `gpl-shared` n7.x build also carries NVDEC/cuvid for
Milestone 2.

**Important (Windows DLL search):** putting FFmpeg on `PATH` is *not* sufficient
on Python 3.8+ — `ctypes` (used by `torch.ops.load_library`) resolves a DLL's
dependencies only through directories registered with `os.add_dll_directory()`,
not `PATH`. The fork handles this in `_core/ops.py::_maybe_add_ffmpeg_dll_directories()`,
which registers FFmpeg dirs found on `PATH` (and `CONDA_PREFIX\Library\bin`, and an
explicit `TORCHCODEC_FFMPEG_DIR` override) before loading. So after this fix,
FFmpeg-on-PATH works; if you keep FFmpeg elsewhere, set `TORCHCODEC_FFMPEG_DIR` to
its `bin\`.

### CMake 4.x / pybind11 3.x
Pin `cmake<4` to match TorchCodec CI and avoid CMake-4 policy breakage in older
torch CMake modules. pybind11 3.x resolves fine via `python -m pybind11 --cmakedir`.

## 8. Status

- [x] Branch based on `v0.7.0`
- [x] **Milestone 1 — DONE.** CPU build compiles + links + imports + decodes against
  torch 2.6 (cp311, FFmpeg 7). Decoded `nasa_13013.mp4` → `[3, 270, 480] uint8`.
  Toolchain: CUDA 12.3 + MSVC 14.39 + Ninja; runtime FFmpeg 7.x shared on PATH.
- [x] **Milestone 2 — DONE.** NVDEC GPU decode works: `ENABLE_CUDA=1` build
  (72/72, warnings only; `CudaDeviceInterface.cpp` compiles clean against torch
  2.6) links `CUDA::nppicc/nppig/nppc`+`cudart`; FFmpeg n7.1.5 (`h264_cuvid`) +
  CUDA 12.3 NPP DLLs resolve at runtime. `VideoDecoder('...mp4', device='cuda')[0]`
  → `torch.Size([3, 270, 480]) torch.uint8 cuda:0`.
- [x] NVDEC verified on hardware: `nvidia-smi dmon` showed the `dec` engine at
  63–68% during a CUDA decode loop (≈3.7k fps on the 480p clip); 0% on CPU runs.
  (At 480p, CPU decode is faster in wall-clock — NVDEC's win is high-res /
  multi-stream / keeping frames on-GPU; this is overhead-bound, not a fallback.)
- [x] Plain wheel built: `torchcodec-0.7.0+cu124.torch26-cp311-cp311-win_amd64.whl`.
- [x] NPP-bundling implemented (`TORCHCODEC_BUNDLE_CUDA_DLLS=1`) for self-contained wheels.
- [ ] Validate bundled wheel imports with no CUDA toolkit / NPP on PATH; deploy into ComfyUI (cp311).
- [ ] Tag fork commit + GitHub Release hosting the `.whl`.
- [ ] Milestone 3 — CI producing labeled wheels (optional).
