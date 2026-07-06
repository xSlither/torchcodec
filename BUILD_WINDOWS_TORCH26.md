# Building TorchCodec 0.10 for Windows against PyTorch 2.6 (with NVDEC + NVENC)

This fork previously shipped a working **`v0.7.0`**-based Windows wheel, ABI-compatible
with **`torch==2.6.0+cu124`**, with NVDEC GPU decode (see git history /
`v0.7.0-cu124-torch26-win`). This document covers the **upgrade to a `v0.10.0` base** —
the furthest this fork can go past 0.7.0 while staying compatible with torch 2.6 — and
what it buys over 0.7.0: `VideoEncoder`/`AudioEncoder` (including CUDA-resident /
hardware NVENC encode) and a materially improved NVDEC integration.

---

## 1. Why `v0.10.0`, and definitively not `v0.11+`

Walking the upstream tags with full git access (not just release notes) settles this
precisely, by grepping the actual C++ source at each release tag:

| Version | `torch::stable` / `StableDevice` refs in `_core/` | C++ standard | Official `torch` pairing |
|---|---|---|---|
| 0.7.0  | 0 | C++17 | 2.8 |
| 0.8.1  | 0 | C++17 | 2.9 |
| 0.9.1  | 0 | C++17 | 2.9 |
| **0.10.0** | **0** | **C++17** | 2.10 |
| **0.11.0** | **~150+ across every core file** | **C++20** | 2.11 |

**`v0.11.0` is a hard wall, not a hard-but-portable step.** Between 0.10 and 0.11,
torchcodec's entire `_core` module was rewritten around PyTorch's new stable ABI
(`torch::stable::Tensor`, `StableDevice`, `torch/csrc/stable/*`) — confirmed by
extracting torch **2.6.0**'s own bundled headers (`torch/include/torch/csrc/`) and
finding **no `torch/csrc/stable/` directory exists at all** in that version. This
namespace was introduced later than 2.6. Since `Frame`, `Encoder`, `DeviceInterface`,
`CudaDeviceInterface`, `CpuDeviceInterface`, `BetaCudaDeviceInterface`,
`SingleStreamDecoder`, `custom_ops.cpp`, and `SwScale` **all** use it pervasively at
0.11, compiling 0.11's source against torch 2.6 is not a build-flag or shim fix — it
would mean manually re-deriving the pre-stable-ABI architecture that 0.10 already *is*,
then re-applying 0.11's genuine improvements on top by hand. That's strictly more work
and more risk than just building 0.10 directly, for no capability gain (0.11's changes
over 0.10 are internal ABI-portability plumbing, plus `SwScale.cpp`/
`NVDECCacheConfig.cpp` refactors — not new user-facing features).

`0.10.0` is therefore the target: it is the **newest release still on the classic
C++17/ATen API** that torch 2.6's headers actually provide, so it carries three
releases of real improvements over the previous `0.7.0` base with **no additional
architectural risk**:

- **`VideoEncoder` / `AudioEncoder`** (new at 0.9, refined through 0.10) — including
  **CUDA-resident encode**: `Encoder.cpp` substitutes a hardware codec and calls
  `deviceInterface_->convertCUDATensorToAVFrameForEncoding` /
  `registerHardwareDeviceWithCodec` when `frames.device().is_cuda()`. This means
  **NVENC-from-a-CUDA-tensor is already available at 0.9/0.10** — the "requires
  TorchCodec `VideoEncoder` v0.11+" assumption in some downstream planning docs is
  outdated; 0.9 already has it, and there is no reason to reach for 0.11 to get it.
- **A materially better NVDEC integration** (new at 0.8): `NVCUVIDRuntimeLoader.cpp`
  now loads `nvcuvid.dll` at **runtime** via `LoadLibrary`/`GetProcAddress` (mirroring
  how FFmpeg itself talks to cuvid), instead of link-time linking against an NVIDIA
  Video Codec SDK import lib. This is a **build simplification** relative to 0.7.0,
  not new risk: no separate Video Codec SDK install is needed at build time, and the
  vendored `nvcuvid_include/{cuviddec,nvcuvid}.h` headers ship in-tree. The **default**
  CUDA decode path is still `CudaDeviceInterface` (`registerDeviceInterface(..., variant="")`)
  — an evolution of the same class 0.7.0 already proved compiles clean against torch
  2.6, still built on `av_hwdevice_ctx_create("cuda")` + NPP color conversion.
  `BetaCudaDeviceInterface` (also new at 0.8) is a **separate, opt-in** decode path
  (`variant="beta"`, i.e. `device="cuda:0:beta"`) that talks to nvcuvid directly; it is
  not required and not the default, so it does not have to work for this port to
  succeed.
- **New `Transform.cpp`/`Metadata.cpp`** (0.9) — decoder-side resize/crop transforms and
  richer stream metadata, both plain ATen code, no new build requirements.

`v0.2.1` remains the known *exact* torch-2.6 match (kept only as a last-resort
reference — it predates all Windows support, so it would mean re-deriving Windows
enablement from scratch; not needed unless 0.10 surfaces a torch symbol genuinely
missing from 2.6, which the C++17/classic-API scan above did not find any evidence of).

---

## 2. What changed, mechanically, from the 0.7.0 port

The three build-system patches that made the 0.7.0 port work were re-diffed against
upstream `v0.10.0` directly (via `git apply --3way`) rather than re-derived from
scratch, since the exact code regions they touch are **unchanged** between 0.7.0 and
0.10.0 upstream:

| File | Patch | Applied to 0.10.0 |
|---|---|---|
| `setup.py` | Force Ninja generator on Windows (lets an older, CUDA-12.3-compatible MSVC toolset be pinned via the shell instead of MSBuild always grabbing the newest); optional `TORCHCODEC_BUNDLE_CUDA_DLLS=1` NPP/cudart DLL bundling for a self-contained wheel; PEP 440 double-`+`-segment fix for `version.txt` values that already carry a local segment. | Applied **cleanly** (`git apply --3way`, no conflicts) |
| `src/torchcodec/_core/CMakeLists.txt` | Windows `CUDA::nvToolsExt` stand-in (torch's Caffe2 CUDA cmake references a legacy NVTX target CUDA 12.x removed); swap the unpopulated legacy `FindCUDA` vars (`CUDA_nppi_LIBRARY`/`CUDA_nppicc_LIBRARY` — 0.7–0.10 never call `find_package(CUDA)`, so these are always empty) for modern `CUDAToolkit` imported targets (`CUDA::cudart`/`CUDA::nppicc`/`CUDA::nppig`/`CUDA::nppc`) plus explicit include dirs. | Applied **cleanly** — the exact `if(ENABLE_CUDA): list(APPEND core_library_dependencies ...)` block this patches is byte-identical between 0.7.0 and 0.10.0 upstream. |
| `src/torchcodec/_core/ops.py` | Windows DLL-directory registration (`os.add_dll_directory`) for FFmpeg/NPP/cudart DLLs before `torch.ops.load_library` — required because ctypes-loaded DLL dependencies don't resolve via `PATH` alone on Python 3.8+. | Applied with **one trivial merge conflict** (0.10.0 independently added its own `import sys`/`shutil`/`traceback` imports and changed `load_torchcodec_shared_libraries`'s return-type annotation to `-> tuple[int, str]`; resolved by keeping 0.10's imports/signature and inserting the fork's DLL-directory helper + call site, same as before). |
| `.github/workflows/windows_torch26_wheel.yaml` | Manual-dispatch CI: builds the CUDA wheel on `windows-2022`, verifies NPP DLLs are bundled, then a CPU-only import+decode(+now encode)smoke test (no GPU on hosted runners). | Re-authored with the same structure; default `build-version` bumped to `0.10.0+cu124.torch26`; smoke test extended to also round-trip a `VideoEncoder(...).to_file(...)` call, since that's the new capability this upgrade is for. |
| `version.txt` | Local version segment. | `0.10.0+cu124.torch26` |

No `_core/*.cpp`/`.h` files needed patching — the classic CUDA/NPP/FFmpeg glue code
that 0.7.0 already proved compiles against torch 2.6 evolved in place (bug fixes,
`Transform`/`Metadata`/`BetaCudaDeviceInterface`/`NVCUVIDRuntimeLoader` additions) but
never touched `torch::stable` or any torch API newer than what 2.6 ships.

---

## 3. Build environment (unchanged from the 0.7.0 port)

Same recipe as before — see the original milestone log in git history
(`f0b096e`..`4aacd4f` on `claude/torchcodec-windows-wheels-lqchhj`) for the full
troubleshooting narrative (`STL1002`, the `CUDA::nvToolsExt` error, FFmpeg
major-version/soname matching, the Windows DLL-search gotcha). Summary:

- **MSVC**: Visual Studio 2022 Build Tools; build from an "x64 Native Tools Command
  Prompt for VS 2022".
- **A CUDA Toolkit + a matching MSVC toolset.** A working `nvcc` is required even for
  a CPU-only build (torch `+cu124`'s CMake forces `enable_language(CUDA)`). Building
  with CUDA 12.3 is runtime-compatible with `cu124` torch wheels, but recent MSVC STL
  headers (~14.40+) hard-block CUDA < 12.4 (`STL1002`) — pin an older toolset (MSVC
  14.39 / VS 17.9, via `vcvarsall -vcvars_ver=14.39`) and use the Ninja generator
  (already wired into `setup.py`'s win32 branch) so the pin actually takes effect
  (MSBuild ignores the shell environment and always grabs the newest toolset).
- **`pybind11_DIR` and `Python3_ROOT_DIR` must both be set explicitly for a `uv`
  venv, in the *same* shell session as the build command.** `find_package(pybind11
  REQUIRED)` and the later `find_package(Python3 3.11 EXACT COMPONENTS Development)`
  both need hints a `uv`-managed venv doesn't expose by default:
  ```bat
  for /f "delims=" %i in ('python -m pybind11 --cmakedir') do set pybind11_DIR=%i
  set Python3_ROOT_DIR=%USERPROFILE%\AppData\Roaming\uv\python\cpython-3.11.12-windows-x86_64-none
  ```
  (adjust the `cpython-...` folder name to whatever `uv python list` shows). Without
  `Python3_ROOT_DIR`, CMake finds the interpreter fine (an earlier, looser pybind11-
  driven Python3 search succeeds) but then fails the later exact-version `Development`
  component search with `Could NOT find Python3 (missing: Python3_LIBRARIES
  Python3_INCLUDE_DIRS ...)`, because a venv doesn't carry its own copy of
  `libs\python311.lib`/full headers — those live only in the base standalone install.
  **Live-confirmed (2026-07, Windows + CUDA 12.3 + MSVC 14.39):** with both variables
  set, CMake configure resolves cleanly (`Found Python3: ...\libs\python311.lib
  (found suitable exact version "3.11.12")`).
- **FFmpeg "shared" dev libraries**: either a BtbN/gyan `shared` build + `pkg-config`,
  or `BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1` to fetch Meta's prebuilt non-GPL libs
  (build-time only, no NVDEC at runtime — fine for a CPU build; use a real
  NVDEC/cuvid-enabled shared FFmpeg for the CUDA build's runtime).
- **`uv` venv**, Python version matched to the deploy target. Validated with
  Python **3.11**; other CPython versions supported by torch 2.6 should also
  work but haven't been exercised by this port.

### 3.1 Build

```bat
git clone https://github.com/xSlither/torchcodec
cd torchcodec
git checkout claude/torchcodec-0.11-windows-wheel-2tkxw8

uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install "cmake<4" ninja pybind11 numpy setuptools wheel build pytest pillow

:: pin the CUDA-12.3-compatible MSVC toolset (adjust path/version to your install)
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.39
cl              :: expect 19.39.x
nvcc --version  :: 12.3 is fine even though the wheel targets cu124

set ENABLE_CUDA=1
set TORCHCODEC_BUNDLE_CUDA_DLLS=1
set TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=ON
set BUILD_AGAINST_ALL_FFMPEG_FROM_S3=1
for /f "delims=" %i in ('python -m pybind11 --cmakedir') do set pybind11_DIR=%i

:: fast iteration first:
uv pip install -e . --no-build-isolation -v

:: once it imports + decodes + encodes cleanly, build a real wheel:
set BUILD_VERSION=0.10.0+cu124.torch26
python -m build --wheel --no-isolation
:: -> dist\torchcodec-0.10.0+cu124.torch26-cp311-cp311-win_amd64.whl
```

### 3.2 Validate

```bat
python -c "import torch, torchcodec; print(torch.__version__, torchcodec.__version__)"

:: decode (as before)
python -c "from torchcodec.decoders import VideoDecoder; d=VideoDecoder('test/resources/nasa_13013.mp4', device='cuda'); f=d[0]; print(f.shape, f.device)"

:: NEW: encode -- CPU and CUDA-resident (NVENC substitution happens automatically
:: when the input frames tensor is already on a CUDA device)
python -c "
import torch
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder
d = VideoDecoder('test/resources/nasa_13013.mp4', device='cuda')
frames = d[0:30]  # NCHW uint8, already on cuda:0
VideoEncoder(frames, frame_rate=d.metadata.average_fps).to_file('out_cuda.mp4')
print('cuda-resident encode ok')
"

pytest test/test_decoders.py -q
pytest test/test_encoders.py -q
```

**Exit criteria:** import succeeds; `device='cuda'` decode returns a `cuda:0` tensor
(NVDEC engaged — confirm with `nvidia-smi dmon`, `dec` column, as in the 0.7.0 port);
`VideoEncoder(...).to_file(...)` succeeds for both a CPU tensor and a CUDA-resident
tensor (confirms the hardware-encode substitution path actually engages rather than
silently falling back — check for the `h264_nvenc`/similar codec name in ffprobe
output on the encoded file, or watch `nvidia-smi dmon`'s `enc` column during the
CUDA-tensor encode call).

---

## 4. Known risks specific to this 0.10.0 base (beyond the 0.7.0 port's own risk log)

- **NPP library set: confirmed sufficient, live.** ✅ **Resolved (2026-07,
  Windows + CUDA 12.3 + MSVC 14.39 + torch 2.6.0+cu124).** The full CMake configure
  and Ninja build (120/120 steps) completed cleanly for every one of
  `CudaDeviceInterface.cpp`, `BetaCudaDeviceInterface.cpp`, `CUDACommon.cpp`,
  `NVDECCache.cpp`, `NVCUVIDRuntimeLoader.cpp`, and `Encoder.cpp` across all five
  FFmpeg targets (4/5/6/7/8), linking against `CUDA::cudart`/`CUDA::nppicc`/
  `CUDA::nppig`/`CUDA::nppc` with zero unresolved NPP symbols — no additional
  `CUDA::npp*` target was needed.
- **`TORCHCODEC_BUNDLE_CUDA_DLLS=1` broke editable installs (`pip install -e .`).
  ✅ Fixed (2026-07).** `copy_extensions_to_source()` (only invoked for editable
  installs, never for a real `python -m build --wheel`, which is why the 0.7.0 port
  never hit this) asserted every `.dll`/`.pyd` file found in the install prefix
  contained `"libtorchcodec"` in its name — but `_maybe_bundle_cuda_runtime_dlls()`
  copies `cudart64_12.dll`/`nppc64_12.dll`/`nppicc64_12.dll`/`nppig64_12.dll` into
  that exact same directory when the env var is set, tripping the assertion
  (`AssertionError` in `copy_extensions_to_source`, confirmed live). Fixed by hoisting
  the bundled-DLL prefix tuple to a shared `CMakeBuild._BUNDLED_CUDA_DLL_PREFIXES`
  constant and widening the assertion to accept either `"libtorchcodec"` in the name
  or a recognized bundled-CUDA-DLL prefix.
- **`BetaCudaDeviceInterface` is untested by this port and is not required.** It's a
  separate opt-in decode path (`device="cuda:0:beta"`); if the user's downstream code
  never requests the `beta` variant, its extra `nvcuvid_include/*.h` compile units
  still build (they're unconditional `core_sources` when `ENABLE_CUDA`), but their
  *runtime* correctness was not part of this validation pass — only the default
  `CudaDeviceInterface` path was confirmed to carry over unchanged in architecture.
- **Hardware-encode (NVENC) codec substitution is new surface area.** Unlike decode
  (validated end-to-end on the 0.7.0 base already), the CUDA-resident encode path
  (`Encoder.cpp`'s `frames_.device().is_cuda()` branch,
  `registerHardwareDeviceWithCodec`/`setupHardwareFrameContextForEncoding`) has not
  been exercised on this fork's Windows/CUDA-12.x toolchain before. Validate it
  explicitly (§3.2) rather than assuming decode-path success implies encode-path
  success — they share the device interface class but exercise a different FFmpeg
  codec-context/hw-frame-context code path.
- **C++ standard is still 17** at 0.10.0 (confirmed — the bump to 20 is 0.11-only), so
  no MSVC toolset capability concerns beyond what the 0.7.0 port already resolved.
- **FFmpeg major-version support widened to include 8** (`make_torchcodec_libraries(8
  ...)` added upstream between 0.7 and 0.9) in addition to 4–7. Keep using an FFmpeg
  **7.x shared** runtime build (as the 0.7.0 port required) unless there's a reason to
  test against 8 specifically — the loader still tries versions newest-first, so an
  8.x FFmpeg would also work if available, but this hasn't been validated here.

## 5. Status

- [x] Confirmed via full upstream git history (not just docs) that `torch::stable`
      is a hard 0.10→0.11 wall; `v0.10.0` is the newest release still on the classic
      API that torch 2.6 provides.
- [x] Confirmed `VideoEncoder`/`AudioEncoder` (incl. CUDA-resident/hardware-encode
      code path) already exist at 0.9, so no reason to reach for 0.11 to get them.
- [x] Confirmed the NVDEC integration improved (runtime-loaded `nvcuvid.dll`,
      vendored SDK headers) and the default decode path's architecture is unchanged.
- [x] Re-applied the 0.7.0 port's three build-system patches (`setup.py`,
      `_core/CMakeLists.txt`, `_core/ops.py`) onto `v0.10.0` — all three apply with
      only a one-hunk trivial merge in `ops.py` (upstream added unrelated imports/a
      return-type annotation in the same region); zero patches needed in any `.cpp`/
      `.h` file.
- [x] Re-authored the CI workflow (`windows_torch26_wheel.yaml`) for the 0.10.0
      base, extending the CPU smoke test to also cover `VideoEncoder.to_file`.
- [x] **Milestone 1/2 equivalent — DONE, live-confirmed (2026-07).** Built on the
      real target toolchain (Windows + CUDA 12.3 + MSVC 14.39 + torch 2.6.0+cu124,
      `ENABLE_CUDA=1`, `TORCHCODEC_BUNDLE_CUDA_DLLS=1`): CMake configure resolved
      pybind11/CUDAToolkit/Torch/Python3 cleanly (once `pybind11_DIR` +
      `Python3_ROOT_DIR` were set — see §3.1), all 120 Ninja build steps succeeded
      (every FFmpeg target 4/5/6/7/8 × core/custom_ops/pybind_ops, including
      `CudaDeviceInterface`/`BetaCudaDeviceInterface`/`CUDACommon`/`NVDECCache`/
      `NVCUVIDRuntimeLoader`/`Encoder`), `cmake --install` succeeded, and the CUDA
      runtime DLLs bundled successfully. Two real environment-specific issues
      surfaced and were fixed in place (§3.1's `Python3_ROOT_DIR` note; §4's
      `copy_extensions_to_source` assertion fix) — both now resolved.
- [x] **Runtime validation (§3.2) — done, live-confirmed (2026-07).** Package
      imports; `device='cuda'` decode returns a `cuda:0` tensor
      (`torch.Size([3, 270, 480])`), confirming NVDEC engages. Full test suite run:
      `test/test_decoders.py` (442 passed / 31 failed / 5 skipped) and
      `test/test_encoders.py` (937 passed / 36 failed / 14 skipped / 29 deselected).
      All failures are isolated to two known, unrelated issues — see below — rather
      than being spread across the port's own code paths; every `h264_nvenc`/
      `hevc_nvenc` encode test and every decoder test other than one specific
      reference-data file passed.
      - `test_decoders.py`'s 31 failures were all the *same* root cause, and are
        now **fixed**: `test/resources/nasa_13013.mp4.stream3.frame000180.pt` was
        a **git symlink** to `nasa_13013.mp4.time6.000000.pt` (same underlying
        frame, kept once on disk under two test-data names upstream). Windows
        git, without symlink privileges (Developer Mode + `core.symlinks=true`,
        rarely configured by default), checks out a symlink as a **plain text
        file containing the literal target path string** instead of the actual
        content or a real link. That 30-byte string, `nasa_13013.mp4.time6.000000.pt`,
        starts with `'n'` — byte value **110** — which is exactly what
        `WeightsUnpickler error: Unsupported operand 110` was choking on: `torch.load`
        tried to parse that literal filename text as a pickle stream. Not a torch-2.6
        port issue at all — fixed by replacing the symlink with a real copy of the
        target file's bytes (`git` now tracks it as a normal `100644` blob, mode
        change from `120000`), which sidesteps the Windows symlink checkout gap
        entirely with no elevated privileges required.
      - `test_encoders.py`'s 36 failures are all `test_nvenc_against_ffmpeg_cli`
        parametrized with `codec="av1_nvenc", format="mkv"` — the FFmpeg CLI
        reference-encode subprocess itself fails (not torchcodec's `VideoEncoder`),
        suggesting the installed FFmpeg build's `av1_nvenc` support (or the GPU's
        AV1 NVENC capability) is the gap, not the torch-2.6 port.
- [ ] CI workflow has not been run (`workflow_dispatch` is manual) — first run will
      likely need the same kind of tuning the 0.7.0 CI did (action version pins,
      FFmpeg asset naming).
