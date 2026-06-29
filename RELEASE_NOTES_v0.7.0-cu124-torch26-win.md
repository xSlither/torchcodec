# torchcodec 0.7.0 — Windows / CUDA 12.x / torch 2.6.0 (NVDEC)

An unofficial Windows build of [torchcodec](https://github.com/meta-pytorch/torchcodec)
**v0.7.0**, built from source to be ABI-compatible with **PyTorch 2.6.0+cu124**,
with **NVDEC GPU decoding** enabled. Upstream ships no Windows wheel that works
with torch 2.6 (Windows wheels begin at 0.7 but target torch 2.8; the only
torch-2.6 torchcodec, 0.2, is Linux/macOS-only) — so this fills that exact gap.

## Wheel

`torchcodec-0.7.0+cu124.torch26-cp311-cp311-win_amd64.whl`

| | |
|---|---|
| OS | Windows x64 |
| Python | 3.11 (`cp311`) only — other versions need a rebuild |
| PyTorch | `2.6.0+cu124` (cu124 build) |
| GPU decode | NVDEC, verified (`nvidia-smi` `dec` engine ~65% during decode) |
| Built with | CUDA 12.3 toolkit + MSVC 14.39 + Ninja |

## Install

```bat
pip install torchcodec-0.7.0+cu124.torch26-cp311-cp311-win_amd64.whl
```

You also need **FFmpeg 7.x "shared"** at runtime (this is normal for torchcodec;
it is *not* bundled). Use a build with NVDEC, e.g. BtbN
`ffmpeg-n7.1-latest-win64-gpl-shared` (must contain `avcodec-61.dll` and
`h264_cuvid`), and put its `bin\` on `PATH` (or set `TORCHCODEC_FFMPEG_DIR` to it).

```python
from torchcodec.decoders import VideoDecoder
d = VideoDecoder("video.mp4", device="cuda")   # or "cpu"
frame = d[0]                                     # -> uint8 [C,H,W] on cuda:0
```

## What's bundled (and what isn't)

- **Bundled:** the NVIDIA **NPP** + **cudart** runtime DLLs (NVIDIA
  redistributables) are packaged inside the wheel, so it imports and GPU-decodes
  on machines **without a CUDA toolkit installed** (verified in an embedded
  Python with no toolkit present). You still need a recent NVIDIA driver.
- **Not bundled:** FFmpeg (GPL) — user-provided, as above.

## Notes / caveats

- **Performance:** NVDEC's advantage shows on high-resolution / many-stream
  decoding and when feeding frames straight to GPU models (no host copy). On
  tiny clips, CPU decode can be faster — that's overhead, not a fallback.
- **Version locks:** torch must be `2.6.x` cu124; FFmpeg must be **7.x**
  (`avcodec-61`) — FFmpeg 8 (`avcodec-62/63`) will not load.
- Built from the [`claude/torchcodec-windows-wheels-lqchhj`](../../tree/claude/torchcodec-windows-wheels-lqchhj)
  branch. The full build recipe and every gotcha are in
  [`BUILD_WINDOWS_TORCH26.md`](../../blob/claude/torchcodec-windows-wheels-lqchhj/BUILD_WINDOWS_TORCH26.md).

Unofficial community build; not affiliated with or endorsed by Meta/PyTorch.
Licensed under torchcodec's BSD-3 license. CUDA/NPP redistributables are subject
to the NVIDIA CUDA EULA.
