# Chapter 4: CUDA Examples (CPU vs GPU on the Jetson Orin Nano)

Six short programs that show **why the GPU matters for AI**. Every Python example
races the **CPU** (6 Arm cores) against the **GPU** (1024 CUDA cores) on the same job.

| # | Run | What you'll see |
|---|-----|-----------------|
| 1 | `python3 01_hello_gpu.py` | Your GPU's name, CUDA cores and memory, plus a first calculation on the GPU |
| 2 | `python3 02_cpu_vs_gpu_add.py` | Adding lists of numbers: the CPU wins on small lists, the GPU wins big on huge lists |
| 3 | `python3 03_matrix_race.py` | Matrix multiplication (the maths inside neural networks), including a Tensor Core bonus |
| 4 | `python3 04_edge_filter.py` | Live camera filters (edges / blur / cartoon). Press **G** to switch CPU and GPU and watch the FPS |
| 5 | `python3 05_mandelbrot.py` | Draws a fractal on the CPU and the GPU and saves `mandelbrot.png`. Try `--zoom 100` |
| 6 | `nvcc vector_add.cu -o vector_add && ./vector_add` | A real CUDA C++ kernel: 16.7 million GPU threads, one addition each |

## Setup (same environment as the Chapter 5 projects)
```bash
mkdir ~/ai_workshop && cd ~/ai_workshop
# copy the cuda_examples and cnn_activity folders here

python3 -m venv --system-site-packages env
source env/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/
pip install matplotlib

cd cuda_examples
python3 01_hello_gpu.py
```
The examples use **PyTorch** to talk to CUDA, plus NumPy and OpenCV (both already on the Jetson).

## Notes
- **First GPU call is slow.** CUDA needs a moment to start up, so every example does a warm-up run before timing.
- **Timing the GPU correctly.** GPU work runs in the background, so the scripts call `torch.cuda.synchronize()`
  (wait until the GPU has finished) before stopping the clock.
- **Example 4 without a camera:** `python3 04_edge_filter.py --benchmark` times the filters on a test image.
  Use `--csi` for a CSI camera.
- **Example 6:** if you get `nvcc: command not found`, run `export PATH=/usr/local/cuda/bin:$PATH`.
  You can also add `-arch=sm_87` (the Orin's GPU) to the `nvcc` command.
- **Faster results:** run `sudo nvpmodel -m 0 && sudo jetson_clocks` to put the Jetson in its maximum performance mode (optional).

## What to notice
1. **Small jobs:** the CPU is fine. Starting the GPU has a small cost.
2. **Big, repetitive jobs** (the same small calculation on millions of numbers or pixels): the GPU wins by a lot.
3. **AI is exactly that kind of job.** Neural networks are millions of multiplications, and image filters are the same
   sliding-grid maths as a CNN layer. That's why we train our CNNs on the GPU in Chapter 5.
