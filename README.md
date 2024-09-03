# Implementation of Rotary Position Embedding in Triton

This repository contains python code for Rotary Position Embedding(RoPE) implemented in Triton.

Each forward, backward kernel is defined in triton_utils/kernel.py,
and wrapped as a layer in triton_utils/layer.py using torch.autograd.Function.

Regarding RoPE, you can refer to this [paper](https://arxiv.org/pdf/2104.09864).

## Environment

For reproduce, you can build your image using Dockerfile:
```bash
docker build --tag {IMAGE}:{TAG} . 
```

Once it's built, run container as following:
```bash
docker run -it {IMAGE}:{TAG} bash
```
You are all set!

## Testing

You can test triton version of RoPE with following command:

```bash
PYTHONPATH={PATH_TO_REPOSITORY_ROOT} python triton_utils/layer_test.py
```


## Benchmark

You can run *benchmark.py* to compare time comsumption between triton vs torch vs cuda.

```bash
python benchmark.py
```

Once it's done, results will be dumped to "./results"

Current comparison result is as follows:

```bash
# lower is better
RoPE performance:
   dimension_size     Triton      Torch       Cuda
            16.0   0.203776   0.089088   0.076800
            32.0   0.250048   0.181040   0.111616
            64.0   0.342016   0.400160   0.195584
           128.0   0.539648   1.125376   0.368640
           256.0   1.022976   2.401216   0.788480
           512.0   2.040832   4.825088   1.742848
          1024.0   3.968000   9.717248   3.917792
          2048.0   8.015360  19.300575   7.745536
          4096.0  16.146927  39.028225  15.905792
          8192.0  32.131073  75.272324  31.818785
```
RoPE(triton) obviously outperforms torch version of RoPE, but slightly slower than that of cuda fused version.
