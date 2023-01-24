Repo to test PyTorch 2.0 for inference optimization on CPU and GPU

```pip install -r requirements.txt```

## Experiments
This folder contains quick reproducers to observe different behaviors. The base scripts give a benchmark without any optimization. Then we can observe what happens when optimizing different things. This table regroups the total time for 4 steps (all executed on an `Intel(R) Xeon(R) CPU @ 2.20GHz`):

We measure total time here since in diffusion inference we usually to multiple forward passes of the same model.

*For the optimized model the total time is measured after a warmup step.*

Batch size 4, steps 4:

| Script | Toal time |
|:--|:-:|
| base_cpu | 19.36s |
| optimize_cpu | 18.69s |

Batch size 8, steps 4:

| Script | Toal time |
|:--|:-:|
| base_cpu | 31.36s |
| optimize_cpu | 38.69s |
