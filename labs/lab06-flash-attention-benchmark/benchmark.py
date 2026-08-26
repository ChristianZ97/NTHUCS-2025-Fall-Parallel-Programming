# -*- coding: utf-8 -*-
"""FlashAttention benchmark exported from the companion notebook.

This file retains IPython shell commands. Use ``benchmark.ipynb`` as the
primary runnable artifact.
"""

# Install notebook dependencies.
!pip install flash-attn==1.0.9 --no-build-isolation -q

!pip install wandb -q

import torch
!nvidia-smi
print("CUDA Usage :", torch.cuda.is_available())
print("GPU :", torch.cuda.get_device_name(0))

import wandb

import math
import json
import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

# ---------------------------
# FLOPs and Efficiency Calculation
# ---------------------------
def flops(batch_size, seq_len, head_dim, num_heads, causal, mode='fwd'):
    assert mode in ['fwd', 'bwd', 'fwd_bwd']
    f = 4 * batch_size * seq_len**2 * num_heads * head_dim // (2 if causal else 1)
    return f if mode == 'fwd' else (2.5 * f if mode == 'bwd' else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if time > 0 else 0.0

# ---------------------------
# Custom Benchmark Function
# ---------------------------
def benchmark_fwd_bwd(func, qkv, cu_seqlens, dropout_p, causal, repeats=30):
    fwd_times, bwd_times = [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        out = func(qkv, cu_seqlens, qkv.shape[0], dropout_p, causal=causal)
        end.record()
        torch.cuda.synchronize()
        fwd_times.append(start.elapsed_time(end) / 1000.0)

        grad = torch.randn_like(out)
        start.record()
        out.backward(grad, retain_graph=True)
        end.record()
        torch.cuda.synchronize()
        bwd_times.append(start.elapsed_time(end) / 1000.0)

    return fwd_times, bwd_times

# ---------------------------
# PyTorch baseline attention
# ---------------------------
def pytorch_attn_func(qkv, dropout_p=0.0, causal=True):
    batch_size, seq_len, _, num_heads, head_dim = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(head_dim)

    scores = torch.empty(batch_size * num_heads, seq_len, seq_len, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=num_heads)
    if causal:
        causal_mask = torch.triu(torch.full((seq_len, seq_len), -10000.0, device=scores.device), 1)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)

# ---------------------------
# Main Benchmark Function
# ---------------------------
def benchmark_attention(batch_size, seq_len, num_heads, emb_dim, impl, causal, repeats, output):

    # 初始化 wandb
    wandb.init(
        project="2025-PP-Lab6",  # 專案名稱
        config={
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "emb_dim": emb_dim,
            "implementation": impl,
            "causal": causal,
            "repeats": repeats,
            "head_dim": emb_dim // num_heads
        }
    )


    assert impl in ['Pytorch', 'Flash1']
    device = 'cuda'
    dtype = torch.float16
    dropout_p = 0.0
    head_dim = emb_dim // num_heads

    qkv = torch.randn(
        batch_size, seq_len, 3, num_heads, head_dim,
        device=device, dtype=dtype, requires_grad=True
    )

    if impl == 'Flash1':
        total_len = batch_size * seq_len
        cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=device)
        qkv_unpad = rearrange(qkv, 'b s three h d -> (b s) three h d')
        attention_func = lambda x, cu, tot, dp, causal: flash_attn_unpadded_qkvpacked_func(
            x, cu, tot, dropout_p=dp, causal=causal
        )
        input_tensor = qkv_unpad
        cu_input = cu_seqlens
    else:
        attention_func = lambda x, cu, tot, dp, causal: pytorch_attn_func(x, dp, causal)
        input_tensor = qkv
        cu_input = None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    fwd_times, bwd_times = benchmark_fwd_bwd(attention_func, input_tensor, cu_input, dropout_p, causal=causal, repeats=repeats)

    forward_time = sum(fwd_times) / len(fwd_times)
    backward_time = sum(bwd_times) / len(bwd_times)
    peak_memory_usage = torch.cuda.max_memory_allocated() / (1024**2)

    benchmark_result = {
        'forward': {
            'time(s)': forward_time,
            'FLOPS(TFLOPs/s)': efficiency(
                flops(batch_size, seq_len, head_dim, num_heads, causal, mode='fwd'),
                forward_time
            )
        },
        'backward': {
            'time(s)': backward_time,
            'FLOPS(TFLOPs/s)': efficiency(
                flops(batch_size, seq_len, head_dim, num_heads, causal, mode='bwd'),
                backward_time
            )
        },
        'forward_backward': {
            'time(s)': forward_time + backward_time,
            'FLOPS(TFLOPs/s)': efficiency(
                flops(batch_size, seq_len, head_dim, num_heads, causal, mode='fwd_bwd'),
                forward_time + backward_time
            )
        },
        'peak_memory_usage(MB)': peak_memory_usage,
    }


    wandb.log({
        "forward_time": forward_time,
        "backward_time": backward_time,
        "total_time": forward_time + backward_time,
        "forward_tflops": benchmark_result['forward']['FLOPS(TFLOPs/s)'],
        "backward_tflops": benchmark_result['backward']['FLOPS(TFLOPs/s)'],
        "total_tflops": benchmark_result['forward_backward']['FLOPS(TFLOPs/s)'],
        "peak_memory_mb": peak_memory_usage
    })

    with open(output, 'w') as json_file:
        json.dump(benchmark_result, json_file, indent=2)

    print(f"Benchmark completed. Results saved to {output}")
    print(json.dumps(benchmark_result, indent=2))

    wandb.finish()

# Run benchmark for FlashAttention v1
"""
benchmark_attention(
    batch_size=16,
    seq_len=512,
    num_heads=8,
    emb_dim=512,
    impl='Flash1',  # Choose between 'Pytorch' or 'Flash1'
    causal=True,
    repeats=30,
    output='flash1_benchmark.json'
)
"""

configs = [
    # ============================================================
    # 第一組：系統性測試 batch_size 的影響 (固定其他參數)
    # ============================================================
    {"batch_size": 1, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 2, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 4, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 8, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 16, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 32, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 64, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},

    # ============================================================
    # 第二組：系統性測試 seq_len 的影響 (展現 O(N^2) 複雜度)
    # ============================================================
    {"batch_size": 16, "seq_len": 64, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 16, "seq_len": 128, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 16, "seq_len": 256, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 16, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 16, "seq_len": 768, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 16, "seq_len": 1024, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 16, "seq_len": 1536, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 16, "seq_len": 2048, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},

    # ============================================================
    # 第三組：測試不同 num_heads (並行化程度)
    # ============================================================
    # 固定 emb_dim=512
    {"batch_size": 16, "seq_len": 512, "num_heads": 1, "emb_dim": 64, "impl": "Flash1", "causal": True},   # head_dim=64
    {"batch_size": 16, "seq_len": 512, "num_heads": 2, "emb_dim": 128, "impl": "Flash1", "causal": True},  # head_dim=64
    {"batch_size": 16, "seq_len": 512, "num_heads": 4, "emb_dim": 256, "impl": "Flash1", "causal": True},  # head_dim=64
    {"batch_size": 16, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},  # head_dim=64
    {"batch_size": 16, "seq_len": 512, "num_heads": 16, "emb_dim": 1024, "impl": "Flash1", "causal": True}, # head_dim=64
    {"batch_size": 16, "seq_len": 512, "num_heads": 32, "emb_dim": 2048, "impl": "Flash1", "causal": True}, # head_dim=64

    # 固定 emb_dim=512，改變 head_dim
    {"batch_size": 16, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},  # head_dim=64
    {"batch_size": 16, "seq_len": 512, "num_heads": 16, "emb_dim": 512, "impl": "Flash1", "causal": True}, # head_dim=32
    {"batch_size": 16, "seq_len": 512, "num_heads": 32, "emb_dim": 512, "impl": "Flash1", "causal": True}, # head_dim=16

    # ============================================================
    # 第四組：測試不同 emb_dim (模型大小)
    # ============================================================
    {"batch_size": 16, "seq_len": 512, "num_heads": 2, "emb_dim": 128, "impl": "Flash1", "causal": True},   # head_dim=64
    {"batch_size": 16, "seq_len": 512, "num_heads": 4, "emb_dim": 256, "impl": "Flash1", "causal": True},   # head_dim=64
    {"batch_size": 16, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},   # head_dim=64 (BERT-base)
    {"batch_size": 16, "seq_len": 512, "num_heads": 12, "emb_dim": 768, "impl": "Flash1", "causal": True},  # head_dim=64 (GPT-2)
    {"batch_size": 16, "seq_len": 512, "num_heads": 16, "emb_dim": 1024, "impl": "Flash1", "causal": True}, # head_dim=64 (BERT-large)
    {"batch_size": 16, "seq_len": 512, "num_heads": 24, "emb_dim": 1536, "impl": "Flash1", "causal": True}, # head_dim=64
    {"batch_size": 16, "seq_len": 512, "num_heads": 32, "emb_dim": 2048, "impl": "Flash1", "causal": True}, # head_dim=64

    # ============================================================
    # 第五組：Pytorch vs Flash1 詳細對比
    # ============================================================
    # 小規模
    {"batch_size": 8, "seq_len": 256, "num_heads": 8, "emb_dim": 512, "impl": "Pytorch", "causal": True},
    {"batch_size": 8, "seq_len": 256, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},

    # 中等規模
    {"batch_size": 16, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Pytorch", "causal": True},
    {"batch_size": 16, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},

    # 大規模
    {"batch_size": 16, "seq_len": 1024, "num_heads": 8, "emb_dim": 512, "impl": "Pytorch", "causal": True},
    {"batch_size": 16, "seq_len": 1024, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},

    # 超大規模
    {"batch_size": 16, "seq_len": 2048, "num_heads": 8, "emb_dim": 512, "impl": "Pytorch", "causal": True},
    {"batch_size": 16, "seq_len": 2048, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},

    # 不同 emb_dim 對比
    {"batch_size": 16, "seq_len": 512, "num_heads": 12, "emb_dim": 768, "impl": "Pytorch", "causal": True},
    {"batch_size": 16, "seq_len": 512, "num_heads": 12, "emb_dim": 768, "impl": "Flash1", "causal": True},

    # ============================================================
    # 第六組：Causal vs Non-Causal 詳細對比
    # ============================================================
    # Flash1 對比
    {"batch_size": 16, "seq_len": 256, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": False},
    {"batch_size": 16, "seq_len": 256, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},

    {"batch_size": 16, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": False},
    {"batch_size": 16, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},

    {"batch_size": 16, "seq_len": 1024, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": False},
    {"batch_size": 16, "seq_len": 1024, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},

    # Pytorch 對比
    {"batch_size": 16, "seq_len": 256, "num_heads": 8, "emb_dim": 512, "impl": "Pytorch", "causal": False},
    {"batch_size": 16, "seq_len": 256, "num_heads": 8, "emb_dim": 512, "impl": "Pytorch", "causal": True},

    {"batch_size": 16, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Pytorch", "causal": False},
    {"batch_size": 16, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Pytorch", "causal": True},

    # ============================================================
    # 第七組：head_dim 的影響 (固定 emb_dim，改變 num_heads)
    # ============================================================
    {"batch_size": 16, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},   # head_dim=64
    {"batch_size": 16, "seq_len": 512, "num_heads": 16, "emb_dim": 512, "impl": "Flash1", "causal": True},  # head_dim=32
    {"batch_size": 16, "seq_len": 512, "num_heads": 32, "emb_dim": 512, "impl": "Flash1", "causal": True},  # head_dim=16

    {"batch_size": 16, "seq_len": 512, "num_heads": 16, "emb_dim": 1024, "impl": "Flash1", "causal": True}, # head_dim=64
    {"batch_size": 16, "seq_len": 512, "num_heads": 32, "emb_dim": 1024, "impl": "Flash1", "causal": True}, # head_dim=32
    {"batch_size": 16, "seq_len": 512, "num_heads": 64, "emb_dim": 1024, "impl": "Flash1", "causal": True}, # head_dim=16

    # ============================================================
    # 第八組：交叉測試 - batch_size x seq_len
    # ============================================================
    {"batch_size": 4, "seq_len": 256, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 4, "seq_len": 1024, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 32, "seq_len": 256, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 32, "seq_len": 1024, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},

    # ============================================================
    # 第九組：真實模型配置測試
    # ============================================================
    # BERT-base
    {"batch_size": 32, "seq_len": 512, "num_heads": 12, "emb_dim": 768, "impl": "Pytorch", "causal": False},
    {"batch_size": 32, "seq_len": 512, "num_heads": 12, "emb_dim": 768, "impl": "Flash1", "causal": False},

    # GPT-2 small
    {"batch_size": 16, "seq_len": 1024, "num_heads": 12, "emb_dim": 768, "impl": "Pytorch", "causal": True},
    {"batch_size": 16, "seq_len": 1024, "num_heads": 12, "emb_dim": 768, "impl": "Flash1", "causal": True},

    # BERT-large
    {"batch_size": 16, "seq_len": 512, "num_heads": 16, "emb_dim": 1024, "impl": "Pytorch", "causal": False},
    {"batch_size": 16, "seq_len": 512, "num_heads": 16, "emb_dim": 1024, "impl": "Flash1", "causal": False},

    # GPT-2 medium
    {"batch_size": 8, "seq_len": 1024, "num_heads": 16, "emb_dim": 1024, "impl": "Pytorch", "causal": True},
    {"batch_size": 8, "seq_len": 1024, "num_heads": 16, "emb_dim": 1024, "impl": "Flash1", "causal": True},

    # ============================================================
    # 第十組：記憶體壓力測試 (逐步增加)
    # ============================================================
    {"batch_size": 64, "seq_len": 256, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 64, "seq_len": 512, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 32, "seq_len": 1024, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
    {"batch_size": 16, "seq_len": 2048, "num_heads": 8, "emb_dim": 512, "impl": "Flash1", "causal": True},
]

# 執行所有實驗
for i, config in enumerate(configs):
    head_dim = config["emb_dim"] // config["num_heads"]

    print(f"\n{'='*60}")
    print(f"Experiment {i+1}/{len(configs)}")
    print(f"{'='*60}")
    print(f"Config: batch_size={config['batch_size']}, seq_len={config['seq_len']}, "
          f"num_heads={config['num_heads']}, emb_dim={config['emb_dim']}")
    print(f"Implementation: {config['impl']}, Causal: {config['causal']}")
    print(f"head_dim = {head_dim}")
    print(f"{'='*60}\n")

    try:
        benchmark_attention(
            batch_size=config["batch_size"],
            seq_len=config["seq_len"],
            num_heads=config["num_heads"],
            emb_dim=config["emb_dim"],
            impl=config["impl"],
            causal=config["causal"],
            repeats=30,
            output=f'benchmark_exp{i+1:03d}.json'
        )
    except Exception as e:
        print(f"Experiment {i+1} failed with error: {e}")
        continue

print(f"\nAll {len(configs)} experiments completed!")
