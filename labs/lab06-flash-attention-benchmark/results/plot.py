import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

# ============================================================
# 初始化設定
# ============================================================
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 100
})

output_dir = './plot_results'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 讀取資料
# ============================================================
csv_files = glob.glob('./*.csv')
if not csv_files:
    print("❌ 找不到 CSV 檔案！")
    exit()

df = pd.read_csv(csv_files[0])
print(f"✓ 載入 {len(df)} 筆資料，欄位: {list(df.columns)}\n")

# ============================================================
# 輔助函數
# ============================================================
def add_value_labels(ax, spacing=5):
    """在圖表上添加數值標籤"""
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=spacing, fontsize=9)

def add_line_labels(ax, x_data, y_data, offset=0.02):
    """在折線圖上添加數值標籤"""
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for x, y in zip(x_data, y_data):
        if not np.isnan(y):
            ax.text(x, y + y_range * offset, f'{y:.3f}', 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')

def save_plot(name):
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {name}")

# ============================================================
# 圖 1: Sequence Length vs Time (O(N²))
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
flash = df[(df['implementation'] == 'Flash1') & (df['causal'] == True) & (df['batch_size'] == 16)]
pytorch = df[(df['implementation'] == 'Pytorch') & (df['causal'] == True) & (df['batch_size'] == 16)]

for data, label, color, marker in [(flash, 'Flash1', '#3498db', 'o'), 
                                     (pytorch, 'Pytorch', '#e74c3c', 's')]:
    if not data.empty:
        grouped = data.groupby('seq_len')['total_time'].mean().sort_index()
        ax.plot(grouped.index, grouped.values, marker=marker, label=label, 
               linewidth=2.5, markersize=8, color=color, alpha=0.85)
        add_line_labels(ax, grouped.index, grouped.values)

ax.set_xlabel('Sequence Length')
ax.set_ylabel('Total Time (s)')
ax.set_title('Sequence Length vs Time (O(N²) Complexity)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
save_plot('01_seq_len_vs_time.png')

# ============================================================
# 圖 2: Sequence Length vs TFLOPs
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
for data, label, color, marker in [(flash, 'Flash1', '#3498db', 'o'), 
                                     (pytorch, 'Pytorch', '#e74c3c', 's')]:
    if not data.empty:
        grouped = data.groupby('seq_len')['total_tflops'].mean().sort_index()
        ax.plot(grouped.index, grouped.values, marker=marker, label=label, 
               linewidth=2.5, markersize=8, color=color, alpha=0.85)
        add_line_labels(ax, grouped.index, grouped.values)

ax.set_xlabel('Sequence Length')
ax.set_ylabel('TFLOPs/s')
ax.set_title('Sequence Length vs Throughput', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
save_plot('02_seq_len_vs_tflops.png')

# ============================================================
# 圖 3: Batch Size vs Time
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
batch_data = df[(df['implementation'] == 'Flash1') & (df['seq_len'] == 512) & (df['causal'] == True)]
if not batch_data.empty:
    grouped = batch_data.groupby('batch_size')['total_time'].mean().sort_index()
    ax.plot(grouped.index, grouped.values, 'o-', linewidth=2.5, 
           markersize=8, color='#27ae60', alpha=0.85)
    add_line_labels(ax, grouped.index, grouped.values)

ax.set_xlabel('Batch Size')
ax.set_ylabel('Total Time (s)')
ax.set_title('Batch Size vs Time (Flash1)', fontweight='bold')
ax.grid(True, alpha=0.3)
save_plot('03_batch_size_vs_time.png')

# ============================================================
# 圖 4: Flash1 Speedup over Pytorch
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6)) # 稍微加寬以容納標籤
seq_lengths = sorted([s for s in df['seq_len'].unique() if not np.isnan(s)])
speedups = []

# 計算加速比
for seq_len in seq_lengths:
    pt = df[(df['implementation'] == 'Pytorch') & (df['seq_len'] == seq_len) & 
            (df['batch_size'] == 16) & (df['causal'] == True)]['total_time']
    fl = df[(df['implementation'] == 'Flash1') & (df['seq_len'] == seq_len) & 
            (df['batch_size'] == 16) & (df['causal'] == True)]['total_time']
    speedups.append(pt.values[0] / fl.values[0] if len(pt) > 0 and len(fl) > 0 else np.nan)

bars = ax.bar(range(len(seq_lengths)), speedups, color='#3498db', alpha=0.8, edgecolor='black', width=0.6)

# 設定 X 軸標籤為 "Seq=..."
labels = [f"Seq={int(s)}" for s in seq_lengths]
ax.set_xticks(range(len(seq_lengths)))
ax.set_xticklabels(labels, rotation=0, fontsize=10)

ax.set_ylabel('Speedup Ratio')
ax.set_title('Flash1 Speedup over Pytorch (Batch=16)', fontweight='bold')
ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.6)
ax.grid(True, alpha=0.3, axis='y')
add_value_labels(ax, spacing=2) # 顯示數值
save_plot('04_flash1_speedup.png')

# ============================================================
# 圖 5: Causal vs Non-Causal
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
for causal, label, color, marker in [(True, 'Causal', '#e74c3c', 'o'), 
                                      (False, 'Non-Causal', '#3498db', 's')]:
    data = df[(df['implementation'] == 'Flash1') & (df['causal'] == causal) & 
              (df['batch_size'] == 16) & (df['num_heads'] == 8)]
    if not data.empty:
        grouped = data.groupby('seq_len')['total_time'].mean().sort_index()
        ax.plot(grouped.index, grouped.values, marker=marker, label=label, 
               linewidth=2.5, markersize=8, color=color, alpha=0.85)
        add_line_labels(ax, grouped.index, grouped.values, offset=0.03)

ax.set_xlabel('Sequence Length')
ax.set_ylabel('Total Time (s)')
ax.set_title('Causal vs Non-Causal Masking (Flash1)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
save_plot('05_causal_vs_noncausal.png')

# ============================================================
# 圖 6: Number of Heads vs TFLOPs
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
heads_data = df[(df['implementation'] == 'Flash1') & (df['seq_len'] == 512) & 
                (df['batch_size'] == 16) & (df['causal'] == True)]

if not heads_data.empty:
    grouped = heads_data.groupby('num_heads')['total_tflops'].mean().sort_index()
    
    # 使用 range 作為 x 座標，讓間距均勻
    x_pos = range(len(grouped))
    bars = ax.bar(x_pos, grouped.values, color='#e67e22', alpha=0.8, 
                  edgecolor='black', width=0.6)
    
    # 設定 X 軸標籤
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"H={int(h)}" for h in grouped.index], fontsize=11)
    
    add_value_labels(ax, spacing=2)

ax.set_ylabel('TFLOPs/s')
ax.set_title('Number of Heads vs Throughput', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
save_plot('06_num_heads_vs_tflops.png')

# ============================================================
# 圖 7: Embedding Dimension vs Time
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
emb_data = df[(df['implementation'] == 'Flash1') & (df['seq_len'] == 512) & 
              (df['batch_size'] == 16) & (df['causal'] == True)]
if not emb_data.empty:
    grouped = emb_data.groupby('emb_dim')['total_time'].mean().sort_index()
    ax.plot(grouped.index, grouped.values, 'o-', linewidth=2.5, 
           markersize=8, color='#9b59b6', alpha=0.85)
    add_line_labels(ax, grouped.index, grouped.values)

ax.set_xlabel('Embedding Dimension')
ax.set_ylabel('Total Time (s)')
ax.set_title('Embedding Dimension vs Time', fontweight='bold')
ax.grid(True, alpha=0.3)
save_plot('07_emb_dim_vs_time.png')

# ============================================================
# 圖 8: Memory Usage Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))
memory_data = df[(df['batch_size'] == 16) & (df['causal'] == True)]
colors = {'Flash1': '#3498db', 'Pytorch': '#e74c3c'}
markers = {'Flash1': 'o', 'Pytorch': 's'}

for impl in ['Flash1', 'Pytorch']:
    impl_data = memory_data[memory_data['implementation'] == impl]
    if not impl_data.empty:
        grouped = impl_data.groupby('seq_len')['peak_memory_mb'].mean().sort_index()
        ax.plot(grouped.index, grouped.values, marker=markers[impl], label=impl, 
               linewidth=2.5, markersize=8, color=colors[impl], alpha=0.85)
        add_line_labels(ax, grouped.index, grouped.values, offset=0.04)

ax.set_xlabel('Sequence Length')
ax.set_ylabel('Peak Memory (MB)')
ax.set_title('Memory Usage: Flash1 vs Pytorch', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
save_plot('08_memory_usage.png')

# ============================================================
# 圖 9: Forward vs Backward Time
# ============================================================
fig, ax = plt.subplots(figsize=(14, 7)) # 加寬很多，因為標籤很長

# 選擇前 6 筆數據 (或者你可以自定義篩選條件)
time_data = df[(df['implementation'] == 'Flash1') & (df['causal'] == True)].head(6)

if len(time_data) > 0:
    x = np.arange(len(time_data))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, time_data['forward_time'], width, label='Forward', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, time_data['backward_time'], width, label='Backward', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    add_value_labels(ax, spacing=2)
    
    # 製作詳細標籤：換行顯示 BS, Seq, Head
    labels = []
    for _, row in time_data.iterrows():
        label = f"BS={int(row['batch_size'])}\nSeq={int(row['seq_len'])}\nHead={int(row['num_heads'])}"
        labels.append(label)
        
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=10) # 不旋轉，靠換行
    
    ax.set_ylabel('Time (s)')
    ax.set_title('Forward vs Backward Time Breakdown', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

save_plot('09_forward_vs_backward.png')

# ============================================================
# 圖 10: Head Dimension vs TFLOPs
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
head_dim_data = df[(df['implementation'] == 'Flash1') & (df['seq_len'] == 512) & 
                   (df['batch_size'] == 16) & (df['causal'] == True)]

if not head_dim_data.empty:
    grouped = head_dim_data.groupby('head_dim')['total_tflops'].mean().sort_index()
    
    x_pos = range(len(grouped))
    bars = ax.bar(x_pos, grouped.values, color='#9b59b6', alpha=0.8, 
                  edgecolor='black', width=0.6)
    
    # 設定 X 軸標籤
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Dim={int(d)}" for d in grouped.index], fontsize=11)
    
    add_value_labels(ax, spacing=2)

ax.set_ylabel('TFLOPs/s')
ax.set_title('Head Dimension vs Throughput', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
save_plot('10_head_dim_vs_tflops.png')

# ============================================================
# 圖 11: Performance Heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
heatmap_data = df[(df['implementation'] == 'Flash1') & (df['causal'] == True)]
pivot = heatmap_data.pivot_table(values='total_tflops', index='batch_size', 
                                  columns='seq_len', aggfunc='mean')

if not pivot.empty:
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'TFLOPs/s'}, linewidths=1, ax=ax)
    ax.set_title('TFLOPs/s Heatmap (Batch Size × Seq Length)', fontweight='bold')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Batch Size')

save_plot('11_performance_heatmap.png')

# ============================================================
# 圖 12: Efficiency Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
eff_data = df[(df['batch_size'] == 16) & (df['seq_len'] == 512) & (df['causal'] == True)]

metrics = []
for impl in ['Flash1', 'Pytorch']:
    data = eff_data[eff_data['implementation'] == impl]
    if not data.empty:
        metrics.append({
            'impl': impl,
            'fwd': data['forward_tflops'].mean(),
            'bwd': data['backward_tflops'].mean()
        })

if metrics:
    x = np.arange(len(metrics))
    width = 0.35
    
    fwd_vals = [m['fwd'] for m in metrics]
    bwd_vals = [m['bwd'] for m in metrics]
    
    bars1 = ax.bar(x - width/2, fwd_vals, width, label='Forward', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, bwd_vals, width, label='Backward', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    add_value_labels(ax, spacing=3)
    ax.set_xticks(x)
    ax.set_xticklabels([m['impl'] for m in metrics])
    ax.set_ylabel('TFLOPs/s')
    ax.set_title('Forward & Backward Efficiency Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

save_plot('12_efficiency_comparison.png')

# ============================================================
# 統計摘要
# ============================================================
print("\n" + "="*60)
print("統計摘要")
print("="*60)

flash_avg = df[df['implementation'] == 'Flash1']['total_time'].mean()
pytorch_avg = df[df['implementation'] == 'Pytorch']['total_time'].mean()

if not np.isnan(flash_avg) and not np.isnan(pytorch_avg):
    print(f"\nFlash1 vs Pytorch 平均加速比: {pytorch_avg/flash_avg:.2f}x")

print("\n最佳配置 (Top 5 TFLOPs):")
top5 = df.nlargest(5, 'total_tflops')[['batch_size', 'seq_len', 'num_heads', 
                                         'emb_dim', 'implementation', 'total_tflops']]
print(top5.to_string(index=False))
print(f"\n✓ 所有圖表已儲存至 {output_dir}/")
