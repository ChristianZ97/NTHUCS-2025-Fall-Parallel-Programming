import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ==========================================
# 1. 數據與設定
# ==========================================
data = {
    "versions": [
        "v0\n(Baseline)",
        "v1\n(SM, B=256)",
        "v2\n(SoA, B=256)",
        "v3\n(SoA, B=64)",
    ],
    # 優化後的顏色定義 (對應 v0 -> v3)
    # v0 (Red): 警示/基準
    # v1 (Orange): 過渡/改進中
    # v2 (Teal): 顯著優化
    # v3 (Dark Blue): 最終/最佳版本
    "colors": ["#E63946", "#F4A261", "#2A9D8F", "#264653"],
    "memory": {
        "gld_eff": [
            25.00,
            25.00,
            99.85,
            99.85,
        ],  # gld_efficiency Global Memory Load Efficiency
        "gst_eff": [
            99.85,
            100.00,
            99.85,
            99.85,
        ],  # gst_efficiency Global Memory Store Efficiency
        "shared_store_tx": [
            0.0,
            1.433628,
            1.204969,
            1.163399,
        ],  # shared_store_transactions_per_request Shared Memory Store Transactions Per Request
    },
    "compute": {
        "sm_eff": [42.78, 42.39, 42.98, 99.47],  # sm_efficiency Multiprocessor Activity
        "occupancy": [
            11.6879,
            12.4999,
            12.4998,
            5.1413,
        ],  # achieved_occupancy Achieved Occupancy
    },
    "stalls": {
        "exec": [
            84.45,
            92.38,
            91.77,
            95.20,
        ],  # stall_exec_dependency Issue Stall Reasons (Execution Dependency)
        "memory": [
            13.02,
            0.18,
            0.05,
            0.27,
        ],  # stall_memory_dependency Issue Stall Reasons (Data Request)
        "sync": [
            0.0,
            5.63,
            6.07,
            1.69,
        ],  # stall_sync Issue Stall Reasons (Synchronization)
    },
    "time": {"elapsed": [5.96, 6.01, 5.93, 4.05]},
}

stall_colors = {
    "exec": "#457B9D",
    "memory": "#E63946",
    "sync": "#E9C46A",
}

plt.rcParams.update({"font.size": 12, "figure.dpi": 300})
versions = data["versions"]
colors = data["colors"]
x = np.arange(len(versions))


# 通用的標籤添加函數
def add_labels(
    ax, rects, format_str="{:.1f}", rotation=0, color="black", fontsize=9, y_offset=2
):
    for rect in rects:
        height = rect.get_height()
        if height > 0:  # 只標註非 0 的數值
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height + y_offset,
                format_str.format(height),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=color,
                rotation=rotation,
                fontweight="bold",
            )


# ==========================================
# 圖表 1: Memory Efficiency (已加標籤)
# ==========================================
fig1, ax1 = plt.subplots(figsize=(10, 7))  # 稍微加大以容納標籤
width = 0.25

# Bar 1 & 2
rects1 = ax1.bar(
    x - width,
    data["memory"]["gld_eff"],
    width,
    label="GLD Efficiency",
    color=colors,
    edgecolor="white",
    alpha=0.9,
)
rects2 = ax1.bar(
    x,
    data["memory"]["gst_eff"],
    width,
    label="GST Efficiency",
    color=colors,
    edgecolor="white",
    alpha=0.5,
    hatch="//",
)

# 加入標籤
add_labels(ax1, rects1, format_str="{:.1f}%", fontsize=9, y_offset=1)
add_labels(
    ax1, rects2, format_str="{:.1f}%", fontsize=9, y_offset=1
)  # 錯開高度避免重疊

ax1.set_ylabel("Efficiency (%)", fontweight="bold")
ax1.set_title("Memory Subsystem Efficiency", fontweight="bold", pad=15)
ax1.set_ylim(0, 125)  # 增加 Y 軸上限給標籤空間
ax1.grid(axis="y", linestyle="--", alpha=0.3)
ax1.set_xticks(x)
ax1.set_xticklabels(versions)

# 右軸
ax2 = ax1.twinx()
rects3 = ax2.bar(
    x + width,
    data["memory"]["shared_store_tx"],
    width,
    label="Shared Store TX",
    color="gray",
    alpha=0.2,
    edgecolor="black",
)
ax2.set_ylabel("Shared Store TX (Ideal=1.0)", fontweight="bold", color="#555555")
ax2.set_ylim(0, 3.5)
ax2.axhline(y=1.0, color="#E63946", linestyle="--", linewidth=2)

# 右軸標籤 (使用不同顏色區分)
for rect in rects3:
    height = rect.get_height()
    if height > 0:
        ax2.text(
            rect.get_x() + rect.get_width() / 2.0,
            height + 0.05,
            "{:.2f}".format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
            fontweight="bold",
        )

# Legend 移到底部
legend_elements = [
    Patch(facecolor="gray", edgecolor="white", alpha=0.9, label="GLD Eff (%)"),
    Patch(
        facecolor="gray", edgecolor="white", alpha=0.5, hatch="//", label="GST Eff (%)"
    ),
    Patch(facecolor="gray", edgecolor="black", alpha=0.2, label="Shared Store TX"),
]
ax1.legend(
    handles=legend_elements,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    frameon=False,
)

plt.tight_layout()
plt.savefig("1_memory_efficiency_labeled.png")
print("Saved: 1_memory_efficiency_labeled.png")
plt.close()


# ==========================================
# 圖表 2: Compute Utilization (已加標籤)
# ==========================================
fig2, ax3 = plt.subplots(figsize=(10, 7))
width = 0.35

rects_sm = ax3.bar(
    x - width / 2,
    data["compute"]["sm_eff"],
    width,
    label="SM Efficiency",
    color=colors,
    edgecolor="white",
    alpha=0.9,
)
rects_occ = ax3.bar(
    x + width / 2,
    data["compute"]["occupancy"],
    width,
    label="Achieved Occupancy",
    color=colors,
    edgecolor="white",
    alpha=0.4,
    hatch="..",
)

# 加入標籤
add_labels(ax3, rects_sm, format_str="{:.1f}%", fontsize=10)
add_labels(ax3, rects_occ, format_str="{:.1f}%", fontsize=10)

ax3.set_ylabel("Percentage (%)", fontweight="bold")
ax3.set_title("Compute Utilization", fontweight="bold", pad=15)
ax3.set_ylim(0, 115)
ax3.grid(axis="y", linestyle="--", alpha=0.3)
ax3.set_xticks(x)
ax3.set_xticklabels(versions)

legend_elements_2 = [
    Patch(facecolor="gray", edgecolor="white", alpha=0.9, label="SM Efficiency"),
    Patch(
        facecolor="gray",
        edgecolor="white",
        alpha=0.4,
        hatch="..",
        label="Achieved Occupancy",
    ),
]
ax3.legend(
    handles=legend_elements_2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=False,
)

plt.tight_layout()
plt.savefig("2_compute_utilization_labeled.png")
print("Saved: 2_compute_utilization_labeled.png")
plt.close()

# (圖3 和 圖4 保持不變，照舊執行即可)
# ==========================================
# 圖表 3: Stall Analysis (Stacked)
# ==========================================
fig3, ax4 = plt.subplots(figsize=(8, 6))
stalls = data["stalls"]
bottom = np.zeros(len(versions))
p1 = ax4.bar(
    versions,
    stalls["exec"],
    label="Execution Dep.",
    color=stall_colors["exec"],
    edgecolor="white",
    width=0.6,
)
bottom += stalls["exec"]
p2 = ax4.bar(
    versions,
    stalls["memory"],
    bottom=bottom,
    label="Memory Dep.",
    color=stall_colors["memory"],
    edgecolor="white",
    width=0.6,
)
bottom += stalls["memory"]
p3 = ax4.bar(
    versions,
    stalls["sync"],
    bottom=bottom,
    label="Sync",
    color=stall_colors["sync"],
    edgecolor="white",
    width=0.6,
)

ax4.set_ylabel("Stall Percentage (%)", fontweight="bold")
ax4.set_title("Instruction Stall Analysis", fontweight="bold", pad=15)
ax4.set_ylim(0, 100)
ax4.grid(axis="y", linestyle="--", alpha=0.3)
ax4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
for i, v in enumerate(stalls["exec"]):
    ax4.text(
        i,
        v / 2,
        f"{v}%",
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
        fontsize=10,
    )
plt.tight_layout()
plt.savefig("3_stall_analysis.png")
print("Saved: 3_stall_analysis.png")
plt.close()

# ==========================================
# 圖表 4: Execution Time
# ==========================================
fig4, ax5 = plt.subplots(figsize=(8, 6))
bars = ax5.bar(
    versions,
    data["time"]["elapsed"],
    color=colors,
    edgecolor="black",
    width=0.6,
    alpha=0.9,
)
ax5.set_ylabel("Elapsed Time (seconds)", fontweight="bold")
ax5.set_title("End-to-End Performance", fontweight="bold", pad=15)
ax5.grid(axis="y", linestyle="--", alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax5.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.05,
        f"{height:.2f}s",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )
baseline = data["time"]["elapsed"][0]
best = min(data["time"]["elapsed"])
speedup = baseline / best
ax5.text(
    0.95,
    0.95,
    f"Max Speedup\n{speedup:.1f}x",
    transform=ax5.transAxes,
    fontsize=14,
    fontweight="bold",
    color="white",
    ha="right",
    va="top",
    bbox=dict(
        boxstyle="round,pad=0.5", facecolor="#264653", alpha=0.9, edgecolor="none"
    ),
)
plt.tight_layout()
plt.savefig("4_execution_time.png")
print("Saved: 4_execution_time.png")
plt.close()
