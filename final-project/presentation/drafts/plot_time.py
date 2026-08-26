# plot_time.py

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Recorded-data worksheet
# ==========================================
data = {
    # 版本名稱 (對應 execution_times 的欄位)
    "versions": [
        "v0 (Baseline, B=256)",
        "v1 (SM, B=256)",
        "v2 (SoA, B=256)",
        "v3 (SoA, B=64)",
        "v4 (AMD-Ported)",
    ],
    # 測資名稱 (對應 execution_times 的列)
    "testcases": [
        "c1 (N=2048)",
        "c2 (N=4096)",
        "c3 (N=8192)",
        "c4 (N=16384)",
    ],
    # 執行時間矩陣 (Row: Testcase, Col: Version)
    # 單位: 秒
    # 格式: [ [v0_c1, v1_c1, v2_c1, v3_c1], [v0_c2, v1_c2...], ... ]
    "execution_times": [
        [5.96, 6.01, 5.93, 4.05, 4.82],  # c1 results
        [11.76, 11.39, 11.36, 10.96, 8.77],  # c2 draft values
        [39.71, 40.47, 40.27, 39.41, 16.74],  # c3 draft values
        [150.09, 151.81, 152.28, 133.24, 32.93],  # c4 draft values
    ],
    # v0 (Red): Baseline
    # v1 (Orange): Shared Mem
    # v2 (Teal): SoA Basic
    # v3 (Dark Blue): SoA Tuned (NVIDIA)
    # v4 (Purple/Magenta): AMD MI100 port
    "colors": ["#E63946", "#F4A261", "#2A9D8F", "#264653", "#9B5DE5"],
    # 對應 markers
    # v4 uses a star marker to distinguish the AMD port.
    "markers": ["X", "s", "^", "o", "*"],
}

# ==========================================
# 2. 繪圖邏輯
# ==========================================
plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
plt.figure(figsize=(10, 6))

versions = data["versions"]
testcases = data["testcases"]
times_matrix = np.array(data["execution_times"])  # 轉成 numpy array 方便切片

# 針對每個版本畫一條線
for i, ver in enumerate(versions):
    # 取出該版本在所有測資下的時間 (即 column i)
    times = times_matrix[:, i]

    plt.plot(
        testcases,
        times,
        label=ver,
        color=data["colors"][i],
        marker=data["markers"][i],
        linewidth=2.5,
        markersize=9,
        linestyle="-",
    )

    # 在每個點上方標註數值 (可選)
    # for j, val in enumerate(times):
    #     plt.text(j, val * 1.05, f"{val:.1f}s", ha='center', fontsize=9, color=data["colors"][i])

# 設定圖表屬性
plt.title("End-to-End Execution Time Scaling", fontsize=16, fontweight="bold")
plt.xlabel("Testcase (Input Size)", fontsize=14, fontweight="bold")
plt.ylabel("Total Time (s) - Log Scale", fontsize=14, fontweight="bold")

plt.yscale("log")  # 開啟 Log Scale
plt.grid(True, which="both", ls="--", alpha=0.5)

# 優化 Legend
plt.legend(
    title="Version Configuration",
    title_fontsize=12,
    fontsize=11,
    loc="best",
    frameon=True,
    shadow=True,
)

plt.tight_layout()
plt.savefig("execution_time_comparison.png", dpi=300)
print("Plot generated: execution_time_comparison.png")
plt.show()
