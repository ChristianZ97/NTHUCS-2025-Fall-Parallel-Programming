import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns  # 引入 seaborn 讓圖表更漂亮

# 設定圖片儲存目錄
RESULT_DIR = "osu_results"
MULTI_NODE_DIR = "osu_results_multi"
OUTPUT_IMG_DIR = "plots"

if not os.path.exists(OUTPUT_IMG_DIR):
    os.makedirs(OUTPUT_IMG_DIR)

# 設定繪圖風格 (使用 seaborn)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

def find_column(df, keyword):
    """輔助函數：模糊搜尋欄位名稱"""
    for col in df.columns:
        if keyword in col:
            return col
    return None

def plot_data(df, title, filename, y_label, is_log_y=False):
    plt.figure(figsize=(12, 7))
    
    # 動態尋找欄位 (ud_verbs vs shm)
    col_ud = find_column(df, "_ud_verbs")
    col_shm = find_column(df, "_shm")
    
    has_data = False
    
    # 畫線：Single-node UD Verbs
    if col_ud:
        plt.plot(df['Size'], df[col_ud], marker='o', linestyle='-', color='#e74c3c', label='UD Verbs (Single-node)')
        has_data = True
        
    # 畫線：Single-node Shared Memory
    if col_shm:
        plt.plot(df['Size'], df[col_shm], marker='^', linestyle='-', color='#3498db', label='Shared Memory (Intra-node)')
        has_data = True

    # 特殊處理：如果是 Latency 圖，嘗試疊加 Multi-node 數據
    if "latency" in filename.lower() and "multi" not in filename.lower():
        multi_csv = os.path.join(MULTI_NODE_DIR, "multi_node_latency.csv")
        if os.path.exists(multi_csv):
            try:
                df_multi = pd.read_csv(multi_csv)
                col_multi = find_column(df_multi, "MultiNode")
                if col_multi:
                    # 修正 linestyle 衝突：只用 color 和 linestyle 參數，不使用 fmt 字串
                    plt.plot(df_multi['Size'], df_multi[col_multi], marker='x', linestyle='--', color='#2ecc71', label='UD Verbs (Multi-node)')
                    has_data = True
            except Exception as e:
                print(f"  Warning: Could not load multi-node data: {e}")

    if not has_data:
        print(f"  Skipping {filename}: No matching columns found.")
        plt.close()
        return

    plt.title(title, pad=20, fontweight='bold')
    plt.xlabel('Message Size (Bytes)')
    plt.ylabel(y_label)
    plt.xscale('log', base=2)
    
    if is_log_y:
        plt.yscale('log')
        
    # 優化 Grid
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls=":", alpha=0.4)
    
    plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc='best')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_IMG_DIR, filename)
    plt.savefig(save_path, dpi=300) # 提高解析度
    print(f"Saved: {save_path}")
    plt.close()

def main():
    csv_files = glob.glob(os.path.join(RESULT_DIR, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {RESULT_DIR}")
        return

    for csv_file in csv_files:
        try:
            filename = os.path.basename(csv_file)
            df = pd.read_csv(csv_file)
            
            # 判斷圖表類型
            if "latency" in filename.lower():
                title = "Latency Comparison"
                y_label = "Latency (us)"
                is_log = True
                
                if "put" in filename.lower():
                    title = "RMA Put Latency Comparison"
                elif "get" in filename.lower():
                    title = "RMA Get Latency Comparison"
                elif "pt2pt" in filename.lower():
                    title = "Point-to-Point Latency Comparison"
                
                plot_data(df, title, filename.replace(".csv", ".png"), y_label, is_log_y=is_log)
                
            elif "bandwidth" in filename.lower():
                title = "Bandwidth Comparison"
                y_label = "Bandwidth (MB/s)"
                is_log = False # Bandwidth 通常不用 Log Y
                
                if "bibandwidth" in filename.lower():
                    title = "Bidirectional Bandwidth Comparison"
                elif "put" in filename.lower():
                    title = "RMA Put Bandwidth Comparison"
                
                plot_data(df, title, filename.replace(".csv", ".png"), y_label, is_log_y=is_log)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    main()
