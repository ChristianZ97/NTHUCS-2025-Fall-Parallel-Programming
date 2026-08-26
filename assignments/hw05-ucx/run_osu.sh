#!/bin/bash

# ==========================================
# 設定區
# ==========================================
OSU_DIR="$HOME/hw5/UCX-lsalab/test/mpi/osu"
OUT_DIR="$HOME/hw5/osu_results"
mkdir -p $OUT_DIR
MPI_CMD="mpiucx -n 2"

# UCX TLS 設定
TLS_SHM="all"
TLS_UD="ud_verbs"

# ==========================================
# 核心函數
# ==========================================
run_test_and_extract() {
    local tls_name=$1
    local tls_config=$2
    local exe_path=$3
    local temp_out=$4
    
    local log_file="${temp_out%.tmp}.log"

    echo "  >> Running $(basename $exe_path) with $tls_name..."
    # echo "     (Log: $log_file)"
    
    $MPI_CMD -x UCX_TLS=$tls_config $exe_path 2>&1 | tee $log_file | \
    grep -v "UCX" | grep -v "^#" | grep -v "^\[" | \
    awk '{if(NF==2 && $1 ~ /^[0-9]+$/) print $1 "," $2}' > $temp_out

    if [ ! -s $temp_out ]; then
        echo "     [WARNING] No data generated for $exe_path"
    fi
}

# 輔助函數：執行一組比較並產生 CSV
run_comparison() {
    local test_name=$1       # e.g., "Pt2Pt_Latency"
    local exe_subpath=$2     # e.g., "pt2pt/osu_latency"
    local col_prefix=$3      # e.g., "Latency"
    
    local exe="$OSU_DIR/$exe_subpath"
    local csv_final="$OUT_DIR/${test_name}.csv"
    local tmp_ud="$OUT_DIR/${test_name}_ud.tmp"
    local tmp_shm="$OUT_DIR/${test_name}_shm.tmp"

    echo "=== Testing $test_name ==="
    
    # Run Experiments
    run_test_and_extract "ud_verbs" "$TLS_UD" "$exe" "$tmp_ud"
    run_test_and_extract "shm"      "$TLS_SHM" "$exe" "$tmp_shm"

    # Merge CSV
    echo "Size,${col_prefix}_ud_verbs,${col_prefix}_shm" > $csv_final
    join -t, -1 1 -2 1 $tmp_ud $tmp_shm >> $csv_final
    echo "   [Saved] $csv_final"
    echo ""
}

# ==========================================
# 1. Point-to-Point Tests
# ==========================================
run_comparison "pt2pt_latency" "pt2pt/osu_latency" "Latency"
run_comparison "pt2pt_bandwidth" "pt2pt/osu_bw" "Bandwidth"
run_comparison "pt2pt_bibandwidth" "pt2pt/osu_bibw" "BiBandwidth" # 新增

# ==========================================
# 2. One-sided Tests
# ==========================================
run_comparison "rma_put_latency" "one-sided/osu_put_latency" "PutLatency"
run_comparison "rma_get_latency" "one-sided/osu_get_latency" "GetLatency" # 新增
run_comparison "rma_put_bandwidth" "one-sided/osu_put_bw" "PutBandwidth" # 新增

# ==========================================
# 清理與完成
# ==========================================
rm $OUT_DIR/*.tmp
echo "=== All Extended Experiments Completed ==="
echo "Results are in: $OUT_DIR"
ls -1 $OUT_DIR/*.csv
