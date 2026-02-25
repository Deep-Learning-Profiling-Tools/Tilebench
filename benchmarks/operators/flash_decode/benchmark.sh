#!/bin/bash

# ================= 配置区域 =================

# 1. 输出目录
OUTPUT_DIR="ncu_reports/stage2_scan"
mkdir -p $OUTPUT_DIR

# 2. Python 脚本路径 (请根据你的实际位置调整)
PYTHON_SCRIPT="/home/zhongchun/repo/Tilebench/benchmarks/operators/flash_decode/impl_cutile.py"

# 3. NVTX Range 名字
# 对应 Python 代码中的: torch.cuda.nvtx.range_push("FlashDecodeStage2_cuTile")
# 这里的 / 是必须的，表示过滤该 Range 下的所有 Kernel
NVTX_TARGET="FlashDecodeStage2_cuTile/"

# ================= 扫描参数定义 =================

# 扫描 Batch Size:
# 小 Batch (1, 4) 用于观察 Latency Bound / Occupancy 不足的问题
# 大 Batch (32, 128) 用于观察 Bandwidth Bound / 满载情况
BATCH_SIZES=(1 4 32 128)

# 扫描 Head Numbers (Query Heads):
# 8:  模拟 GQA 且 Query Head 较少的情况
# 32: 标准 Llama2-7B/Llama3-8B 配置
# 64: Llama3-70B 配置
# 128: Llama3-405B 配置
HEAD_NUMS=(8 32 64 128)

# 固定参数 (控制变量法)
SEQ_LEN=16384   # 长文本场景
HEAD_DIM=128    # 标准 Head Dim
BLOCK_SEQ=128   # Stage 1 切分大小

# ================= Sudo 保活逻辑 (严格复刻) =================
echo "========================================================"
echo "权限请求：请输入 sudo 密码以启动 NCU Profiling..."
sudo -v

# 在后台启动一个循环，每60秒更新一次sudo时间戳，防止超时
# 只要当前脚本进程 ($$) 还在运行，这个后台循环就会一直运行
while true; do
    sudo -n true
    sleep 60
    kill -0 "$$" || exit
done 2>/dev/null &

echo "Sudo 权限已获取，开始长时间扫描任务..."
echo "========================================================"

# ================= 实验循环 =================

# 1. 遍历 Batch Size
for B in "${BATCH_SIZES[@]}"; do
    # 2. 遍历 Head Number
    for H in "${HEAD_NUMS[@]}"; do
        
        # 构造报告文件名
        REPORT_NAME="${OUTPUT_DIR}/Stage2_B${B}_H${H}_Seq${SEQ_LEN}_profile"
        
        echo "[$(date '+%H:%M:%S')] 正在运行: Batch=${B}, Heads=${H}, SeqLen=${SEQ_LEN}..."
        
        # 执行 NCU 命令
        # --nvtx-include: 核心参数，只 Profile 指定 NVTX Range 内的 Kernel
        # --set full: 收集所有指标 (Memory, Compute, Occupancy 等)
        # --target-processes all: 确保捕获所有子进程
        sudo env "PATH=$PATH" ncu \
            --nvtx \
            --nvtx-include "${NVTX_TARGET}" \
            --set full \
            --import-source yes \
            --target-processes all \
            --force-overwrite \
            -o "${REPORT_NAME}" \
            python3 ${PYTHON_SCRIPT} \
            --batch ${B} \
            --heads ${H} \
            --seq-len ${SEQ_LEN} \
            --head-dim ${HEAD_DIM} \
            --block-seq ${BLOCK_SEQ}

        echo "  -> 报告已生成: ${REPORT_NAME}.ncu-rep"
        echo "--------------------------------------------------------"
        
    done
done

echo "所有 Stage 2 扫描任务已完成！结果保存在 ${OUTPUT_DIR}"