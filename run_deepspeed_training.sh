#!/bin/bash
# Gemma3-LISA DeepSpeed学習実行スクリプト
# このスクリプトはDeepSpeedを使用してGemma3-LISAモデルを学習します

# 実行方法:
# bash run_deepspeed_training.sh

# DeepSpeedが利用可能か確認
if ! python -c "import deepspeed" &> /dev/null; then
    echo "エラー: DeepSpeedがインストールされていません"
    echo "インストールしてください: pip install deepspeed"
    exit 1
fi

# デバイス情報を表示
echo "=== GPU情報 ==="
nvidia-smi
echo "================"

# 学習ハイパーパラメータ
MASTER_PORT=24999
BATCH_SIZE=2
GRAD_ACCUM=8
EPOCHS=3
STEPS_PER_EPOCH=1000
LEARNING_RATE=0.0003
DATASET="sem_seg,refer_seg,vqa,reason_seg"
SAMPLE_RATES="9,3,3,1"
EXP_NAME="lisa-gemma3-deepspeed"
VISION_MODEL="C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth"
DATASET_DIR="H:/download/LISA-dataset/dataset"
GEMMA_MODEL="google/gemma-3-4b-it"

# 学習コマンドのログを記録
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="deepspeed_training_${TIMESTAMP}.log"

echo "ログファイル: ${LOG_FILE}"
echo "実験名: ${EXP_NAME}"
echo "バッチサイズ: ${BATCH_SIZE}, 勾配蓄積: ${GRAD_ACCUM}"
echo "エポック数: ${EPOCHS}, ステップ数/エポック: ${STEPS_PER_EPOCH}"
echo "学習率: ${LEARNING_RATE}"
echo "データセット: ${DATASET}"
echo "サンプルレート: ${SAMPLE_RATES}"
echo "開始時刻: $(date)"
echo ""

# DeepSpeedで学習を実行
deepspeed --master_port=${MASTER_PORT} train_full_ds.py \
  --version="${GEMMA_MODEL}" \
  --dataset_dir="${DATASET_DIR}" \
  --vision_pretrained="${VISION_MODEL}" \
  --dataset="${DATASET}" \
  --sample_rates="${SAMPLE_RATES}" \
  --batch_size=${BATCH_SIZE} \
  --grad_accumulation_steps=${GRAD_ACCUM} \
  --epochs=${EPOCHS} \
  --steps_per_epoch=${STEPS_PER_EPOCH} \
  --lr=${LEARNING_RATE} \
  --exp_name="${EXP_NAME}" \
  --deepspeed_config="ds_config.json" \
  --use_mm_start_end \
  --train_mask_decoder \
  --gradient_checkpointing \
  --precision="bf16" 2>&1 | tee ${LOG_FILE}

echo ""
echo "終了時刻: $(date)"
echo "ログファイル: ${LOG_FILE}" 