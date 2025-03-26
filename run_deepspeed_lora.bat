@echo off
chcp 65001 > nul
REM Gemma3-LISA DeepSpeed実行コマンド (LoRA設定)

REM Pythonが利用可能か確認
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python が見つかりません。Pythonがインストールされ、PATHに追加されていることを確認してください。
    goto :eof
)

REM PyTorchがインストールされているか確認
python -c "import torch" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo PyTorch がインストールされていません。
    echo 以下のコマンドでインストールしてください:
    echo pip install torch torchvision torchaudio
    goto :eof
)

REM DeepSpeedがインストールされているか確認
python -c "import deepspeed" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo DeepSpeed がインストールされていません。
    echo インストールには以下の順序で実行してください:
    echo 1. pip install torch torchvision torchaudio
    echo 2. pip install deepspeed
    goto :eof
)

set MASTER_PORT=24999
set MODEL_VERSION=google/gemma-3-4b-it
set DATASET_DIR=H:/download/LISA-dataset/dataset
set SAM_PATH=C:/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth
set EXP_NAME=lisa-gemma3-ds-lora

echo.
echo === DeepSpeed LoRA実行を開始します ===
echo モデル: %MODEL_VERSION%
echo 実験名: %EXP_NAME%
echo 設定: LoRAを使用した効率的な学習
echo.

REM コマンド実行 (LoRA設定)
python -m deepspeed --master_port=%MASTER_PORT% train_full_ds.py ^
  --version="%MODEL_VERSION%" ^
  --dataset_dir="%DATASET_DIR%" ^
  --vision_pretrained="%SAM_PATH%" ^
  --dataset="sem_seg||refer_seg||vqa||reason_seg" ^
  --sample_rates="9,3,3,1" ^
  --exp_name="%EXP_NAME%" ^
  --lora_r=8 ^
  --lora_alpha=16 ^
  --lora_dropout=0.05 ^
  --lora_target_modules="q_proj,k_proj,v_proj" ^
  --deepspeed_config="ds_config.json" ^
  --use_mm_start_end ^
  --train_mask_decoder ^
  --gradient_checkpointing ^
  --precision="bf16"

REM 終了メッセージ
echo.
echo LoRA設定での実行が終了しました
echo. 