@echo off
chcp 65001 > nul
REM Gemma3-LISA 学習環境セットアップスクリプト

echo ===================================
echo Gemma3-LISA 学習環境セットアップ
echo ===================================
echo.

REM Pythonが利用可能か確認
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python が見つかりません。Pythonをインストールしてください。
    echo https://www.python.org/downloads/
    goto :eof
)

echo Pythonバージョン:
python --version
echo.

REM 必要なライブラリをインストール
echo 必要なライブラリをインストールします...
echo.

REM PyTorchのインストール（CUDA対応版）
echo ステップ1: PyTorch + CUDA のインストール
echo 注意: GPUを使用する場合はCUDA対応版をインストールしてください
echo.
echo 以下のオプションから選択してください:
echo 1. CUDA 12.1 対応版 PyTorch
echo 2. CUDA 11.8 対応版 PyTorch
echo 3. CPU版のみ（GPU非対応）
echo 4. スキップ（既にインストール済み）
set /p torch_option="選択 (1-4): "

if "%torch_option%"=="1" (
    echo CUDA 12.1 対応版 PyTorch をインストールします...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else if "%torch_option%"=="2" (
    echo CUDA 11.8 対応版 PyTorch をインストールします...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "%torch_option%"=="3" (
    echo CPU版 PyTorch をインストールします...
    pip install torch torchvision torchaudio
) else if "%torch_option%"=="4" (
    echo PyTorchのインストールをスキップします。
) else (
    echo 無効な選択です。
    goto :eof
)

echo.
echo ステップ2: DeepSpeed のインストール
echo 1. インストールする
echo 2. スキップ（既にインストール済み）
set /p ds_option="選択 (1-2): "

if "%ds_option%"=="1" (
    echo DeepSpeed をインストールします...
    pip install deepspeed
) else if "%ds_option%"=="2" (
    echo DeepSpeedのインストールをスキップします。
) else (
    echo 無効な選択です。
    goto :eof
)

echo.
echo ステップ3: その他の必要なライブラリのインストール
echo 1. インストールする
echo 2. スキップ
set /p other_option="選択 (1-2): "

if "%other_option%"=="1" (
    echo その他の必要なライブラリをインストールします...
    pip install transformers peft bitsandbytes accelerate
) else if "%other_option%"=="2" (
    echo その他のライブラリのインストールをスキップします。
) else (
    echo 無効な選択です。
    goto :eof
)

echo.
echo ステップ4: 環境情報を確認
echo 1. 確認する
echo 2. スキップ
set /p check_option="選択 (1-2): "

if "%check_option%"=="1" (
    echo 環境情報を確認します...
    python check_torch_cuda.py
) else if "%check_option%"=="2" (
    echo 環境情報の確認をスキップします。
) else (
    echo 無効な選択です。
    goto :eof
)

echo.
echo ===================================
echo セットアップが完了しました。
echo 以下のコマンドでDeepSpeed学習を開始できます:
echo.
echo デバッグモード: .\run_deepspeed_debug.bat
echo 通常学習: .\run_deepspeed.bat
echo LoRA学習: .\run_deepspeed_lora.bat
echo =================================== 