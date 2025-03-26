@echo off
chcp 65001 > nul
REM DeepSpeed JITインストールスクリプト（修正版）

echo ===================================
echo DeepSpeed インストール（修正版）
echo ===================================
echo.

REM 環境変数設定
set DS_BUILD_OPS=0
set DS_BUILD_AIO=0
set DS_BUILD_SPARSE_ATTN=0
set DS_BUILD_EVOFORMER_ATTN=0
set DS_SKIP_CUDA_CHECK=1
set TORCH_EXTENSIONS_DIR=./torch-extensions

echo 環境変数を設定しました:
echo DS_BUILD_OPS=%DS_BUILD_OPS% (オペレーションのビルドを無効化)
echo DS_BUILD_AIO=%DS_BUILD_AIO% (AIOオペレーションを無効化)
echo DS_BUILD_SPARSE_ATTN=%DS_BUILD_SPARSE_ATTN% (Sparse Attentionを無効化)
echo DS_BUILD_EVOFORMER_ATTN=%DS_BUILD_EVOFORMER_ATTN% (Evoformer Attentionを無効化)
echo DS_SKIP_CUDA_CHECK=%DS_SKIP_CUDA_CHECK% (CUDAバージョン確認をスキップ)
echo TORCH_EXTENSIONS_DIR=%TORCH_EXTENSIONS_DIR% (拡張機能のキャッシュディレクトリ)
echo.

REM PyTorchが利用可能か確認
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo エラー: PyTorchのインポートに失敗しました。
    echo PyTorchのインストールを確認してください: pip install torch
    goto :eof
)

echo PyTorchが正常に検出されました。
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
echo.

echo インストール方法の選択:
echo 1. 基本インストール (DeepSpeed 0.8.3)
echo 2. 依存関係を無視してインストール
echo 3. ソースからのJITモードインストール
set /p install_option="選択 (1-3): "

if "%install_option%"=="1" (
    echo DeepSpeed 0.8.3をインストールします...
    pip install deepspeed==0.8.3 --no-deps
) else if "%install_option%"=="2" (
    echo 依存関係を無視してDeepSpeedをインストールします...
    pip install --no-deps --no-build-isolation deepspeed
) else if "%install_option%"=="3" (
    echo DeepSpeedをJITモードでインストールします...
    pip install deepspeed --no-cache-dir
) else (
    echo 無効な選択です。
    goto :eof
)

echo.
echo インストール完了後、以下のコマンドでDeepSpeedの状態を確認できます:
echo python -m deepspeed.env_report
echo.
echo =================================== 