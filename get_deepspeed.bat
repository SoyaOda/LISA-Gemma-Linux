@echo off
chcp 65001 > nul
REM DeepSpeed JITインストールスクリプト

echo ===================================
echo DeepSpeed JITモードインストール
echo ===================================
echo.

REM 環境変数設定
set DS_BUILD_OPS=0
set DS_SKIP_CUDA_CHECK=1
set TORCH_EXTENSIONS_DIR=./torch-extensions

echo 環境変数を設定しました:
echo DS_BUILD_OPS=%DS_BUILD_OPS% (オペレーションのビルドを無効化)
echo DS_SKIP_CUDA_CHECK=%DS_SKIP_CUDA_CHECK% (CUDAバージョン確認をスキップ)
echo TORCH_EXTENSIONS_DIR=%TORCH_EXTENSIONS_DIR% (拡張機能のキャッシュディレクトリ)
echo.

echo DeepSpeedをJITモードでインストールします...
pip install deepspeed

echo.
echo インストール完了後、以下のコマンドでDeepSpeedの状態を確認できます:
echo python -m deepspeed.env_report
echo.
echo =================================== 