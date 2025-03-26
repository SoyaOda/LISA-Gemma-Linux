import torch
import sys

print('='*40)
print('PyTorch環境情報')
print('='*40)
print(f'PyTorchバージョン: {torch.__version__}')
print(f'CUDAが利用可能: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDAバージョン: {torch.version.cuda}')
    print(f'利用可能なGPU数: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('警告: CUDAが利用できません。GPUでのDeepSpeed学習ができない可能性があります。')
    print('NVIDIA GPUがある場合は、CUDAとcuDNNをインストールしてください。')

print('\n推奨インストール手順:')
print('1. 適切なCUDAバージョンに対応したPyTorchをインストール:')
print('   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
print('   (CUDA 11.8の例。他のバージョンは https://pytorch.org/get-started/locally/ を参照)')
print('2. DeepSpeedをインストール:')
print('   pip install deepspeed')
print('='*40)

# 情報をファイルに保存
with open('torch_cuda_info.txt', 'w', encoding='utf-8') as f:
    f.write('='*40 + '\n')
    f.write('PyTorch環境情報\n')
    f.write('='*40 + '\n')
    f.write(f'PyTorchバージョン: {torch.__version__}\n')
    f.write(f'CUDAが利用可能: {torch.cuda.is_available()}\n')
    
    if torch.cuda.is_available():
        f.write(f'CUDAバージョン: {torch.version.cuda}\n')
        f.write(f'利用可能なGPU数: {torch.cuda.device_count()}\n')
        for i in range(torch.cuda.device_count()):
            f.write(f'GPU {i}: {torch.cuda.get_device_name(i)}\n')
    else:
        f.write('警告: CUDAが利用できません。GPUでのDeepSpeed学習ができない可能性があります。\n')
        f.write('NVIDIA GPUがある場合は、CUDAとcuDNNをインストールしてください。\n')
    
    f.write('\n推奨インストール手順:\n')
    f.write('1. 適切なCUDAバージョンに対応したPyTorchをインストール:\n')
    f.write('   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n')
    f.write('   (CUDA 11.8の例。他のバージョンは https://pytorch.org/get-started/locally/ を参照)\n')
    f.write('2. DeepSpeedをインストール:\n')
    f.write('   pip install deepspeed\n')
    f.write('='*40 + '\n')

print(f'情報は torch_cuda_info.txt に保存されました。') 