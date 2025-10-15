"""
モデルロードとGPU設定のテストコード
"""
import torch
import time
import sys
import os

# 親ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_NAME, ATTENTION_IMPLEMENTATION
from transformers import AutoModel, AutoTokenizer


def test_model_load_time():
    """モデルのロード時間を計測するテスト"""
    print("\n" + "="*60)
    print("Test: Model Load Time")
    print("="*60)

    # モデルロード時間の計測
    start_time = time.time()

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=ATTENTION_IMPLEMENTATION
    )

    end_time = time.time()
    load_duration = end_time - start_time

    print(f"✓ Model loaded in {load_duration:.2f} seconds")

    # 適切なロード時間の確認（29GBのモデルなので、最低でも5秒はかかるはず）
    if load_duration < 1.0:
        print(f"⚠ Warning: Load time is suspiciously fast ({load_duration:.2f}s)")
        print("   This may indicate caching or incomplete loading.")
    elif load_duration > 300:
        print(f"⚠ Warning: Load time is very slow ({load_duration:.2f}s)")
    else:
        print(f"✓ Load time is within expected range")

    # モデルのパラメータ数を確認
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")

    # 期待されるパラメータ数（Phi-4は約14B）
    expected_params = 14e9
    if abs(total_params - expected_params) / expected_params > 0.1:
        print(f"⚠ Warning: Parameter count differs significantly from expected {expected_params/1e9:.1f}B")
    else:
        print(f"✓ Parameter count matches expected range")

    print("="*60 + "\n")

    return model


def test_gpu_memory():
    """GPUメモリ使用量を確認するテスト"""
    print("\n" + "="*60)
    print("Test: GPU Memory Usage")
    print("="*60)

    if not torch.cuda.is_available():
        print("⚠ CUDA is not available. Skipping GPU memory test.")
        print("="*60 + "\n")
        return

    # メモリをクリア
    torch.cuda.empty_cache()
    initial_allocated = torch.cuda.memory_allocated(0) / 1024**3
    initial_reserved = torch.cuda.memory_reserved(0) / 1024**3

    print(f"Initial GPU Memory:")
    print(f"  Allocated: {initial_allocated:.2f} GB")
    print(f"  Reserved: {initial_reserved:.2f} GB")

    # モデルをロード
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=ATTENTION_IMPLEMENTATION
    )

    after_load_allocated = torch.cuda.memory_allocated(0) / 1024**3
    after_load_reserved = torch.cuda.memory_reserved(0) / 1024**3

    print(f"\nAfter Loading Model:")
    print(f"  Allocated: {after_load_allocated:.2f} GB (+{after_load_allocated - initial_allocated:.2f} GB)")
    print(f"  Reserved: {after_load_reserved:.2f} GB (+{after_load_reserved - initial_reserved:.2f} GB)")

    # Phi-4 14B モデルの場合、BF16で約28GB必要
    expected_memory = 14  # BF16なので約14GB（29GB / 2）
    actual_memory = after_load_allocated - initial_allocated

    if actual_memory < expected_memory * 0.5:
        print(f"\n⚠ Warning: GPU memory usage ({actual_memory:.2f} GB) is much lower than expected (~{expected_memory:.1f} GB)")
        print("   Model may not be fully loaded to GPU.")
    else:
        print(f"\n✓ GPU memory usage is within expected range")

    # モデルがGPUにあるか確認
    first_param_device = next(model.parameters()).device
    print(f"\nModel device: {first_param_device}")

    if first_param_device.type == 'cuda':
        print("✓ Model is on GPU")
    else:
        print("⚠ Warning: Model is not on GPU!")

    print("="*60 + "\n")

    del model
    torch.cuda.empty_cache()


def test_device_map_auto():
    """device_map="auto" の動作を確認するテスト"""
    print("\n" + "="*60)
    print("Test: device_map='auto' Functionality")
    print("="*60)

    if not torch.cuda.is_available():
        print("⚠ CUDA is not available. Skipping device_map test.")
        print("="*60 + "\n")
        return

    # device_map="auto" でロード
    model_auto = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=ATTENTION_IMPLEMENTATION
    )

    # 各パラメータのデバイスを確認
    devices = set()
    for name, param in model_auto.named_parameters():
        devices.add(str(param.device))

    print(f"Model parameters are distributed across devices: {devices}")

    if 'cuda:0' in devices or 'cuda' in str(list(devices)[0]):
        print("✓ Model is correctly placed on GPU with device_map='auto'")
    else:
        print("⚠ Warning: Model may not be on GPU")

    print("="*60 + "\n")

    del model_auto
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Model Load and GPU Tests")
    print("#"*60 + "\n")

    try:
        # テスト1: モデルロード時間
        test_model_load_time()

        # テスト2: GPUメモリ使用量
        test_gpu_memory()

        # テスト3: device_map="auto" の動作
        test_device_map_auto()

        print("\n" + "#"*60)
        print("# All tests completed")
        print("#"*60 + "\n")

    except Exception as e:
        print(f"\n⚠ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
