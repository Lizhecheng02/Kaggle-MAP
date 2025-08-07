#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_scheduler.py

学習率スケジューラーの動作確認スクリプト
"""

import numpy as np
import matplotlib.pyplot as plt

def test_scheduler(scheduler_type, initial_lr, min_lr, warmup_rounds, total_rounds):
    """スケジューラーの動作をテスト"""
    learning_rates = []
    
    for i in range(total_rounds):
        if i < warmup_rounds:
            # ウォームアップフェーズ
            lr = min_lr + (initial_lr - min_lr) * (i / warmup_rounds)
        else:
            # メインフェーズ
            progress = (i - warmup_rounds) / (total_rounds - warmup_rounds)
            
            if scheduler_type == 'cosine':
                lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))
            elif scheduler_type == 'linear':
                lr = initial_lr - (initial_lr - min_lr) * progress
            elif scheduler_type == 'exponential':
                lr = initial_lr * (min_lr / initial_lr) ** progress
            else:
                lr = initial_lr
        
        learning_rates.append(lr)
    
    return learning_rates

# パラメータ設定
scheduler_config = {
    'initial_learning_rate': 0.05,
    'min_learning_rate': 0.001,
    'warmup_rounds': 50,
    'total_rounds': 1500
}

# 各スケジューラータイプをテスト
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

scheduler_types = ['cosine', 'linear', 'exponential', 'constant']

for idx, scheduler_type in enumerate(scheduler_types):
    lrs = test_scheduler(
        scheduler_type,
        scheduler_config['initial_learning_rate'],
        scheduler_config['min_learning_rate'],
        scheduler_config['warmup_rounds'],
        scheduler_config['total_rounds']
    )
    
    ax = axes[idx]
    ax.plot(lrs, linewidth=2)
    ax.set_title(f'{scheduler_type.capitalize()} Scheduler', fontsize=12)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)
    
    # ウォームアップ領域を強調表示
    ax.axvline(x=scheduler_config['warmup_rounds'], color='red', linestyle='--', alpha=0.5, label='Warmup End')
    ax.legend()

plt.tight_layout()
plt.savefig('learning_rate_schedulers.png', dpi=150)
print("Learning rate scheduler visualization saved as 'learning_rate_schedulers.png'")

# フォールドごとの学習率減衰も表示
print("\nFold-wise learning rate decay:")
initial_lr = scheduler_config['initial_learning_rate']
min_lr = scheduler_config['min_learning_rate']
decay_rate = 0.95

for fold in range(5):
    current_lr = max(initial_lr * (decay_rate ** fold), min_lr)
    print(f"Fold {fold + 1}: {current_lr:.4f}")