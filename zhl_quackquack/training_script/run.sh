#!/bin/bash
set -euo pipefail

# 8 张卡一一对应
gpus=(0 1 2 3 4 5 6 7)

# version 固定为 1..8（你也可以手动改）
vers=(52 53 54 55 56 57 58 59)

# 模型名
# models=("Qwen/Qwen3-14B" "Qwen/Qwen3-14B" "Qwen/Qwen3-14B" "Qwen/Qwen3-14B" "Qwen/Qwen3-14B" "Qwen/Qwen3-14B" "Qwen/Qwen3-14B" "Qwen/Qwen3-14B")
models=("microsoft/phi-4" "microsoft/phi-4" "microsoft/phi-4" "microsoft/phi-4" "microsoft/phi-4" "microsoft/phi-4" "microsoft/phi-4" "microsoft/phi-4")

# r 固定
rs=(32 32 32 32 64 64 64 64)

# dropout 固定，但有变化
dropouts=(0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05)

# 新的学习率 sweep
lrs=(1.75e-4 2.0e-4 2.25e-4 2.5e-4 1.75e-4 2.0e-4 2.25e-4 2.5e-4)

# lora_alpha sweep
lora_alphas=(32 32 32 32 32 32 32 32)

# epochs sweep
epochs=(2 2 2 2 2 2 2 2)

# 简单的长度校验
n=${#vers[@]}
if [[ $n -ne 8 || $n -ne ${#gpus[@]} || $n -ne ${#models[@]} || $n -ne ${#rs[@]} \
      || $n -ne ${#dropouts[@]} || $n -ne ${#lrs[@]} || $n -ne ${#lora_alphas[@]} \
      || $n -ne ${#epochs[@]} ]]; then
  echo "数组长度不一致或不是 8，请检查！"
  exit 1
fi

mkdir -p logs configs

for ((i=0; i<8; i++)); do
  gpu=${gpus[$i]}
  ver=${vers[$i]}
  model=${models[$i]}
  r=${rs[$i]}
  dropout=${dropouts[$i]}
  lr=${lrs[$i]}
  lora_alpha=${lora_alphas[$i]}
  epoch=${epochs[$i]}

  short_model=$(basename "$model")
  log_file="logs/ver${ver}_r${r}_${short_model}_lr${lr}_drop${dropout}_alpha${lora_alpha}_ep${epoch}.log"
  cfg_file="configs/ver${ver}_r${r}_${short_model}_lr${lr}_drop${dropout}_alpha${lora_alpha}_ep${epoch}.txt"

  echo "Launching: GPU=$gpu ver=$ver r=$r model=$model lr=$lr dropout=$dropout lora_alpha=$lora_alpha epochs=$epoch"
  echo " -> log: $log_file"
  echo " -> cfg: $cfg_file"

  # 保存本次运行配置
  cat > "$cfg_file" <<EOF
ver: $ver
model_name: $model
max_len: 256
cv_fold: 5
cv_seed: 42
r: $r
lora_alpha: $lora_alpha
lr: $lr
dropout: $dropout
epochs: $epoch
lr_scheduler: linear
gpu: $gpu
EOF

  CUDA_VISIBLE_DEVICES=$gpu python code_less_classes_qid_specific_mloss_neither_na_share_option.py \
    --ver "$ver" \
    --model_name "$model" \
    --max_len 256 \
    --cv_fold 5 \
    --cv_seed 42 \
    --r "$r" \
    --lora_alpha "$lora_alpha" \
    --lr "$lr" \
    --dropout "$dropout" \
    --epochs "$epoch" \
    --lr_scheduler linear \
    > "$log_file" 2>&1 &
done

wait
echo "全部 8 个任务已完成启动；日志在 ./logs，配置在 ./configs"
