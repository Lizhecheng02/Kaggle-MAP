# Qwen2.5-VL-3B プロンプト最適化戦略

## 現状分析
- 現在のMAP@3スコア: 0.9416
- 詳細なChatML形式のプロンプトを使用
- 全ての誤概念をリストとして含めている

## o3からの推奨事項に基づく改善案

### 1. 誤概念リストの番号付け
現在は誤概念を改行区切りのテキストとして提供しているが、番号付きリストに変更し、モデルが番号のみを返すようにする。これにより：
- 出力の一貫性が向上
- 後処理が簡単になる
- 曖昧な一致を避けられる

### 2. 厳格な出力フォーマット指示
システムプロンプトで明確に「番号のみをカンマ区切りで返す」ことを指示。Qwen2.5-VLは厳格な出力制約を良く守る。

### 3. Few-shot学習の導入（オプション）
- 1つの良い例を追加（+1-2pp MAPの改善が期待できる）
- 動的few-shot（最も類似した訓練例を埋め込み検索で取得）が最も効果的
- 複数の例は逆効果になる可能性がある

### 4. 温度とtop-pの調整
- 現在の設定を確認し、T=0.3、top-p=0.95に調整
- MAP@3タスクでは適度な多様性が必要

### 5. TASKマーカーの使用
明確な「TASK:」マーカーを使用して、指示部分を強調

## 提案する新しいプロンプト形式

```python
def format_input_improved(row):
    """改善されたプロンプトフォーマット"""
    if row['is_correct']:
        status = "Yes"
    else:
        status = "No"
    
    # 誤概念を番号付きリストとして準備
    misconceptions = [
        'Adding_across', 'Adding_terms', 'Additive', 'Base_rate', 'Certainty',
        'Definition', 'Denominator-only_change', 'Division', 'Duplication',
        'Firstterm', 'FlipChange', 'Ignores_zeroes', 'Incomplete',
        'Incorrect_equivalent_fraction_addition', 'Interior', 'Inverse_operation',
        'Inversion', 'Irrelevant', 'Longer_is_bigger', 'Mult', 'Multiplying_by_4',
        'NA', 'Not_variable', 'Positive', 'Scale', 'Shorter_is_bigger',
        'Subtraction', 'SwapDividend', 'Tacking', 'Unknowable', 'WNB',
        'Whole_numbers_larger', 'Wrong_Fraction', 'Wrong_Operation',
        'Wrong_fraction', 'Wrong_term'
    ]
    
    # 番号付きリストを作成
    numbered_misconceptions = []
    for i, misc in enumerate(misconceptions, 1):
        numbered_misconceptions.append(f"{i}. {misc}")
    miscs_text = " | ".join(numbered_misconceptions)  # 1行にパック
    
    prompt = (
        f"<|im_start|>system\n"
        f"You are an expert math-education researcher.\n"
        f"Identify up to three *numbered* misconceptions from the allowed list that best match the student's explanation.\n"
        f"Respond with the numbers only, separated by commas, no other text.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Question: {row['QuestionText']}\n"
        f"Student answer (correct? {status}): {row['MC_Answer']}\n"
        f"Student explanation:\n"
        f"{row['StudentExplanation']}\n\n"
        f"Allowed misconceptions:\n"
        f"{miscs_text}\n\n"
        f"TASK: Return the IDs of the TOP 3 misconceptions (most probable first).\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt
```

## Few-shot例の追加（オプション）

```python
def add_few_shot_example(prompt):
    """Few-shot例を追加"""
    example = """
<|im_start|>user
Question: What is 1/2 + 1/3?
Student answer (correct? No): 2/5
Student explanation:
I added the numerators (1+1=2) and the denominators (2+3=5) to get 2/5.

Allowed misconceptions:
[同じ番号付きリスト]

TASK: Return the IDs of the TOP 3 misconceptions (most probable first).
<|im_end|>
<|im_start|>assistant
14,1,2
<|im_end|>
"""
    # プロンプトの適切な位置に例を挿入
    return prompt.replace("<|im_start|>user\n", example + "<|im_start|>user\n")
```

## 推論時の設定変更

```python
# config.pyに追加
TEMPERATURE = 0.3
TOP_P = 0.95

# 推論スクリプトで使用
generation_config = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_new_tokens": 20,  # 番号のみなので短く
}
```

## 後処理の改善

```python
def parse_model_output(output):
    """モデル出力から誤概念IDを抽出"""
    # カンマ区切りの番号を抽出
    numbers = output.strip().split(',')
    misconception_ids = []
    
    for num in numbers[:3]:  # 最大3つ
        try:
            idx = int(num.strip()) - 1  # 0ベースのインデックスに変換
            if 0 <= idx < len(misconceptions):
                misconception_ids.append(idx)
        except:
            continue
    
    return misconception_ids
```

## 実装の優先順位

1. **高優先度**（すぐに実装）
   - 誤概念の番号付け
   - 厳格な出力フォーマット指示
   - 温度設定の調整

2. **中優先度**（テスト後に実装）
   - Few-shot例の追加
   - 後処理の改善

3. **低優先度**（オプション）
   - 動的few-shot選択
   - Logit bias（推論スタックがサポートする場合）

## 期待される改善
- MAP@3スコア: 0.9416 → 0.95+（+1-2pp）
- より一貫した出力フォーマット
- 後処理エラーの削減