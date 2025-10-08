import pandas as pd
from datasets import Dataset

from utils import (
    format_input,
    tokenize_dataset,
    PROMPT_TASK_HEADER,
    PROMPT_FIELD_QUESTION,
    PROMPT_FIELD_ANSWER,
    PROMPT_FIELD_CORRECT,
    PROMPT_FIELD_EXPLANATION,
    PROMPT_BOOL_LABELS,
)


class StubTokenizer:
    def __init__(self, fail_apply=False):
        self.fail_apply = fail_apply
        self.applied_messages = []
        self.received_texts = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # 記録だけしてテンプレート文字列を返す/失敗を投げる
        self.applied_messages.append(messages)
        if self.fail_apply:
            raise ValueError("no chat template")
        content = messages[-1]["content"]
        return f"<templated>{content}</templated>"

    def __call__(self, texts, padding=False, truncation=True, max_length=512, return_tensors=None):
        # 呼び出されたテキストを記録し、ダミーのトークン列を返す
        if isinstance(texts, str):
            texts = [texts]
        self.received_texts.extend(texts)
        # ダミーのトークナイズ結果（長さに応じて可変）
        input_ids = [[1] * min(len(t), 5) for t in texts]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_format_input_no_chat_tokens():
    row = pd.Series({
        "QuestionText": "What is 2+2?",
        "MC_Answer": "3",
        "is_correct": 0,
        "StudentExplanation": "Because 2 and 2 are consecutive numbers.",
    })

    text = format_input(row)
    assert "<|im_start|>" not in text
    assert PROMPT_TASK_HEADER in text
    assert f"{PROMPT_FIELD_QUESTION}:" in text
    assert f"{PROMPT_FIELD_ANSWER}:" in text
    assert f"{PROMPT_FIELD_CORRECT}: {PROMPT_BOOL_LABELS[0]}" in text  # No
    assert f"{PROMPT_FIELD_EXPLANATION}:" in text


def test_tokenize_dataset_uses_chat_template_when_available():
    df = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]})
    ds = Dataset.from_pandas(df)
    tok = StubTokenizer(fail_apply=False)

    out = tokenize_dataset(ds, tok, max_len=32)
    # apply_chat_template が2サンプル分呼ばれている
    assert len(tok.applied_messages) == 2
    # __call__ に渡った文字列がテンプレート済みであること
    assert all(s.startswith("<templated>") for s in tok.received_texts)
    # set_format の影響により columns がトーチ用になっていること
    for key in ("input_ids", "attention_mask"):
        assert key in out.features


def test_tokenize_dataset_fallback_when_template_missing():
    df = pd.DataFrame({"text": ["foo"], "label": [0]})
    ds = Dataset.from_pandas(df)
    tok = StubTokenizer(fail_apply=True)

    out = tokenize_dataset(ds, tok, max_len=32)
    # フォールバックでは apply_chat_template 例外後に __call__ が呼ばれる
    assert len(tok.received_texts) == 1
    # ChatML風のフォールバックで始まることを確認
    assert tok.received_texts[0].startswith("<|im_start|>user")
    for key in ("input_ids", "attention_mask"):
        assert key in out.features
