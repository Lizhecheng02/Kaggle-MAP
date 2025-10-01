import types
from types import SimpleNamespace
import torch
import torch.nn as nn


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # LLMForSequenceClassification が参照する config.hidden_size
        self.config = SimpleNamespace(hidden_size=16)
        # フラグ
        self.gc_enabled = False
        self.input_require_grads_enabled = False

    # Transformers のAPI互換の簡易メソッド
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gc_enabled = True

    def gradient_checkpointing_disable(self):
        self.gc_enabled = False

    def enable_input_require_grads(self):
        self.input_require_grads_enabled = True

    # forwardは本テストでは使用しないが、Moduleとしての体裁のため定義
    def forward(self, *args, **kwargs):
        raise NotImplementedError


def test_llm_exposes_gc_methods(monkeypatch):
    import train as train_mod

    # AutoModelForCausalLM.from_pretrained をモックしてダミーバックボーンを返す
    class _DummyAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyBackbone()

    monkeypatch.setattr(train_mod, 'AutoModelForCausalLM', _DummyAuto)

    # インスタンス生成
    m = train_mod.LLMForSequenceClassification(
        model_name='dummy', num_labels=3, attn_implementation=None, torch_dtype=torch.float32
    )

    # メソッドが存在すること
    assert hasattr(m, 'gradient_checkpointing_enable')
    assert hasattr(m, 'gradient_checkpointing_disable')
    assert hasattr(m, 'enable_input_require_grads')

    # 有効化→無効化がバックボーンに反映されること
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert isinstance(m.backbone, DummyBackbone)
    assert m.backbone.gc_enabled is True

    m.gradient_checkpointing_disable()
    assert m.backbone.gc_enabled is False

    # 入力にrequires_gradを付与（委譲）
    m.enable_input_require_grads()
    assert m.backbone.input_require_grads_enabled is True

