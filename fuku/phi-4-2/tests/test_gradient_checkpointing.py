import torch
from models import HierPhi4ForSequenceClassification


class DummyBackbone(torch.nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = type('cfg', (), {'hidden_size': hidden_size})()
        self.enabled = False
        self.last_kwargs = None

    def forward(self, input_ids=None, attention_mask=None):
        B, T = input_ids.shape
        hs = torch.randn(B, T, self.config.hidden_size, dtype=torch.float32)
        return type('Out', (), {'last_hidden_state': hs})()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.enabled = True
        self.last_kwargs = gradient_checkpointing_kwargs

    def gradient_checkpointing_disable(self):
        self.enabled = False


def build_model():
    # joint_to_cat/mc など最小構成
    n_cat, n_mc, n_joint = 3, 3, 5
    joint_to_cat = torch.tensor([0, 1, 2, 2, 0], dtype=torch.long)
    joint_to_mc = torch.tensor([0, 0, 1, 2, 1], dtype=torch.long)
    cat_is_mis = torch.tensor([False, False, True], dtype=torch.bool)
    mc_na = 0

    return HierPhi4ForSequenceClassification(
        backbone=DummyBackbone(hidden_size=8),
        hidden_size=8,
        n_joint=n_joint,
        n_cat=n_cat,
        n_mc=n_mc,
        joint_to_cat=joint_to_cat,
        joint_to_mc=joint_to_mc,
        cat_is_misconc=cat_is_mis,
        mc_na_index=mc_na,
    )


def test_gradient_checkpointing_enable_disable_delegation():
    model = build_model()
    # enable with kwargs
    kwargs = {"use_reentrant": False}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs)
    assert isinstance(model.backbone, DummyBackbone)
    assert model.backbone.enabled is True
    assert model.backbone.last_kwargs == kwargs

    # disable
    model.gradient_checkpointing_disable()
    assert model.backbone.enabled is False

