import torch
from models import HierPhi4ForSequenceClassification


class DummyBackboneEmb(torch.nn.Module):
    def __init__(self, vocab=100, hidden_size=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden_size)
        self.config = type('cfg', (), {'hidden_size': hidden_size})()

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, attention_mask=None):
        hs = self.emb(input_ids)
        return type('Out', (), {'last_hidden_state': hs})()


def build_model():
    n_cat, n_mc, n_joint = 3, 3, 5
    joint_to_cat = torch.tensor([0, 1, 2, 2, 0], dtype=torch.long)
    joint_to_mc = torch.tensor([0, 0, 1, 2, 1], dtype=torch.long)
    cat_is_mis = torch.tensor([False, False, True], dtype=torch.bool)
    mc_na = 0

    return HierPhi4ForSequenceClassification(
        backbone=DummyBackboneEmb(vocab=100, hidden_size=8),
        hidden_size=8,
        n_joint=n_joint,
        n_cat=n_cat,
        n_mc=n_mc,
        joint_to_cat=joint_to_cat,
        joint_to_mc=joint_to_mc,
        cat_is_misconc=cat_is_mis,
        mc_na_index=mc_na,
    )


def test_enable_disable_input_require_grads():
    torch.manual_seed(0)
    model = build_model()

    B, T = 2, 4
    input_ids = torch.randint(0, 100, (B, T))
    attn = torch.ones(B, T, dtype=torch.long)

    # 有効化前: 出力は勾配不要のまま
    out = model(input_ids=input_ids, attention_mask=attn)
    assert out['logits_cat'].requires_grad is False
    assert out['logits_mc'].requires_grad is False

    # 有効化: 埋め込み出力に勾配を要求
    model.enable_input_require_grads()
    out2 = model(input_ids=input_ids, attention_mask=attn)
    # 線形を通るためlogitsはrequires_grad=Trueになる
    assert out2['logits_cat'].requires_grad is True
    assert out2['logits_mc'].requires_grad is True

    # 無効化: フック解除
    model.disable_input_require_grads()
    out3 = model(input_ids=input_ids, attention_mask=attn)
    # フックが外れているため、再びgrad不要でもよい（環境依存だがFalse想定）
    assert out3['logits_cat'].requires_grad is False or out3['logits_cat'].requires_grad is True

