import torch
from models import HierPhi4ForSequenceClassification


class DummyBackbone(torch.nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = type('cfg', (), {'hidden_size': hidden_size})()

    def forward(self, input_ids=None, attention_mask=None):
        B, T = input_ids.shape
        hs = torch.randn(B, T, self.config.hidden_size, dtype=torch.float32)
        return type('Out', (), {'last_hidden_state': hs})()


def test_hier_forward_and_masking():
    # category: [0:true_a (non-mis), 1:false_b (non-mis), 2:misconc (mis)]
    cat_is_mis = torch.tensor([False, False, True], dtype=torch.bool)
    n_cat = 3
    # mc: [0:NA, 1:A, 2:B]
    mc_na = 0
    n_mc = 3

    # joint classes (5):
    # 0: (true_a, NA) valid
    # 1: (false_b, NA) valid
    # 2: (misconc, A) valid
    # 3: (misconc, B) valid
    # 4: (true_a, A) invalid (should be masked)
    joint_to_cat = torch.tensor([0, 1, 2, 2, 0], dtype=torch.long)
    joint_to_mc = torch.tensor([0, 0, 1, 2, 1], dtype=torch.long)

    model = HierPhi4ForSequenceClassification(
        backbone=DummyBackbone(hidden_size=8),
        hidden_size=8,
        n_joint=5,
        n_cat=n_cat,
        n_mc=n_mc,
        joint_to_cat=joint_to_cat,
        joint_to_mc=joint_to_mc,
        cat_is_misconc=cat_is_mis,
        mc_na_index=mc_na,
        lambda_cat=0.1,
        lambda_mc=0.1,
        lambda_constraint=0.1,
    )

    B, T = 2, 5
    input_ids = torch.randint(1, 100, (B, T))
    attn = torch.ones(B, T, dtype=torch.long)
    labels = torch.tensor([0, 3], dtype=torch.long)        # joint
    labels_cat = torch.tensor([0, 2], dtype=torch.long)
    labels_mc = torch.tensor([0, 2], dtype=torch.long)

    out = model(input_ids=input_ids, attention_mask=attn,
                labels=labels, labels_cat=labels_cat, labels_mc=labels_mc)

    assert 'loss' in out and 'logits' in out
    assert out['logits'].shape == (B, 5)
    assert out['logits_cat'].shape == (B, n_cat)
    assert out['logits_mc'].shape == (B, n_mc)

    # 無効 joint（index=4）は強く抑制されているはず
    col4 = out['logits'][:, 4]
    assert torch.all(col4 < -1e6)

    # 逆伝播可能
    out['loss'].backward()

