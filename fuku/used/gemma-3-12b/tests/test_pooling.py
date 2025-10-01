import torch
from utils import get_last_token_indices, pool_last_token


def test_get_last_token_indices_basic():
    am = torch.tensor([
        [1,1,1,0,0],  # last=2
        [1,1,0,0,0],  # last=1
        [1,1,1,1,1],  # last=4
    ])
    idx = get_last_token_indices(am)
    assert idx.tolist() == [2,1,4]


def test_pool_last_token_matches_hidden_state():
    batch, seq_len, hidden = 2, 5, 3
    hs = torch.arange(batch*seq_len*hidden, dtype=torch.float32).reshape(batch, seq_len, hidden)
    am = torch.tensor([
        [1,1,1,0,0],
        [1,1,1,1,0],
    ])
    pooled = pool_last_token(hs, am)
    # 期待される行
    expected0 = hs[0, 2]
    expected1 = hs[1, 3]
    assert torch.allclose(pooled[0], expected0)
    assert torch.allclose(pooled[1], expected1)

