import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class HierPhi4ForSequenceClassification(nn.Module):
    """
    Phi-4 の最終隠れ状態から3ヘッド（Category/Misconception/Joint(合成)）を出す階層モデル。

    - logits_cat:  (B, n_cat)
    - logits_mc:   (B, n_mc)  (NA を含む)
    - logits:      (B, n_joint) ＝ cat/mc からの合成（無効組は大負値でマスク）

    forward 入力：
      input_ids, attention_mask, labels(=joint), labels_cat, labels_mc を受け取り、
      CE(joint) + λ1*CE(cat) + λ2*CE(mc) + λ3*constraint を返す。
    """

    def __init__(
        self,
        backbone,  # transformers.AutoModel 等（last_hidden_state を返す）
        hidden_size: int,
        n_joint: int,
        n_cat: int,
        n_mc: int,
        joint_to_cat: torch.LongTensor,
        joint_to_mc: torch.LongTensor,
        cat_is_misconc: torch.BoolTensor,  # (n_cat,) Category が *_Misconception か
        mc_na_index: int,
        lambda_cat: float = 0.5,
        lambda_mc: float = 0.5,
        lambda_constraint: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.n_joint = n_joint
        self.n_cat = n_cat
        self.n_mc = n_mc
        self.joint_to_cat = joint_to_cat  # (n_joint,)
        self.joint_to_mc = joint_to_mc    # (n_joint,)
        self.cat_is_misconc = cat_is_misconc  # (n_cat,)
        self.mc_na_index = int(mc_na_index)
        self.lambda_cat = lambda_cat
        self.lambda_mc = lambda_mc
        self.lambda_constraint = lambda_constraint

        self.dropout = nn.Dropout(dropout)
        self.fc_cat = nn.Linear(hidden_size, n_cat)
        self.fc_mc = nn.Linear(hidden_size, n_mc)

        self.ce = nn.CrossEntropyLoss()

    def _compose_joint_logits(self, logits_cat: torch.Tensor, logits_mc: torch.Tensor) -> torch.Tensor:
        """cat/mc ロジットから joint ロジットを作る。
        無効（cat が *_Misconception でないのに mc != NA）を強く抑制。
        実装: log p(y_cat) + log p(y_mc) を合成しつつ、無効組には -1e9 を加える。
        """
        # log-softmax で確率空間へ
        logp_cat = torch.log_softmax(logits_cat, dim=-1)  # (B, n_cat)
        logp_mc = torch.log_softmax(logits_mc, dim=-1)    # (B, n_mc)

        B = logp_cat.size(0)
        device = logp_cat.device

        # joint → cat, mc の index を展開
        cat_idx = self.joint_to_cat.to(device)  # (n_joint,)
        mc_idx = self.joint_to_mc.to(device)    # (n_joint,)

        # (B, n_joint) へ gather して和を取る
        jp_cat = logp_cat.index_select(dim=1, index=cat_idx)  # (B, n_joint)
        jp_mc = logp_mc.index_select(dim=1, index=mc_idx)     # (B, n_joint)
        joint = jp_cat + jp_mc

        # 無効組のマスク：cat が非 Misconception かつ mc != NA の joint を抑制
        # cat ごとのフラグを joint の cat_idx に対応づけ
        cat_is_mis = self.cat_is_misconc.to(device)[cat_idx]  # (n_joint,)
        mc_is_na = (mc_idx == self.mc_na_index).to(device)
        invalid = (~cat_is_mis) & (~mc_is_na)                 # (n_joint,)
        if invalid.any():
            # (B, n_joint) にブロードキャスト
            joint = joint.masked_fill(invalid.unsqueeze(0), -1e9)

        return joint

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,        # joint labels
        labels_cat: Optional[torch.Tensor] = None,
        labels_mc: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # 末端トークンの隠れ状態（Phi-4 の既存実装に合わせる）
        pooled = out.last_hidden_state[:, -1, :]
        pooled = self.dropout(pooled)

        logits_cat = self.fc_cat(pooled)   # (B, n_cat)
        logits_mc = self.fc_mc(pooled)     # (B, n_mc)
        logits_joint = self._compose_joint_logits(logits_cat, logits_mc)  # (B, n_joint)

        loss = None
        if (labels is not None):
            loss_joint = self.ce(logits_joint, labels)
            loss = loss_joint

            if labels_cat is not None:
                loss = loss + self.lambda_cat * self.ce(logits_cat, labels_cat)
            if labels_mc is not None:
                loss = loss + self.lambda_mc * self.ce(logits_mc, labels_mc)

            # 期待整合ペナルティ（soft）: E[cat が非 Misconception] × (1 - p_mc(NA))
            if self.lambda_constraint > 0:
                p_cat = torch.softmax(logits_cat, dim=-1)              # (B, n_cat)
                p_mc = torch.softmax(logits_mc, dim=-1)                # (B, n_mc)
                p_cat_not_mis = (p_cat * (~self.cat_is_misconc).to(p_cat.dtype).to(p_cat.device)).sum(dim=-1)
                p_mc_not_na = 1.0 - p_mc[:, self.mc_na_index]
                penalty = (p_cat_not_mis * p_mc_not_na).mean()
                loss = loss + self.lambda_constraint * penalty

        return {
            'loss': loss,
            'logits': logits_joint,   # Trainer/compute_metrics は joint を使う
            'logits_cat': logits_cat,
            'logits_mc': logits_mc,
        }

    # --- Trainer/PEFT 互換のためのユーティリティ ---
    # HF Transformers の Trainer は `model.gradient_checkpointing_enable(...)` を呼び出す前提で実装されています。
    # 本モデルは `nn.Module` を継承しているため、このメソッドをバックボーンに委譲して提供します。
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None):
        """Enable gradient checkpointing on the backbone if available.

        Transformers>=4.41 では `gradient_checkpointing_enable(gradient_checkpointing_kwargs=...)` を
        受け取る実装があり、古い版では引数なしの実装です。両方に対応します。
        """
        backbone = getattr(self, 'backbone', None)
        if backbone is None:
            return
        fn = getattr(backbone, 'gradient_checkpointing_enable', None)
        if fn is None:
            # バックボーンが未対応の場合は何もしない（Trainer 側が存在チェックしないため）
            self._gradient_checkpointing = True  # ダミーフラグ（テストやデバッグ用）
            return
        try:
            if gradient_checkpointing_kwargs is not None:
                return fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            else:
                return fn()
        except TypeError:
            # 古い Transformers で `gradient_checkpointing_kwargs` を受け取らない場合
            return fn()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the backbone if available."""
        backbone = getattr(self, 'backbone', None)
        if backbone is None:
            return
        fn = getattr(backbone, 'gradient_checkpointing_disable', None)
        if fn is None:
            self._gradient_checkpointing = False
            return
        return fn()

    # HF推奨: 勾配チェックポイント時に入力埋め込みに勾配を流す
    def enable_input_require_grads(self):
        """Register a hook so that input embeddings require grads for checkpointing.

        多くのHFモデルで `model.enable_input_require_grads()` が存在しますが、
        ない場合に備えてEmbedding層の出力に`requires_grad_(True)`を付与するフックを登録します。
        """
        backbone = getattr(self, 'backbone', None)
        if backbone is None:
            return
        get_embed = getattr(backbone, 'get_input_embeddings', None)
        if get_embed is None:
            return
        emb = get_embed()
        if emb is None:
            return

        def _make_inputs_require_grad(module, inputs, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)

        # 重複登録回避
        if hasattr(self, '_inputs_require_grads_hook') and self._inputs_require_grads_hook is not None:
            try:
                self._inputs_require_grads_hook.remove()
            except Exception:
                pass
        self._inputs_require_grads_hook = emb.register_forward_hook(_make_inputs_require_grad)

    def disable_input_require_grads(self):
        hook = getattr(self, '_inputs_require_grads_hook', None)
        if hook is not None:
            try:
                hook.remove()
            except Exception:
                pass
            finally:
                self._inputs_require_grads_hook = None

    @property
    def config(self):
        """一部ユーティリティが `model.config` を参照するため、backbone の config を露出。"""
        return getattr(self.backbone, 'config', None)
