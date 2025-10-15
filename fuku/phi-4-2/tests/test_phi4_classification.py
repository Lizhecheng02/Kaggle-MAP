"""
修正されたPhi4ForSequenceClassificationクラスのテストコード
"""
import torch
import sys
import os

# 親ディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DummyBackbone(torch.nn.Module):
    """テスト用のダミーバックボーン"""
    def __init__(self, hidden_size=128):
        super().__init__()
        self.config = type('Config', (), {'hidden_size': hidden_size})()
        self.embedding = torch.nn.Embedding(1000, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        # 簡単なforward実装
        embeddings = self.embedding(input_ids)
        return type('Output', (), {'last_hidden_state': embeddings})()


def test_phi4_classification_no_redundant_load():
    """
    Phi4ForSequenceClassificationが不要なモデルロードを行わないことを確認
    """
    print("\n" + "="*60)
    print("Test: Phi4ForSequenceClassification - No Redundant Load")
    print("="*60)

    # train.pyからクラスをインポート
    # （実際のファイルから直接インポートすることで、修正が反映されているか確認）
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train.py")
    )
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)

    Phi4ForSequenceClassification = train_module.Phi4ForSequenceClassification

    # ダミーバックボーンを作成
    dummy_backbone = DummyBackbone(hidden_size=128)
    num_labels = 10

    print("Creating Phi4ForSequenceClassification with dummy backbone...")

    # モデルを作成（backboneを引数として渡す）
    try:
        model = Phi4ForSequenceClassification(
            backbone=dummy_backbone,
            num_labels=num_labels
        )
        print("✓ Model created successfully with backbone argument")
    except TypeError as e:
        print(f"✗ Failed to create model with backbone argument: {e}")
        print("   The __init__ signature may not be updated correctly")
        return False

    # バックボーンが正しく設定されているか確認
    if model.phi is dummy_backbone:
        print("✓ Backbone is correctly assigned (same object reference)")
    else:
        print("✗ Backbone is not the same object (may have been reloaded)")
        return False

    # 分類ヘッドが正しく設定されているか確認
    if model.classifier.out_features == num_labels:
        print(f"✓ Classifier head has correct output size: {num_labels}")
    else:
        print(f"✗ Classifier head output size is incorrect: {model.classifier.out_features} (expected {num_labels})")
        return False

    # hidden_sizeが正しく使われているか確認
    if model.classifier.in_features == dummy_backbone.config.hidden_size:
        print(f"✓ Classifier input size matches backbone hidden size: {dummy_backbone.config.hidden_size}")
    else:
        print(f"✗ Classifier input size mismatch: {model.classifier.in_features} (expected {dummy_backbone.config.hidden_size})")
        return False

    print("="*60 + "\n")
    return True


def test_phi4_classification_forward():
    """
    Phi4ForSequenceClassificationのforward動作を確認
    """
    print("\n" + "="*60)
    print("Test: Phi4ForSequenceClassification - Forward Pass")
    print("="*60)

    # train.pyからクラスをインポート
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train.py")
    )
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)

    Phi4ForSequenceClassification = train_module.Phi4ForSequenceClassification

    # ダミーバックボーンを作成
    dummy_backbone = DummyBackbone(hidden_size=128)
    num_labels = 10

    # モデルを作成
    model = Phi4ForSequenceClassification(
        backbone=dummy_backbone,
        num_labels=num_labels
    )

    # ダミー入力を作成
    batch_size = 4
    seq_length = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, num_labels, (batch_size,))

    print(f"Input shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")

    # Forward pass（ラベルなし）
    try:
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"✓ Forward pass (without labels) successful")
        print(f"  Output logits shape: {output.logits.shape}")

        if output.logits.shape == (batch_size, num_labels):
            print(f"✓ Logits shape is correct: {output.logits.shape}")
        else:
            print(f"✗ Logits shape is incorrect: {output.logits.shape} (expected {(batch_size, num_labels)})")
            return False

        if output.loss is None:
            print(f"✓ Loss is None when labels not provided")
        else:
            print(f"✗ Loss should be None when labels not provided")
            return False

    except Exception as e:
        print(f"✗ Forward pass (without labels) failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Forward pass（ラベルあり）
    try:
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        print(f"✓ Forward pass (with labels) successful")
        print(f"  Loss: {output.loss.item():.4f}")

        if output.loss is not None:
            print(f"✓ Loss is computed when labels provided")
        else:
            print(f"✗ Loss should be computed when labels provided")
            return False

    except Exception as e:
        print(f"✗ Forward pass (with labels) failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("="*60 + "\n")
    return True


def test_phi4_classification_gradient():
    """
    Phi4ForSequenceClassificationで勾配が正しく計算されるか確認
    """
    print("\n" + "="*60)
    print("Test: Phi4ForSequenceClassification - Gradient Computation")
    print("="*60)

    # train.pyからクラスをインポート
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train.py")
    )
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)

    Phi4ForSequenceClassification = train_module.Phi4ForSequenceClassification

    # ダミーバックボーンを作成
    dummy_backbone = DummyBackbone(hidden_size=128)
    num_labels = 10

    # モデルを作成
    model = Phi4ForSequenceClassification(
        backbone=dummy_backbone,
        num_labels=num_labels
    )

    # ダミー入力を作成
    batch_size = 4
    seq_length = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, num_labels, (batch_size,))

    # Forward + Backward
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = output.loss

    print(f"Loss: {loss.item():.4f}")

    # Backpropagation
    loss.backward()

    # 勾配が計算されているか確認
    params_with_grad = 0
    params_without_grad = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                params_with_grad += 1
            else:
                params_without_grad += 1

    print(f"Parameters with gradients: {params_with_grad}")
    print(f"Parameters without gradients: {params_without_grad}")

    if params_without_grad > 0:
        print(f"✗ Some parameters do not have gradients!")
        return False
    else:
        print(f"✓ All trainable parameters have gradients")

    # 分類ヘッドの勾配を確認
    if model.classifier.weight.grad is not None:
        grad_norm = model.classifier.weight.grad.norm().item()
        print(f"✓ Classifier weight gradient norm: {grad_norm:.6f}")
    else:
        print(f"✗ Classifier weight has no gradient!")
        return False

    print("="*60 + "\n")
    return True


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# Phi4ForSequenceClassification Tests")
    print("#"*60 + "\n")

    all_passed = True

    # テスト1: 不要なロードがないことを確認
    if not test_phi4_classification_no_redundant_load():
        all_passed = False

    # テスト2: Forward動作を確認
    if not test_phi4_classification_forward():
        all_passed = False

    # テスト3: 勾配計算を確認
    if not test_phi4_classification_gradient():
        all_passed = False

    print("\n" + "#"*60)
    if all_passed:
        print("# ✓ All tests passed!")
    else:
        print("# ✗ Some tests failed")
    print("#"*60 + "\n")
