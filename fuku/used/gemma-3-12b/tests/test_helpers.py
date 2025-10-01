from helpers import detect_classifier_modules_to_save


class LinearLike:
    def __init__(self, out_features):
        self.out_features = out_features


class DummyModel:
    class Cfg:
        num_labels = 5

    def __init__(self):
        self.config = DummyModel.Cfg()
        # 両方持っているケース: 実際にはどちらか一方であることが多い
        self.score = LinearLike(out_features=5)
        self.classifier = LinearLike(out_features=5)


def test_detect_classifier_modules_to_save_prefers_known_names():
    model = DummyModel()
    names = detect_classifier_modules_to_save(model)
    # 少なくともどちらかは検出される
    assert any(n in names for n in ("score", "classifier"))

