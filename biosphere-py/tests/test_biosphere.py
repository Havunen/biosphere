from pathlib import Path

import numpy as np
import pytest

from biosphere import DecisionTree, RandomForest

_IRIS_FILE = "iris.csv"
_IRIS_PATH = Path(__file__).resolve().parents[2] / "testdata" / _IRIS_FILE


def test_forest():
    data = np.loadtxt(_IRIS_PATH, skiprows=1, delimiter=",", usecols=(0, 1, 2, 3, 4))
    X = data[:, 0:4]
    y = data[:, 4]

    random_forest = RandomForest()
    oob_predictions = random_forest.fit_predict_oob(X, y)
    predictions = random_forest.predict(X)

    oob_mse = np.mean((oob_predictions - y) ** 2)
    mse = np.mean((predictions - y) ** 2)

    assert oob_mse < 0.05
    assert mse < oob_mse / 2


def test_tree():
    data = np.loadtxt(_IRIS_PATH, skiprows=1, delimiter=",", usecols=(0, 1, 2, 3, 4))
    X = data[:, 0:4]
    y = data[:, 4]

    decision_tree = DecisionTree()
    decision_tree.fit(X, y)
    predictions = decision_tree.predict(X)

    mse = np.mean((predictions - y) ** 2)

    assert mse < 0.05


# TODO: Better test checking that supplying parameters had correct effect.
@pytest.mark.parametrize("max_features", [0.2, 3, "sqrt", None])
def test_max_features(max_features):
    _ = RandomForest(max_features=max_features)
    _ = DecisionTree(max_features=max_features)


def test_tree_predicts_different_classes_after_fit():
    X = np.arange(90, dtype=float).reshape(-1, 1)
    y = np.zeros(90, dtype=int)
    y[30:60] = 1
    y[60:] = 2

    tree = DecisionTree(max_depth=None, random_state=0)
    tree.fit(X, y.astype(float))
    predictions = tree.predict(X)

    predicted_classes = np.rint(predictions).astype(int)
    assert np.array_equal(predicted_classes, y)


def test_forest_predicts_different_classes_after_fit():
    X = np.arange(90, dtype=float).reshape(-1, 1)
    y = np.zeros(90, dtype=int)
    y[30:60] = 1
    y[60:] = 2

    forest = RandomForest(
        n_estimators=30,
        max_depth=2,
        max_features=None,
        random_state=0,
        n_jobs=1,
    )
    forest.fit(X, y.astype(float))
    predictions = forest.predict(X)

    predicted_classes = np.rint(predictions).astype(int)
    assert set(predicted_classes.tolist()) >= {0, 1, 2}
    assert np.mean(predicted_classes == y) >= 0.95
