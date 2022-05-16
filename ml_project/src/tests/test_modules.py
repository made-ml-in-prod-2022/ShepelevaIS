import numpy as np

from features import CategoricalEncoder


def test_categorical_encoder_simple():
    data = np.array([['a', 'b', 'd'], ['d', 'f', 'g'], ['a', 'b', 'd']])
    encoded_data = CategoricalEncoder().fit_transform(data)
    expected_data = np.array([[0., 0., 0.], [1., 1., 1.], [0., 0., 0.]])
    assert np.allclose(expected_data, encoded_data)
