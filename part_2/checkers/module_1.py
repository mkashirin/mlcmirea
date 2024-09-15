from typing import Callable

import numpy as np


def check_notebook_1_task_1(splitter: Callable) -> None:
    (x_size, y_size) = ((10, 9), (10, 1))
    (x, y) = (
        np.random.standard_normal(x_size), np.random.standard_normal(y_size)
    )
    data_splitter = splitter(permute=True, random_seed=2024)
    (test_size, valid_size) = (0.3, 0.4)
    expected_test_len = int(test_size * x.shape[0])
    expected_train_len = x.shape[0] - expected_test_len
    expected_valid_len = int(expected_test_len * valid_size)

    splitted = data_splitter.split_data(x, y, test_size=test_size)
    ((x_train, x_test), (y_train, y_test)) = splitted

    assert(x_train != x[expected_test_len:])
    assert(
        x_train.shape[0] == expected_train_len
        and y_train.shape[0] == expected_train_len
    )
    assert(
        x_test.shape[0] == expected_test_len
        and y_test.shape[0] == expected_test_len
    )

    splitted = data_splitter.split_data(
        x, y, test_size=test_size, valid_size=valid_size
    )
    ((x_train, x_test), (y_train, y_test), (x_valid, y_valid)) = splitted
    assert(
        x_valid.shape[0] == expected_valid_len
        and y_valid.shape[0] == expected_valid_len
    )
    assert(x_valid.shape[1] == y_valid.shape[1])