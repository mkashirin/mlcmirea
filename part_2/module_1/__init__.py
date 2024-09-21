from typing import List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray

Selections = Union[
    Tuple[ndarray, ndarray, ndarray, ndarray],
    Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray],
]


class DataSplitter:
    """Data splitting interface, which allows you to separate your data
    on train, validation and test selections.
    """

    def __init__(
        self, permute: bool = False, random_seed: Optional[int] = None
    ):
        """Set parameters for the splitting.

        :parameter permute: Defines whether data will be permuted before
        split operation or not.
            :type permute: :class:`bool`
        :parameter random_seed: Random seed that will be applied during
        the process.
            :type random_seed: :class:`Optional[int]`
        """
        self.random_seed = random_seed
        self.permute = permute
        self._selections: List[ndarray]

    def split_data(
        self,
        x: ndarray,
        y: ndarray,
        *,
        test_size: float,
        valid_size: Optional[float] = None,
    ) -> Selections:
        """Split the data on train, validation and test selections.

        :parameter x: Features data, that would be split on train and
        test selections.
            :type x: :class:`ndarray`
        :parameter y: Target data, that would be split on train and
        test selections.
            :type y: :class:`ndarray`
        :parameter test_size: Percentage of data that will be allocated for the
        test selection.
            :type test_size: :class:`float`

        :keyword valid_size: Percentage of data that will be allocated for the
        validation selection.
            :type valid_size: :class:`Optional[float]`

        :returns: Tuple of selections split according to specified parameters.
            :rtype: :class:`Selections`
        """
        if self.random_seed:
            np.random.seed(self.random_seed)
        if self.permute:
            permutation = np.random.permutation(x.shape[0])
            x, y = x[permutation], y[permutation]

        self._set_standard(x, y, test_size)
        if valid_size:
            test_length = self._selections[1].shape[0]
            self._add_valid(test_length, x, y, valid_size)

        selections: Selections = tuple(
            self._selections  # pyright: ignore[reportAssignmentType]
        )
        return selections

    def _set_standard(self, x: ndarray, y: ndarray, test_size: float) -> None:
        train_test_index = int(x.shape[0] * test_size)

        (x_train, x_test), (y_train, y_test) = (
            (x[train_test_index:], x[:train_test_index]),
            (y[train_test_index:], y[:train_test_index]),
        )
        self._selections = [x_train, x_test, y_train, y_test]

    def _add_valid(
        self, test_length: int, x: ndarray, y: ndarray, valid_size: float
    ) -> None:
        test_valid_index = int(test_length * valid_size)

        self._selections[1], self._selections[3] = (
            self._selections[1][test_valid_index:],
            self._selections[3][test_valid_index:],
        )
        self._selections.insert(1, x[:test_valid_index])
        self._selections.insert(4, y[:test_valid_index])
