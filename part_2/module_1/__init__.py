from abc import ABC, abstractmethod
from typing import Any, Literal, List, Optional, Tuple, Union

import numpy as np
from numpy import ndarray


Selections = Union[
    Tuple[ndarray, ndarray, ndarray, ndarray],
    Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray],
]
StrategyOption = Literal["mean", "median", "constant"]


class DataSplitter:
    """Data splitting interface, which allows you to separate your data
    on train, validation and test selections.
    """

    def __init__(
        self, permute: bool = False, random_seed: Optional[int] = None
    ) -> None:
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


class BasePreprocessor(ABC):
    """The Base Preprocessor class is an abstract base class for preprocessor
    implementations.
    """

    def __init__(self, copy: bool = True) -> None:
        self.copy = copy

    @abstractmethod
    def fit(self, x) -> None:
        """Fit the preprocessor to the provided features."""
        message = "Every preprocessor must implement the `fit()` method."
        raise NotImplementedError(message)

    @abstractmethod
    def transform(self, x) -> Any:
        """Transform the input features."""
        message = "Every preprocessor must implement the `transform()` method."
        raise NotImplementedError(message)

    @staticmethod
    def _get_values_masks(array: ndarray) -> Tuple[bool, bool]:
        non_zero_values_mask = array != 0
        zero_values_mask = ~non_zero_values_mask
        return non_zero_values_mask, zero_values_mask


class ImputingPreprocessor(BasePreprocessor):
    """Imputing Preprocessor class for imputing missing values in x using
    specified strategies.
    """

    def __init__(
        self,
        strategy: StrategyOption = "mean",
        copy: bool = True,
    ) -> None:
        super().__init__(copy)
        self.strategy = strategy
        self.fillers: Any

    def fit(self, x: ndarray, fill_with: Any = None) -> None:
        """Fit the preprocessor on the given x and compute the specified
        statistics for each feature.

        :parameter x: The input features which statistics would be used for
        imputing.
            :type x: :class:`ndarray`
        :parameter fill_with: The object to fill missing values with,
        works when strategy is set to "constant".
            :type fill_with: :class:`Any`

        :returns: :data:`None`
            :rtype: :class:`NoneType`
        """

        match self.strategy:
            case "constant":
                self.fillers = np.full(x.shape[1], fill_with)
            case "mean":
                self.fillers = np.nanmean(x, axis=0)
            case "median":
                self.fillers = np.nanmedian(x, axis=0)
            case _:
                message = (
                    f'Expected strategy to be one of "constant", "mean" '
                    f'or "median", but got "{self.strategy}"'
                )

                raise ValueError(message)

    def transform(self, x: ndarray) -> ndarray:
        """Transform the input features using statistics, calculated with the
        :method:`fit()` method.

        :parameter x: The input x to be imputed.
            :type x: :class:`ndarray`

        :returns: The x imputed.
            :rtype: :class:`ndarray`
        """
        if self.copy:
            x = x.copy()

        nan_mask = np.isnan(x)

        x[nan_mask] = np.take(self.fillers, np.where(nan_mask)[1])
        return x

    def fit_transform(self, x) -> ndarray:
        """Fit and transform at the same time."""

        self.fit(x)

        transformed = self.transform(x)
        return transformed


class MMScalingPreprocessor(BasePreprocessor):
    """Scaling Preprocessor class for scaling the features using MMScaling."""

    def __init__(self, copy: bool = True) -> None:
        super().__init__(copy)
        self.min_values: ndarray
        self.max_values: ndarray

    def fit(self, x: ndarray) -> None:
        """Fit the preprocessor to the input x and computes the (min, max)
        boundaries for each feature.

        :parameter x: The features to fit the preprocessor and compute the
        boundaries.
            :type x: ndarray

        :returns: :data:`None`
            :rtype: :class:`NoneType`
        """
        self.min_values = np.nanmin(x, axis=0)
        self.max_values = np.nanmax(x, axis=0)

    def transform(self, x: ndarray) -> ndarray:
        """Transform the input features and scale the data according to the
        computed boundaries.

        :parameter x: Features to scale and transform.
            :type x: :class:`ndarray`

        :return: Scaled features.
            :rtype: :class:`ndarray`
        """
        if self.copy:
            x = x.copy()

        range_values = self.max_values - self.min_values
        (nonzero_range_mask, zero_range_mask) = self._get_values_masks(
            range_values
        )
        x[:, zero_range_mask] = 0

        x[:, nonzero_range_mask] = (
            x[:, nonzero_range_mask] - self.min_values[nonzero_range_mask]
        ) / np.where(
            range_values[nonzero_range_mask] == 0,
            1,
            range_values[nonzero_range_mask],
        )
        return x

    def fit_transform(self, x) -> ndarray:
        """Fit and transform at the same time."""

        self.fit(x)

        transformed = self.transform(x)
        return transformed


class ZScalingPreprocessor(BasePreprocessor):
    """Z-Scaling Preprocessor class for features standard scaling."""

    def __init__(self, copy: bool = True) -> None:
        super().__init__(copy)
        self.means: ndarray
        self.stds: ndarray

    def fit(self, x: ndarray) -> None:
        """Fit the preprocessor to the input x and computes the mean and
        standard deviation for each feature.

        :parameter x: The features to fit the preprocessor and compute the
        statistics.
            :type x: :class:`ndarray`
        """
        self.means = np.nanmean(x, axis=0)
        self.stds = np.nanstd(x, axis=0)

    def transform(self, x: ndarray) -> ndarray:
        """Transform the input features and standard scale the data according
        to the computed mean and standard deviation.

        :parameter x: Features to scale and transform.
            :type x: :class:`ndarray`

        :return: Standard scaled features.
            :rtype: :class:`ndarray`
        """
        if self.copy:
            x = x.copy()

        (nonzero_std_mask, zero_std_mask) = self._get_values_masks(self.stds)
        (nonzero_mean_mask, _) = self._get_values_masks(self.means)
        x[:, zero_std_mask] = 0

        x[:, nonzero_std_mask] = (
            x[:, nonzero_std_mask] - self.means[nonzero_mean_mask]
        ) / self.stds[nonzero_std_mask]
        return x

    def fit_transform(self, x) -> ndarray:
        """Fit and transform at the same time."""

        self.fit(x)

        transformed = self.transform(x)
        return transformed
