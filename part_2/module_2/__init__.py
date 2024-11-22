from itertools import chain
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy import ndarray


IndicesMap = Dict[Any, int]
ConfusionMatrix = Tuple[ndarray, IndicesMap]


def compute_accuracy(actual: ndarray, predicted: ndarray) -> float:
    """Compute accuracy for any model predictions.

    :parameter actual: Actual target values.
        :type actual: :class:`ndarray`
    :parameter predicted: Predicted target values.
        :type predicted: :class:`ndarray`

    :returns: Accuracy score of the model (% of correct predictions).
        :rtype: :class:`float`

    :raises ValueError: If actual and predicted arrays have
    non-broadcasting shapes.
    """
    accuracy = np.sum(predicted == actual) / len(actual)
    return accuracy


def compute_confusion_matrix(
    actual: ndarray,
    predicted: ndarray,
    indices_map: Optional[IndicesMap] = None,
) -> ConfusionMatrix:
    """Compute confusion matrix and get it with indices map for
    classification model predictions.

    :parameter actual: Actual target values.
        :type actual: :class:`ndarray`
    :parameter predicted: Predicted target values.
        :type predicted: :class:`ndarray`

    :keyword indices_map: Dictionary, where keys are features names and
    values are integer indices in the confusion matrix.
        :type indices_map: :class:`IndicesMap`

    :returns: Confusion matrix of the model with indicies map, which describes
    matrix alignment.
        :rtype: :class:`ConfusionMatrix`

    :raises ValueError: If actual and predicted arrays have
    non-broadcasting shapes.
    """

    def _map_to_integers(array, intmap):
        for i, _ in enumerate(array):
            array[i] = intmap[array[i]]
        return array

    actual_list, predicted_list = (
        list(chain.from_iterable(actual.tolist())),
        list(chain.from_iterable(predicted.tolist())),
    )
    concatenated = actual_list + predicted_list
    n_features = len(set(concatenated))
    if indices_map is None:
        indices_map = {
            key: val for key, val in zip(set(concatenated), range(n_features))
        }

    confusion_matrix = np.zeros((n_features, n_features))
    mapped_actual, mapped_predicted = (
        _map_to_integers(actual_list, indices_map),
        _map_to_integers(predicted_list, indices_map),
    )
    for a, p in zip(mapped_actual, mapped_predicted):
        confusion_matrix[a, p] += 1
    confusion_matrix_with_map: ConfusionMatrix = confusion_matrix, indices_map

    return confusion_matrix_with_map


def compute_sensitivities_and_specificities(
    actual: ndarray, predicted: ndarray, as_array: bool = True
) -> Any:
    """Compute sensitivities and specificities for classification model
    predictions.

    :parameter actual: Actual target values.
        :type actual: :class:`ndarray`
    :parameter predicted: Predicted target values.
        :type predicted: :class:`ndarray`

    :keyword as_array: If :data:`True` is passed, function will return
    regular NumPy :class:`ndarray`, where rows correspond sensitivities to
    and specificities of features (those correspond to columns); otherwise
    function will return :class:`dict` which describes upper-mentioned
    alignment explicitly.
        :type as_array: :class:`bool`

    :returns: NumPy :class:`ndarray` or dictionary of sensitivities and
    specificities (depends on :parameter:`as_array`).
        :rtype: :class:`Any`

    :raises ValueError: If actual and predicted arrays have
    non-broadcasting shapes.
    """

    confusion_matrix, indices_map = compute_confusion_matrix(actual, predicted)
    n_features = len(indices_map)

    sensitivities, specificities = (list(), list())
    for i in range(n_features):
        true_positives = confusion_matrix[i, i]
        false_negatives = np.sum(confusion_matrix[:, i])
        sensitivity = true_positives / (true_positives + false_negatives)
        sensitivities.append(sensitivity)

        upper_left = np.sum(confusion_matrix[:i, :i])
        upper_right = np.sum(confusion_matrix[:i, i + 1 :])
        lower_left = np.sum(confusion_matrix[i + 1 :, :i])
        lower_right = np.sum(confusion_matrix[i + 1 :, i + 1 :])
        true_negatives = np.sum(
            (upper_left, upper_right, lower_left, lower_right)
        )
        false_positives = np.sum(confusion_matrix[i])
        specificity = true_negatives / (true_negatives + false_positives)
        specificities.append(specificity)

    if not as_array:
        features_names = list(indices_map.keys())
        keys = ["sensitivities", "specificities"]
        metrics = sensitivities, specificities
        sensitivities_and_specificities = dict.fromkeys(keys)
        for outer_key, metric in zip(keys, metrics):
            sensitivities_and_specificities[outer_key] = {
                key: val for key, val in zip(features_names, metric)
            }
        return sensitivities_and_specificities

    sensitivities_and_specificities = np.array([sensitivities, specificities])
    return sensitivities_and_specificities
