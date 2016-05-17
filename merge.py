import itertools

import numpy as np
import pandas as pd

import load


def merged_predictions(teams=None, test=False, keep_columns=None):
    """Return each team's predictions in a merged DataFrame with the following format:

                                                 | confidence | prediction | original     |
                                                 | A | B | .. | A | B | .. | keep_columns |
    | orderID | articleID | sizeCode | colorCode |   |   |    |   |   |    |              |
    """
    # Load all teams if none provided
    if teams is None:
        teams = load.team_names()
    classifications = load.predictions(teams, test)

    # Load the superset of the predictions in the original data
    original = load.orders_train() if test else load.orders_class()
    merge_columns = ['orderID', 'articleID', 'colorCode', 'sizeCode']
    predictions = original[merge_columns]

    # Merge the confidence and prediction of each team into 'predictions'
    for team_name, team_df in classifications.items():
        team_df = team_df.rename(columns={
            'confidence': 'confidence_' + team_name,
            'prediction': 'prediction_' + team_name
        })
        predictions = predictions.merge(team_df, on=merge_columns)

    predictions = predictions.set_index(merge_columns)

    # Sort the columns alphabetically to apply a MultiIndex to the DataFrame
    predictions = predictions.reindex_axis(sorted(predictions.columns), axis=1)
    predictions.columns = pd.MultiIndex.from_tuples(list(itertools.product(
        ['confidence', 'prediction'],
        sorted(classifications.keys())
    )))

    # If keep_columns is provided merge back the original with the columns included
    if keep_columns:
        original = original.set_index(merge_columns, drop=False)[keep_columns]
        idx = pd.MultiIndex.from_tuples(list(itertools.product(['original'], keep_columns)))
        original.columns = idx
        predictions = predictions.merge(original, left_index=True, right_index=True)

    return predictions


def weighted_majority_vote(predictions, team_weights, rounded=False):
    predictions = predictions.copy()

    def weighted_row_majority(row):
        weights = row['confidence'].mul(team_weights)
        return row['prediction'].mul(weights).sum() / weights.sum()

    weighted_majority = predictions.apply(weighted_row_majority, axis=1)

    if rounded:
        weighted_majority = weighted_majority.round().astype(bool)

    return weighted_majority


def naive_majority_vote(predictions):
    merged = predictions.copy()
    merged['prediction', 'naive'] = merged['prediction'].mean(axis=1).round().astype(bool)
    merged = merged.sort_index(axis=1)
    return merged


def impute_confidence(predictions):
    """For each confidence subtract the classifier's mean confidence and then divide by the
    standard deviation. Impute missing values with zeroes (the new mean). Finally move the
    distribution to center around 1 and range from 0 to 2.
    """
    imputed = predictions.copy()

    # Convert confidences to standard deviation distances
    imputed['confidence'] = (imputed['confidence']
                             .sub(imputed['confidence'].mean())
                             .div(imputed['confidence'].std()))

    # Fill missing values
    imputed['confidence'] = imputed['confidence'].fillna(0)

    # Confine range to distance of 1 from the mean and move distribution's center to 1.
    imputed['confidence'] = (imputed['confidence']
                             .div(imputed['confidence'].abs().max())
                             .add(1))
    return imputed


def binary_vector(train, test, columns):
    """Create a mask of test values seen in training data.
    """
    known_mask = test[columns].copy().apply(lambda column: column.isin(train[column.name])).astype(int)
    known_mask.columns = ('known_' + c for c in columns)
    return known_mask


def boosting_features(train, predictions, categories):
    # confidence vectors
    confs = predictions.confidence.copy()
    confs.columns = ['confA', 'confB', 'confC']
    # prediction vectors
    preds = predictions.prediction.copy().astype(int)
    preds.columns = ['predA', 'predB', 'predC']
    # binary vectors for known/unknown categories
    known = binary_vector(train, predictions.original, categories)

    # Merge all feature vectors and only mind rows with disagreement in prediction
    M = pd.concat([confs, preds, known, predictions.original.returnQuantity.astype(int)], 1)
    M = M[(M.predA != M.predB) | (M.predA != M.predC)]

    # Create X and y as matrices to be passed to classifiers
    X = M.drop('returnQuantity', 1).as_matrix()
    y = np.squeeze(M.returnQuantity.as_matrix())
    return X, y

def precision(y, y_tick):
    return np.sum(y == y_tick)/len(y)

