import itertools
import multiprocessing as mp

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


def _weighted_row_majority(predictions, weights):
    return sum(predictions * weights) / sum(weights)


def weighted_majority_vote(merged, team_weights, rounded=True):
    merged = merged.copy()
    predictions = merged['prediction']
    weights = merged['confidence'].mul(team_weights).copy()
    rows = zip(predictions.values, weights.values)

    pool = mp.Pool()
    majorities = pool.starmap(_weighted_row_majority, rows, chunksize=1000)

    if rounded:
        majorities = np.round(majorities).astype(bool)

    merged['prediction', 'weighted'] = majorities
    merged = merged.sort_index(axis=1)

    return merged


def naive_majority_vote(predictions):
    merged = predictions.copy()
    merged['prediction', 'naive'] = merged['prediction'].mean(axis=1).round().astype(bool)
    merged = merged.sort_index(axis=1)
    return merged


def impute_confidence(predictions):
    """For each confidence subtract the classifier's mean confidence, impute missing values and
    move the distribution's center to 1.
    """
    imputed = predictions.copy()

    # Convert confidences to mean distances
    imputed['confidence'] = (imputed['confidence'].sub(imputed['confidence'].mean()))

    # Fill missing values
    imputed['confidence'] = imputed['confidence'].fillna(0)

    # Move range to distance of 1 from the mean and move distribution's center to 1.
    imputed['confidence'] = imputed['confidence'].add(1)
    return imputed


def estimate_return_quantity(quantity):
    if quantity == 1 or quantity == 2:
        return 1
    if quantity == 3 or quantity == 4:
        return 2
    if quantity == 5:
        return 3
    return 0
