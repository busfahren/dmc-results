import itertools
import multiprocessing as mp

import numpy as np
import pandas as pd

import load
import evaluate


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
    if test:
        original['returnQuantityMultilabel'] = original.returnQuantity.copy()
        original.returnQuantity = original.returnQuantity.astype(bool)
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


def group_majority_vote(train, merged, weights, rounded=True):
    splits = load.split_weights()
    for split, mask in evaluate._iterate_split_masks(splits, train, merged['original']):
        pass


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
    imputed['confidence'] = (imputed['confidence'].div(imputed['confidence'].mean()))

    # Fill missing values
    imputed['confidence'] = imputed['confidence'].fillna(1)

    # Move range to distance of 1 from the mean and move distribution's center to 1.
    # imputed['confidence'] = imputed['confidence'].add(1)
    return imputed


def estimate_return_quantity(quantity):
    if quantity in [1, 2]:
        return 1
    if quantity in [3, 4]:
        return 2
    if quantity in [5]:
        return 3
    return 0


def finalize(binary_prediction, name):
    class_data = load.orders_class()
    return_quantities = class_data['quantity'].copy()
    class_data = class_data[['orderID', 'articleID', 'colorCode', 'sizeCode']]
    return_quantities.loc[~binary_prediction] = 0
    return_quantities.loc[binary_prediction] = return_quantities[binary_prediction].apply(estimate_return_quantity)
    class_data['prediction'] = return_quantities
    class_data.to_csv('final/' + name + '.csv', index=False, sep=';')


def binary_vector(train, test, columns):
    """Create a mask of test values seen in training data.
    """
    known_mask = test[columns].copy().apply(lambda column: column.isin(train[column.name])).astype(
        int)
    known_mask.columns = ('known_' + c for c in columns)
    return known_mask


def shuffle(df):
    return df.reindex(np.random.permutation(df.index))


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

    # shuffle DF
    M = shuffle(M)

    # Create X and y as matrices to be passed to classifiers
    X = M.drop('returnQuantity', 1).as_matrix()
    y = np.squeeze(M.returnQuantity.as_matrix())
    return X, y


def class_features(train, predictions, categories):
    confs = predictions.confidence.copy()
    confs.columns = ['confA', 'confB', 'confC']
    preds = predictions.prediction.copy().astype(int)
    preds.columns = ['predA', 'predB', 'predC']
    known = binary_vector(train, predictions.original, categories)
    M = pd.concat([confs, preds, known], 1)
    M_dis = M[(M.predA != M.predB) | (M.predA != M.predC)]
    M_agr = M[(M.predA == M.predB) & (M.predA == M.predC)]

    X = M_dis.as_matrix()
    return X, M_dis, M_agr


def precision(y, y_tick):
    return np.sum(y == y_tick) / len(y)


def dmc_cost(y, y_tick):
    return np.sum(np.abs(y - y_tick))
