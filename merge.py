import itertools

import numpy as np
import pandas as pd

import load


def merged_predictions(teams=None, test=False, keep_columns=None):
    if teams is None:
        teams = load.team_names()

    classifications = load.predictions(teams, test)
    merge_columns = ['orderID', 'articleID', 'colorCode', 'sizeCode']

    original = load.orders_train() if test else load.orders_class()
    predictions = original[merge_columns]

    for team_name, team_df in classifications.items():
        team_df = team_df.rename(columns={
            'confidence': 'confidence_' + team_name,
            'prediction': 'prediction_' + team_name
        })
        predictions = predictions.merge(team_df, on=merge_columns)

    predictions = predictions.set_index(merge_columns)
    predictions = predictions.reindex_axis(sorted(predictions.columns), axis=1)
    predictions.columns = pd.MultiIndex.from_tuples(list(itertools.product(
        ['confidence', 'prediction'],
        sorted(classifications.keys())
    )))

    if keep_columns:
        original = original.set_index(merge_columns, drop=False)[keep_columns]
        idx = pd.MultiIndex.from_tuples(list(itertools.product(['original'], keep_columns)))
        original.columns = idx
        predictions = predictions.merge(original, left_index=True, right_index=True)

    return predictions


def majority_vote(df, accuracies, rounded=False):
    names = df.columns.get_level_values('team').unique()
    mean_confidences = pd.Series([df[n]['confidence'].mean() for n in names],
                                 index=names)
    # The ratio of each classification's accuracy to the mean accuracy
    mean_accuracy_ratios = accuracies / accuracies.mean()

    def weighted_row_majority(prediction_row):
        summed_weighted_votes = 0
        summed_weights = 0
        for n in names:
            confidence, prediction = prediction_row[n]
            # Ratio of the confidence in the row to the classification's mean confidence
            confidence_ratio = confidence / mean_confidences[n]
            weight = confidence_ratio * np.square(mean_accuracy_ratios[n])
            if prediction:
                summed_weighted_votes += weight
            summed_weights += weight
        return summed_weighted_votes / summed_weights

    final_predictions = [weighted_row_majority(row) for _, row in df.iterrows()]
    if rounded:
        final_predictions = np.round(final_predictions).astype(int)
    df['all', 'prediction'] = final_predictions
    return df


def impute_confidences(merged):
    return merged[merged.isnull().any(axis=1)]