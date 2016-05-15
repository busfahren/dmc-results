import os
import itertools

import numpy as np
import pandas as pd

import load


def mean_accuracies(teams=None):
    """Check results present and use the evaluation set to calculate the mean accuracy for each team.
    """
    if teams is None:
        teams = load.team_names()

    original_train = load.orders_train()

    accuracies = pd.Series(np.nan, index=teams)
    for team in teams:
        team_df = load.prediction(team, test=True)
        team_df['prediction'] = team_df['prediction'].astype(bool)
        team_df = team_df.set_index(['orderID', 'articleID', 'colorCode', 'sizeCode'], drop=False)
        team_df = team_df[['prediction']].merge(original_train, left_index=True, right_index=True)
        accuracies[team] = (team_df['prediction'] == team_df['returnQuantity']).sum() / len(team_df)

    return accuracies


def merge_predictions(names=None):
    if names is None:
        names = load.team_names()

    classifications = load.predictions(names, test=False)
    merge_columns = ['orderID', 'articleID', 'colorCode', 'sizeCode']

    df = load.orders_class()
    for team_name, team_df in classifications.items():
        team_df = team_df.rename(columns={
            'confidence': team_name + '_confidence',
            'prediction': team_name + '_prediction'
        })
        df = df.merge(team_df, on=merge_columns)

    df = df.set_index(['orderID', 'articleID', 'colorCode', 'sizeCode'])
    df = df.reindex_axis(sorted(df.columns), axis=1)
    df.columns = pd.MultiIndex.from_tuples(list(itertools.product(
        sorted(classifications.keys()),
        ['confidence', 'prediction']
    )), names=['team', 'result'])
    return df


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


def evaluate_split_performance(teams=None):
    if teams is None:
        teams = load.team_names()

    splits = load.splits()

    original_train = load.orders_train()

    for name in teams:
        team_df = load.prediction(name, test=True)
        team_df['prediction'] = team_df['prediction'].astype(bool)
        team_df = team_df.set_index(['orderID', 'articleID', 'colorCode', 'sizeCode'], drop=False)
        team_df = team_df[['prediction']].merge(original_train, left_index=True, right_index=True)

        # Orders that were in original training, but not in test set
        train = load.train_complement(team_df, test=True)

        def extend_splits(i, mask):
            splits.loc[i, name + '_size'] = mask.sum()
            splits.loc[i, name] = (team_df[mask]['returnQuantity']
                                   == team_df[mask]['prediction']).sum() / mask.sum()

        iterate_split_masks(train, team_df, extend_splits)

    # Aggregate split sizes
    size_columns = [c for c in splits.columns if c.endswith('_size')]
    splits['size'] = splits[size_columns].mean(axis=1).astype(int)
    splits = splits.drop(size_columns, axis=1)

    # Best performing team per split
    splits['best'] = splits[teams].idxmax(axis=1)

    return splits


def distinct_predictions(test=True):
    teams = load.team_names()
    team_predictions = load.predictions(teams, test)

    diffs = pd.DataFrame(0, columns=teams, index=teams)
    for t1, t2 in itertools.combinations(teams, 2):
        df1 = team_predictions[t1].drop('confidence', axis=1)
        df2 = team_predictions[t2].drop('confidence', axis=1)
        combined = df1.merge(df2, on=['orderID', 'articleID', 'colorCode', 'sizeCode'])
        different = np.round((combined['prediction_x'] != combined['prediction_y']).sum()
                             / len(combined), 3)
        diffs.loc[t1, t2] = different
        diffs.loc[t2, t1] = different

    return diffs


def distinct_split_predictions(team1, team2):
    pass


def iterate_split_masks(train, test, func):
    splits = load.splits()
    split_columns = splits.columns

    known = pd.DataFrame({col: test[col].isin(train[col]) for col in split_columns})

    for i, row in list(splits.iterrows()):
        mask = pd.Series(True, index=known.index)
        for col in split_columns:
            mask &= (known[col] == row[col])

        func(i, mask)
