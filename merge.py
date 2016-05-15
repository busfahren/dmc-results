import os
import itertools

import numpy as np

from loading import *


def mean_accuracies(teams=None):
    """Check results present and use the evaluation set to calculate the mean accuracy for each team.
    """
    if teams is None:
        teams = team_names()

    original_train = orders_train()

    accuracies = pd.Series(np.nan, index=teams)
    for team in teams:
        team_df = pd.read_csv('tests/' + team + '.csv', delimiter=';')
        team_df['prediction'] = team_df['prediction'].astype(bool)
        team_df = team_df.set_index(['orderID', 'articleID', 'colorCode', 'sizeCode'], drop=False)
        team_df = team_df[['prediction']].merge(original_train, left_index=True, right_index=True)
        accuracies[team] = (team_df['prediction'] == team_df['returnQuantity']).sum() / len(team_df)

    return accuracies


def merge(names=None):
    if names is None:
        names = team_names()

    classifications = predictions(names, test=False)
    merge_columns = ['orderID', 'articleID', 'colorCode', 'sizeCode']

    df = orders_class()
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


def evaluate(teams=None):
    if teams is None:
        teams = team_names()

    split_columns = ['articleID', 'productGroup', 'customerID', 'voucherID']
    splits = pd.read_csv('data/splits.csv', delimiter=';')
    splits.loc[:, split_columns] = splits.loc[:, split_columns].astype(bool)

    original_train = orders_train()

    for name in teams:
        team_df = pd.read_csv('tests/' + name + '.csv', delimiter=';')
        team_df['prediction'] = team_df['prediction'].astype(bool)
        team_df = team_df.set_index(['orderID', 'articleID', 'colorCode', 'sizeCode'], drop=False)
        team_df = team_df[['prediction']].merge(original_train, left_index=True, right_index=True)

        # Orders that were in original training, but not in test set
        train = original_train[~(original_train.orderID.isin(team_df.orderID)
                                 & original_train.articleID.isin(team_df.articleID)
                                 & original_train.colorCode.isin(team_df.colorCode)
                                 & original_train.sizeCode.isin(team_df.sizeCode))]

        known = pd.DataFrame({col: team_df[col].isin(train[col]) for col in split_columns})

        for i, row in list(splits.iterrows()):
            mask = pd.Series(True, index=known.index)
            for col in split_columns:
                mask &= (known[col] == row[col])

            splits.loc[i, name + '_size'] = mask.sum()
            splits.loc[i, name] = (team_df[mask]['returnQuantity']
                                   == team_df[mask]['prediction']).sum() / mask.sum()

    # Aggregate split sizes
    size_columns = [c for c in splits.columns if c.endswith('_size')]
    splits['size'] = splits[size_columns].mean(axis=1).astype(int)
    splits = splits.drop(size_columns, axis=1)

    # Best performing team per split
    splits['best'] = splits[teams].idxmax(axis=1)

    return splits


def differences(test=True):
    teams = team_names()
    team_predictions = predictions(teams, test)

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
