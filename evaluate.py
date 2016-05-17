import itertools

import numpy as np
import pandas as pd

import load

merge_columns = ['orderID', 'articleID', 'colorCode', 'sizeCode']


def mean_accuracies(predictions):
    """Take merged predictions and calculate mean accuracy for each team.
    """
    target = predictions['original', 'returnQuantity']
    means = (predictions['prediction']
             .apply(lambda team: (team == target).sum() / len(target)))
    return pd.DataFrame(means, columns=['accuracy'])


def evaluate_split_performance(train, test):
    target = test['original', 'returnQuantity']
    predictions = test['prediction']
    splits = load.splits()

    team_performances = pd.DataFrame(index=splits.index, columns=predictions.columns)
    for split, mask in _iterate_split_masks(splits, train, test['original']):
        split_size = mask.sum()
        split_predictions = predictions[mask]
        split_target = target[mask]
        split_performances = (split_predictions
                              .apply(lambda col: (col == split_target).sum() / split_size))

        team_performances.loc[split] = split_performances

    team_performances.columns.name = 'team'
    return team_performances.astype(float)


def distinct_predictions(predictions):
    predictions = predictions['prediction']
    teams = predictions.columns

    diffs = pd.DataFrame(0, columns=teams, index=teams)
    for t1, t2 in itertools.combinations(teams, 2):
        diff = (predictions[t1] != predictions[t2]).sum()
        diffs.loc[t1, t2] = diff
        diffs.loc[t2, t1] = diff

    diffs /= len(predictions)
    diffs.columns.name = 'team'
    diffs.index.name = 'team'
    return diffs


def distinct_split_predictions(train, test):
    predictions = test['prediction']
    teams = predictions.columns
    team_combinations = list(itertools.combinations(teams, 2))
    splits = load.splits()

    differences = pd.DataFrame(index=splits.index, columns=team_combinations)
    for split, mask in _iterate_split_masks(splits, train, test['original']):
        split_size = mask.sum()
        split_predictions = predictions[mask]
        for t1, t2 in team_combinations:
            diff = ((split_predictions[t1] != split_predictions[t2]).sum() / split_size)
            differences.loc[split, (t1, t2)] = diff

    differences.columns = [' '.join(c) for c in differences.columns]
    return differences.astype(float)


def split_mean_confidences(train, test):
    confidences = test['confidence']
    splits = load.splits()

    means = pd.DataFrame(index=splits.index, columns=confidences.columns)
    for split, mask in _iterate_split_masks(splits, train, test['original']):
        split_size = mask.sum()
        split_confidences = confidences[mask]
        means.loc[split] = split_confidences.mean()

    return means.astype(float)


def split_sizes(train, test):
    splits = load.splits()
    sizes = pd.DataFrame(columns=['size'], index=splits.index)
    for split, mask in _iterate_split_masks(splits, train, test):
        sizes.loc[split, 'size'] = mask.sum()

    sizes['size'] /= len(test)
    return sizes


def _iterate_split_masks(splits, train, test):
    known = pd.DataFrame({col: test[col].isin(train[col]) for col in splits.columns})

    for split, row in list(splits.iterrows()):
        mask = pd.Series(True, index=known.index)
        for col in splits.columns:
            mask &= (known[col] == row[col])

        yield split, mask


def test_complement(test):
    original_train = load.orders_train()
    test = test.merge(original_train, left_index=True,
                      right_on=merge_columns)
    return original_train[~(original_train['orderID'].isin(test['orderID'])
                            & original_train['articleID'].isin(test['articleID'])
                            & original_train['colorCode'].isin(test['colorCode'])
                            & original_train['sizeCode'].isin(test['sizeCode']))]
