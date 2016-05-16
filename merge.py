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
        original = original.set_index(merge_columns)[keep_columns]
        idx = pd.MultiIndex.from_tuples(list(itertools.product(['original'], keep_columns)))
        original.columns = idx
        predictions = predictions.merge(original, left_index=True, right_index=True)

    return predictions


def mean_accuracies(merged):
    target = merged['original', 'returnQuantity']
    return merged['prediction'].apply(lambda pred: (pred == target).sum() / len(pred))


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
    raise NotImplementedError
    if teams is None:
        teams = load.team_names()

    splits = load.splits()
    split_columns = splits.columns
    original_train = load.orders_train()

    for name in teams:
        team_df = load.prediction(name, test=True)
        team_df['prediction'] = team_df['prediction'].astype(bool)
        team_df = team_df.set_index(['orderID', 'articleID', 'colorCode', 'sizeCode'], drop=False)
        team_df = team_df[['prediction']].merge(original_train, left_index=True, right_index=True)

        # Orders that were in original training, but not in test set
        # TODO train = train_complement(team_df, train=True)

        for split, mask in iterate_split_masks(train, team_df):
            splits.loc[split, name + '_size'] = mask.sum()
            splits.loc[split, name] = (team_df[mask]['returnQuantity']
                                       == team_df[mask]['prediction']).sum() / mask.sum()

    # Aggregate split sizes
    size_columns = [c for c in splits.columns if c.endswith('_size')]
    splits['size'] = splits[size_columns].mean(axis=1)
    splits['size'] /= splits['size'].sum()
    splits = splits.drop(size_columns, axis=1)

    # Best performing team per split
    splits['best'] = splits[teams].idxmax(axis=1)

    # Naming
    splits = splits.drop(split_columns, axis=1)

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

    diffs.columns.name = 'team'
    diffs.index.name = 'team'

    return diffs


def distinct_split_predictions(test=True):
    raise NotImplementedError
    teams = load.team_names()
    team_predictions = load.predictions(teams, test)
    team_combinations = list(itertools.combinations(teams, 2))

    original = load.orders_train() if test else load.orders_class()
    merge_columns = ['orderID', 'articleID', 'colorCode', 'sizeCode']

    splits = load.splits()
    diffs = pd.DataFrame(0, columns=team_combinations, index=splits.index)
    for t1, t2 in team_combinations:
        df1 = team_predictions[t1].drop('confidence', axis=1)
        df2 = team_predictions[t2].drop('confidence', axis=1)

        combined = df1.merge(df2, on=merge_columns)
        combined = combined.merge(original, left_on=merge_columns, right_index=True)

        # TODO train = train_complement(test=combined, train=original, test=test)

        for split, mask in iterate_split_masks(train, combined):
            split_rows = combined[mask]
            if len(split_rows) == 0:
                continue
            difference = np.round((split_rows['prediction_x'] != split_rows['prediction_y']).sum()
                                  / len(split_rows), 3)
            diffs.loc[split, (t1, t2)] = difference

    diffs.columns = pd.Index(['/'.join(teams) for teams in team_combinations], name='combination')
    return diffs


def iterate_split_masks(train, test):
    splits = load.splits()
    known = pd.DataFrame({col: test[col].isin(train[col]) for col in splits.columns})

    for split, row in list(splits.iterrows()):
        mask = pd.Series(True, index=known.index)
        for col in splits.columns:
            mask &= (known[col] == row[col])

        yield split, mask


def train_complement(train, test):
    return train[~(train.orderID.isin(test.orderID)
                   & train.articleID.isin(test.articleID)
                   & train.colorCode.isin(test.colorCode)
                   & train.sizeCode.isin(test.sizeCode))]


def original_split_sizes():
    train = load.orders_train()
    target = load.orders_class()

    splits = load.splits()

    for split, mask in iterate_split_masks(train, target):
        splits.loc[split, 'size'] = mask.sum()

    splits['size'] /= len(target)
    return splits[['size']]
