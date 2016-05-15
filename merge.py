import os
import itertools

import pandas as pd
import numpy as np


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


def team_names():
    return [name.replace('.csv', '') for name in os.listdir('results') if not name.startswith('.')]


def results(names=None):
    if names is None:
        names = team_names()

    team_dfs = {}
    for name in names:
        df = pd.read_csv('results/' + name + '.csv',
                         delimiter=';')
        team_dfs[name] = df
    return team_dfs


def orders_train():
    original_train = pd.read_csv('data/orders_train.txt', delimiter=';')
    original_train['returnQuantity'] = original_train['returnQuantity'].astype(bool)
    return original_train.set_index(['orderID', 'articleID', 'colorCode', 'sizeCode'],
                                    drop=False)


def orders_class():
    df = pd.read_csv('data/orders_class.txt', delimiter=';')
    return df[['orderID', 'articleID', 'colorCode', 'sizeCode']]


def merge(names=None):
    if names is None:
        names = team_names()

    classifications = results(names)
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

    predictions = [weighted_row_majority(row) for _, row in df.iterrows()]
    if rounded:
        predictions = np.round(predictions).astype(int)
    df['all', 'prediction'] = predictions
    return df


def validate():
    # folder = 'evaluation/' if evaluation else 'result/'

    expected_columns = ['orderID', 'articleID', 'colorCode',
                        'sizeCode', 'confidence', 'prediction']

    results = ['results/' + file for file in os.listdir('results') if not file.startswith('.')]
    tests = ['tests/' + file for file in os.listdir('tests') if not file.startswith('.')]

    for file in results + tests:
        df = pd.read_csv(file, delimiter=';')

        try:
            assert list(df.columns) == expected_columns
            assert ~(df[['orderID', 'articleID', 'colorCode', 'sizeCode', 'prediction']]
                     .isnull().any().any())
            assert df['orderID'].astype(str).str.startswith('a').all()
            assert df['articleID'].astype(str).str.startswith('i').all()
            if file.startswith('results/'):
                assert len(df) == 341098
        except AssertionError:
            print('%s: Assertion failed' % file)
            return df

        print('%s is good' % file)


def validate_submission(path):
    df = pd.read_csv(path, delimiter=';')

    assert (df.columns == ['orderID', 'articleID', 'colorCode', 'sizeCode', 'prediction']).all()

    assert df['orderID'].str.startswith('a').all()
    assert df['orderID'].str.replace('a', '').str.isnumeric().all()

    assert df['articleID'].str.startswith('i').all()
    assert df['articleID'].str.replace('i', '').str.isnumeric().all()

    assert df['colorCode'].dtype == int

    assert df['prediction'].dtype == int
    assert (df['prediction'] <= 5).all()


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
            splits.loc[i, name + '_accuracy'] = (team_df[mask]['returnQuantity']
                                                 == team_df[mask]['prediction']).sum() / mask.sum()

    # Apply multi index
    splits = splits.set_index(split_columns)
    splits = splits.reindex_axis(sorted(splits.columns), axis=1)
    splits.columns = pd.MultiIndex.from_tuples(list(itertools.product(
        sorted(teams),
        ['accuracy', 'size']
    )), names=['team', 'result'])

    # Cast 'size' columns to int
    size_columns = splits.columns.get_loc_level('size', level=1)[0]
    splits.loc[:, size_columns] = splits.loc[:, size_columns].astype(int)

    return splits


def _fix_team_c(path):
    df = pd.read_csv(path, index_col=0, sep=';')
    df = df.drop('quantity', axis=1)
    df['orderID'] = 'a' + df['orderID'].astype(str)
    df['articleID'] = 'i' + df['articleID'].astype(str)
    df.to_csv(path, index=False, sep=';')
