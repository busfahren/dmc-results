import os

import pandas as pd


def team_names():
    return [name.replace('.csv', '') for name in os.listdir('results') if not name.startswith('.')]


def predictions(names=None, test=True):
    if names is None:
        names = team_names()

    folder = 'tests' if test else 'results'

    team_dfs = {}
    for name in names:
        df = pd.read_csv(os.path.join(folder, name + '.csv'), delimiter=';')
        team_dfs[name] = df
    return team_dfs


def prediction(team, test=True):
    return predictions([team], test)[team]


def splits():
    known_combinations = pd.read_csv('data/splits.csv', delimiter=';').astype(bool)
    known_combinations.index = pd.Index([', '.join(row.index[row]) for _, row
                                         in known_combinations.iterrows()])
    known_combinations.index.name = 'known'
    return known_combinations


def orders_train():
    train = pd.read_csv('data/orders_train.txt', delimiter=';')
    train['returnQuantity'] = train['returnQuantity'].astype(bool)
    return train.set_index(['orderID', 'articleID', 'colorCode', 'sizeCode'],
                           drop=False)


def orders_class():
    df = pd.read_csv('data/orders_class.txt', delimiter=';')
    return df[['orderID', 'articleID', 'colorCode', 'sizeCode']]
