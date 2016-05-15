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


def orders_train():
    original_train = pd.read_csv('data/orders_train.txt', delimiter=';')
    original_train['returnQuantity'] = original_train['returnQuantity'].astype(bool)
    return original_train.set_index(['orderID', 'articleID', 'colorCode', 'sizeCode'],
                                    drop=False)


def orders_class():
    df = pd.read_csv('data/orders_class.txt', delimiter=';')
    return df[['orderID', 'articleID', 'colorCode', 'sizeCode']]
