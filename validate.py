import os

import pandas as pd


def validate_all():
    results = ['results/' + file for file in os.listdir('results') if not file.startswith('.')]
    tests = ['tests/' + file for file in os.listdir('tests') if not file.startswith('.')]

    expected_columns = ['orderID', 'articleID', 'colorCode', 'sizeCode', 'confidence', 'prediction']

    for file in results + tests:
        df = pd.read_csv(file, delimiter=';')

        try:
            assert list(df.columns) == expected_columns
        except AssertionError:
            print('%s: Wrong columns' % file)
            return df

        try:
            assert ~(df[['orderID', 'articleID', 'colorCode', 'sizeCode', 'prediction']]
                     .isnull().any().any())
        except AssertionError:
            print('%s: Contains nulls' % file)

        try:
            assert df['orderID'].astype(str).str.startswith('a').all()
            assert df['articleID'].astype(str).str.startswith('i').all()
        except AssertionError:
            print('%s: orderID, articleID or both are wrong' % file)
            return df

        try:
            if file.startswith('results/'):
                assert len(df) == 341098
        except AssertionError:
            print('%s: Wrong number of lines in classification file' % file)
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


def _fix_team_c(path):
    df = pd.read_csv(path, index_col=0, sep=';')
    df = df.drop('quantity', axis=1)
    df['orderID'] = 'a' + df['orderID'].astype(str)
    df['articleID'] = 'i' + df['articleID'].astype(str)
    df.to_csv(path, index=False, sep=';')
