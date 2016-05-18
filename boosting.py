import pandas as pd

import evaluate
import merge
import load

from sklearn import ensemble


def boosted_class_data():
    # Load evaluation data
    test_columns = ['returnQuantity', 'articleID', 'productGroup', 'customerID', 'voucherID']
    test_predictions = merge.merged_predictions(test=True, keep_columns=test_columns)
    test_train = evaluate.test_complement(test_predictions)

    # Load classification data
    class_columns = ['articleID', 'productGroup', 'customerID', 'voucherID',]
    class_predictions = merge.merged_predictions(keep_columns=class_columns)
    class_train = load.orders_train()

    # Impute zeroes and convert confidenes to std-distances
    class_imputed = merge.impute_confidence(class_predictions)
    test_imputed = merge.impute_confidence(test_predictions)

    categories = ['articleID', 'productGroup', 'customerID', 'voucherID']
    X_train, y_train = merge.boosting_features(test_train, test_imputed, categories)
    X_class, class_dis, class_agr = merge.class_features(class_train, class_imputed, categories)

    clfF = ensemble.RandomForestClassifier(n_jobs=-1, min_samples_leaf=32, min_samples_split=2, criterion='gini',
                                           max_depth=8, max_features=4)

    clfF.fit(X_train, y_train)
    y_merged = clfF.predict(X_class)
    class_dis['merged_prediction'] = y_merged

    class_agr['merged_prediction'] = class_imputed.prediction.A

    class_unified = pd.concat([class_dis, class_agr])
    return class_unified
