from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from helper_file import connect_to_database, normalize
from knn_main import knn_get_training_feature_set_from_db, knn_get_training_feature_set_from_dir
import knn_database as knn_db

features = {}
features["pitch"] = [
    "vocal_range, %lf\n",
    "max_freq %lf\n",
    "min_freq %lf\n",
    "avg_freq %lf\n",
    "dynamic_tones_percent %lf\n",
    "percent_of_falling_tones %lf\n",
    "percent_of_rising_tones %lf\n",
    "relative_standard_deviation: %lf\n"]

features["pitch_summary"] = [
    "freq_range, %lf\n",
    "max_freq %lf\n",
    "min_freq %lf\n",
    "avg_freq %lf\n",
    "dynamic_tones_percent %lf\n",
    "relative_standard_deviation: %lf\n"
]

features["energy"] = [
    "relative_std_deviation, %lf\n",
    "zero_crossing_rate %lf\n",
    "rms_db %lf\n",
    "peak_db %lf\n",
]


def feature_importance(path_pattern, db_name, db_password):
    db, cursor = connect_to_database(db_name, db_password)
    if (cursor is not None) and knn_db.is_training_set_exists(cursor, knn_db.KNN_DB_TABLES):
        training_set = knn_get_training_feature_set_from_db(cursor)
    else:
        training_set = knn_get_training_feature_set_from_dir(path_pattern)

    for feature in features:
        min_f, max_f = normalize(training_set[feature])

        x, y = [training_set[feature][i][0] for i in range(len(training_set[feature]))], \
               [training_set[feature][i][1] for i in range(len(training_set[feature]))]

        # fit an Extra Trees model to the data
        model = ExtraTreesClassifier()
        model.fit(x, y)
        result = model.feature_importances_

        print("\n--------------------------------------------------")
        print(feature)
        for i in range(len(result)):
            print(result[i])
            print(features[feature][i] %( result[i]))

