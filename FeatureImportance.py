from sklearn.ensemble import ExtraTreesClassifier
from helper_file import  normalize
from knn_main import knn_get_training_feature_set_from_dir


def feature_importance(path_pattern, emotions):
    training_set = knn_get_training_feature_set_from_dir(path_pattern, emotions)
    for feature in ["features"]:
        min_f, max_f = normalize(training_set[feature])

        x, y = [training_set[feature][i][0] for i in range(len(training_set[feature]))], \
               [training_set[feature][i][1] for i in range(len(training_set[feature]))]

        model = ExtraTreesClassifier()
        model.fit(x, y)
        result = model.feature_importances_

        print("\n--------------------------------------------------")
        print(feature)
        for i in range(len(result)):
            print()
            print("feature [%d] : %lf" %(i, result[i]))

