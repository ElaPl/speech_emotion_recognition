from sklearn.ensemble import ExtraTreesClassifier
from helper_file import  normalize
from knn_main import knn_get_training_feature_set_from_dir
import numpy

def feature_importance(path_pattern, emotions):
    """
    Funkcja tworzy wektor cech z plików podanych w path_pattern i wypisuje ważność każdej z cechy
    :param path_pattern: Ścieżka do plików, z których mają być obliczone wektory cech
    :param emotions: Lista emocji które mają być uwzględnione
    :return: None
    """
    training_set = knn_get_training_feature_set_from_dir(path_pattern, emotions)
    for feature in ["features"]:
        normalize(training_set[feature])

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


def avg_features_val(path_pattern, emotions):
    training_set = knn_get_training_feature_set_from_dir(path_pattern, emotions)
    features_num = len(training_set["features"][0][0])

    training_set_emo = {}
    for emotion in emotions:
        training_set_emo[emotion] = []

    for vec in training_set["features"]:
        training_set_emo[vec[1]].append(vec[0])

    # print(training_set_emo[emotions[0]])

    for emotion in training_set_emo.keys():
        print(emotion)
        mean_features = []
        for i in range(0, features_num):
            feature_set = [a for a in (training_set_emo[emotion][j][i] for j in range(0, len(training_set_emo[emotion])))]
            mean_features.append(numpy.mean(feature_set))

        print(mean_features)

        # mean_features = []
        # for i in range(0, features_num):
        #     feature_set = [a for a in (training_set["features"][j][0][i] for j in range(0, len(training_set))
        #                                if training_set["features"][j][1] == emotion)]
        #     mean_features.append(numpy.mean(feature_set))
        # print(mean_features)
