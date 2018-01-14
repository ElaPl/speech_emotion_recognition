from math import sqrt, pow
import glob
import os
import operator
import knn_database as knn_db
import pymysql
from voice_module import get_feature_vectors

# If True than debug print will be displayed
debug = True


def print_debug(text):
    if debug is True:
        print(text)

# Oblicza odległosć euklidesową pomiedzy dwoma wektorami
def euclidean_distance(vec1, vec2):
    """
    :param vec1: wektor cech
    :param vec2: wektor cech
    :return: Dystans pomiedzy dwoma wektorami
    """
    if len(vec1) != len(vec2):
        print("Wektory mają różną długosc?")
    dist = 0
    for i in range(0, len(vec1)):
        dist += pow(vec1[i] - vec2[i], 2)

    return sqrt(dist)


def normalize(feature_vector_set):
    """
    :param: feature_vector_set: Zbiór wektorów cech do znormalizowania
    :return: * wektor najmniejszych wartości z każdej cechy
             * wektor największych wartości z każdej cechy
    """
    min_features_vector = []
    max_features_vector = []

    feature_vec_len = len(feature_vector_set[0][0])

    for feature_id in range(len(feature_vector_set[0][0])):
        min_features_vector.append(min(feature_vector_set[i][0][feature_id] for i in range(0, len(feature_vector_set))))
        max_features_vector.append(max(feature_vector_set[i][0][feature_id] for i in range(0, len(feature_vector_set))))

    for i in range(len(feature_vector_set)):
        for feature_id in range(feature_vec_len):
            if max_features_vector[feature_id] != min_features_vector[feature_id]:
                feature_vector_set[i][0][feature_id] = (feature_vector_set[i][0][feature_id] - min_features_vector[feature_id]) / (max_features_vector[feature_id] - min_features_vector[feature_id])

    return min_features_vector, max_features_vector


# Normalizuje wektor testowy wartościami podanymi jako argumenty
def normalize_vector(feature_vector, min_features, max_features):
    """ Normalizuje wektor testowy wartościami podanymi jako argumenty
    :param feature_vector: wektor cech do znormalizowania
    :param min_features: wektor najmniejszych wartości z każdej cechy, którymi należy znormalizować podany wektor
    :param max_features: wektor największych wartości z każdej cechy, którymi należy znormalizować podany wektor
    """
    for feature_id in range(len(feature_vector)):
        if (max_features[feature_id] != min_features[feature_id]):
            feature_vector[feature_id] = (feature_vector[feature_id] - min_features[feature_id]) / (max_features[feature_id] - min_features[feature_id])


def build_file_set(pattern):
    train_set = []
    for path_and_file in glob.iglob(pattern, recursive=True):
        if path_and_file.endswith('.wav'):
            path, filename = os.path.split(path_and_file)
            emotion = os.path.basename(path)
            train_set.append([path_and_file, emotion])
    return train_set


def safe_in_database(database, cursor, table, db_table_name):
    for row in table:
        knn_db.save_in_dbtable(database, cursor, row[0], row[1], db_table_name)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()


def get_most_frequent_emotion(emotions_set, possible_emotions):
    emotion_counter = {}
    for emotion in possible_emotions:
        emotion_counter[emotion] = 0

    for emotion in emotions_set:
        emotion_counter[emotion] += 1

    return max(emotion_counter.items(), key=operator.itemgetter(1))[0]


def print_summary(summary_table, emotions_list):
    """function illustrating how to document python source code"""
    print()
    for emotion in summary_table.keys():
        string = emotion + '  :\t'
        for i in range(len(emotions_list)):
            string += emotions_list[i] + ": " + str(summary_table[emotion][emotions_list[i]]) + ',\t'
        string += "guessed: " + str(summary_table[emotion]["guessed"]) + ',\t'
        string += "tested: " + str(summary_table[emotion]["tested"]) + ',\t'
        string += "trained: " + str(summary_table[emotion]["trained"])
        print(string)


def create_summary_table(emotions_list):
    summary_table = {}
    for emotion in emotions_list:
        summary_table[emotion] = {}
        for result_emotion in emotions_list:
            summary_table[emotion][result_emotion] = 0
        summary_table[emotion]["guessed"] = 0
        summary_table[emotion]["tested"] = 0
        summary_table[emotion]["trained"] = 0

    return summary_table


def connect_to_database(db_name, db_password):
    try:
        db = pymysql.connect(host="localhost", user="root", passwd=db_password, db=db_name)
        cursor = db.cursor()
        return db, cursor
    except pymysql.Error as e:
        print_debug("Can't connect to db")
        return None, None

features = {
    "pitch": {
        "db_table_name": knn_db.pitch_train_set_name,
        "nearest_neighbour": 15
    },

    "energy": {
        "db_table_name": knn_db.energy_train_set_name,
        "nearest_neighbour": 12
    },

    "pitch_summary": {
        "db_table_name": knn_db.summary_pitch_train_set_name,
        "nearest_neighbour": 4
    }
}


def get_training_feature_set(path_pattern, cursor):
    features_set = {}
    for feature in features.keys():
        features_set[feature] = []

    if (cursor is not None) and knn_db.is_training_set_exists(cursor):
        print_debug("Train set exists, start downloading data")
        for feature in features.keys():
            for row in knn_db.select_all_from_db(cursor, features[feature]["db_table_name"]):
                features_set[feature].append([list(row[0: -1]), row[-1]])
    else:
        print_debug("Train set doesn't exist, start computing data")

        test_set = build_file_set(path_pattern)
        num_files = len(test_set)
        for i in range(num_files):
            print_progress_bar(i + 1, num_files, prefix='Training progress:', suffix='Complete', length=50)
            file = test_set[i][0]
            emotion = test_set[i][1]

            pitch_feature_vectors, energy_feature_vectors, summary_pitch_feature_vector = \
                get_feature_vectors(file)

            for vec in pitch_feature_vectors:
                features_set["pitch"].append([vec, emotion])

            for vec in energy_feature_vectors:
                features_set["energy"].append([vec, emotion])

            features_set["pitch_summary"].append([summary_pitch_feature_vector, emotion])

    return features_set


def save_features_set_in_database(db, cursor, features_set):
    knn_db.create_training_set(db, cursor)
    for feature in features:
        safe_in_database(db, cursor, features_set[feature], features[feature]["db_table_name"])
