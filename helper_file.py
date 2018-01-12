from math import sqrt, pow
import glob
import os
import operator
import knn_database as knn_db


# If True than debug print will be displayed
debug = True


def print_debug(text):
    if debug is True:
        print(text)

# Oblicza odległosć euklidesową pomiedzy dwoma wektorami
def euclidean_distance(vec1, vec2):
    if len(vec1) != len(vec2):
        print("Wektory mają różną długosc?")
    dist = 0
    for i in range(0, len(vec1)):
        dist += pow(vec1[i] - vec2[i], 2)

    return sqrt(dist)


def normalize(table):
    min_table = []
    max_table = []

    feature_vec_len = len(table[0][0])

    for feature_id in range(len(table[0][0])):
        min_table.append(min(table[i][0][feature_id] for i in range(0, len(table))))
        max_table.append(max(table[i][0][feature_id] for i in range(0, len(table))))

    for i in range(len(table)):
        for feature_id in range(feature_vec_len):
            if max_table[feature_id] != min_table[feature_id]:
                table[i][0][feature_id] = (table[i][0][feature_id] - min_table[feature_id]) / (max_table[feature_id] - min_table[feature_id])

    return min_table, max_table


# Normalizuje wektor testowy wartościami podanymi jako argumenty
def normalize_vector(test, min_features, max_features):
    for feature_id in range(len(test)):
        if (max_features[feature_id] != min_features[feature_id]):
            test[feature_id] = (test[feature_id] - min_features[feature_id]) / (max_features[feature_id] - min_features[feature_id])


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

