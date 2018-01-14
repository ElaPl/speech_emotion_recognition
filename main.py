#!/usr/bin/python3.5

"""
.. module:: project_documentation
    :platform: OS X
    :synopsis: module illustrating how to document python source code

.. moduleauthor:: Patrick Kennedy <patkennedy79@gmail.com>
"""

from matplotlib import pyplot as plt
import sys
import pymysql
from warnings import filterwarnings

from hmm_main import hmm_main
from voice_module import *
from KNN import KNN
from helper_file import *

emotions = ["anger", "boredom", "happiness", "sadness"]


def draw_freq_histogram():
    train_set = build_file_set('Berlin_EmoDatabase/wav/' + sys.argv[1] + '/*.wav')
    frame_size = 512

    for i in range(0, len(train_set)):
        freq_vector = get_freq_vector(train_set[i][0], 512)
        print(freq_vector)
        print()
        sample_rate = get_sample_rate(train_set[i][0])

        bins = np.arange(0, len(freq_vector), 1)  # fixed bin size
        plt.plot(bins, freq_vector)
        plt.title('Fundamental freq of ' + train_set[i][0])
        plt.xlabel('time, frame_length= ' + str(sample_rate / frame_size))
        plt.ylabel('Frequency')

        plt.show()


def draw_energy_histogram():
    train_set = build_file_set('Berlin_EmoDatabase/wav/' + sys.argv[1] + '/*.wav')

    for i in range(0, len(train_set)):
        sample_rate = get_sample_rate(train_set[i][0])
        energy_vector = get_energy_vector(train_set[i][0], int(sample_rate / 4))

        bins = np.arange(0, len(energy_vector), 1)
        plt.plot(bins, energy_vector)
        plt.title('Energy histogram of ' + train_set[i][0])
        plt.xlabel('Time')
        plt.ylabel('Energy')

        plt.show()


def get_feature_set(path_pattern, db_name, db_password, summary_table):
    """Funkcja pobiera zbiór wectorów cech dla każego pliku z folderu "path_pattern".
    Jeżeli dane istnieją w bazie danych, zostają one pobrane i zwrócone.
    W przeciwnym wypadku dla każdego pliku zostają obliczone jego właściwości. Jeżeli istnieje połaczenie z bazą danych,
    zostają zapisane. Obliczone wektory cech zostają zwrócone

    :param path_pattern: Ścieżka do folderu zawierające dane
    :type path_pattern: string
    :param db_name: Nazwa bazy danych
    :type db_name: string
    :param db_password: Hasło do bazy danych
    :type db_password: string
    :param summary_table: Pomocnicza tablica do której powinny być zapisywane wyniki
    :type summary_table: dict
    :return: something
    :rtype: string
    :raises: TypeError """
    print_debug("train")
    is_connection = True

    try:
        db = pymysql.connect(host="localhost", user="root", passwd=db_password, db=db_name)
    except pymysql.Error as e:
        print_debug("Can't connect to db")
        is_connection = False

    if is_connection is True:
        print_debug("Connected to db")
        cursor = db.cursor()

    all_pitch_features_vector = []
    all_energy_features_vector = []
    all_summary_pitch_features_vector = []

    if (is_connection is True) and (knn_db.is_training_set_exists(cursor) is True):
        print_debug("Train set exists, start downloading data")
        for row in knn_db.select_all_from_db(cursor, knn_db.pitch_train_set_name):
            all_pitch_features_vector.append([list(row[0: -1]), row[-1]])
            summary_table[row[-1]]["trained"] += 1

        print_progress_bar(2, 3, prefix='Training progress:', suffix='Complete', length=50)
        for row in knn_db.select_all_from_db(cursor, knn_db.summary_pitch_train_set_name):
            all_summary_pitch_features_vector.append([list(row[0: -1]), row[-1]])

        print_progress_bar(3, 3, prefix='Training progress:', suffix='Complete', length=50)
        for row in knn_db.select_all_from_db(cursor, knn_db.energy_train_set_name):
            all_energy_features_vector.append([list(row[0: -1]), row[-1]])

    else:
        print_debug("Train set doesn't exist, start computing data")

        test_set = build_file_set(path_pattern)
        num_files = len(test_set)
        for i in range(num_files):
            print_progress_bar(i + 1, num_files, prefix='Training progress:', suffix='Complete', length=50)
            file = test_set[i][0]
            emotion = test_set[i][1]
            summary_table[emotion]["trained"] += 1

            pitch_feature_vectors, energy_feature_vectors = get_feature_vectors(file)
            summary_pitch_feature_vector = get_summary_pitch_feature_vector(pitch_feature_vectors)

            for vec in pitch_feature_vectors:
                all_pitch_features_vector.append([vec, emotion])

            for vec in energy_feature_vectors:
                all_energy_features_vector.append([vec, emotion])

            all_summary_pitch_features_vector.append([summary_pitch_feature_vector, emotion])

    if is_connection and (knn_db.is_training_set_exists(cursor) is False):
        knn_db.create_training_set(db, cursor)

        print_progress_bar(0, 3, prefix='Saving in database:', suffix='Complete', length=50)
        safe_in_database(db, cursor, all_pitch_features_vector, knn_db.pitch_train_set_name)

        print_progress_bar(1, 3, prefix='Saving in database:', suffix='Complete', length=50)
        safe_in_database(db, cursor, all_energy_features_vector, knn_db.energy_train_set_name)

        print_progress_bar(2, 3, prefix='Saving in database:', suffix='Complete', length=50)
        safe_in_database(db, cursor, all_summary_pitch_features_vector, knn_db.summary_pitch_train_set_name)

        print_progress_bar(3, 3, prefix='Saving in database:', suffix='Complete', length=50)

    return all_pitch_features_vector, all_energy_features_vector, all_summary_pitch_features_vector


def knn_compute_emotions(path_pattern, KNN_modules, summary_table):
    file_set = build_file_set(path_pattern)
    num_files = len(file_set)

    for i in range(num_files):
        print_progress_bar(i + 1, num_files, prefix='Computing progress:', suffix='Complete', length=50)
        file = file_set[i][0]
        emotion = file_set[i][1]

        feature_vectors = {}
        feature_vectors["pitch"], feature_vectors["energy"] = get_feature_vectors(file)
        feature_vectors["pitch_summary"] = get_summary_pitch_feature_vector(feature_vectors["pitch"])

        possible_emotions = []
        for feature in features.keys():
            print("Compute emotion " + feature)
            possible_emotions.extend(KNN_modules[feature].compute_emotion(feature_vectors[feature],
                                     features[feature]["nearest_neighbour"]))

        computed_emotion = get_most_frequent_emotion(possible_emotions, emotions)

        summary_table[emotion]["tested"] += 1
        if computed_emotion == emotion:
            summary_table[emotion]["guessed"] += 1
        else:
            summary_table[emotion][computed_emotion] += 1


def main_KNN(train_path_pattern, test_path_pattern, db_name, db_password):
    summary_table = create_summary_table(emotions)

    db, cursor = connect_to_database(db_name, db_password)
    training_set = get_training_feature_set(train_path_pattern, cursor)

    for emotion in emotions:
        for vec in training_set["pitch"]:
            if vec[1] == emotion:
                summary_table[emotion]["trained"] += 1

    if (db is not None) and (knn_db.is_training_set_exists(cursor) is False):
        save_features_set_in_database(db, cursor, training_set)


    KNN_modules = {}
    for feature in features:
        KNN_modules[feature] = KNN(emotions, training_set[feature])

    knn_compute_emotions(test_path_pattern, KNN_modules, summary_table)

    print_summary(summary_table, emotions)


if __name__ == "__main__":
    filterwarnings('ignore', category=pymysql.Warning)
    db_name = knn_db.DB_NAME
    db_password = "Mout"

    if len(sys.argv) == 1:
        print("Use either KNN or HMM option")

    if len(sys.argv) > 4:
        db_name = sys.argv[2]
        db_password = sys.argv[3]

    if sys.argv[1] == 'KNN':
        main_KNN('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', db_name, db_password)
    else:
        hmm_main('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', db_name, db_password,
                 emotions)

