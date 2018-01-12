import numpy as np
from matplotlib import pyplot as plt
import sys
import pymysql
from warnings import filterwarnings
from sklearn.cluster import KMeans

import knn_database as knn_db
from voice_module import *
from KNN import KNN
from HMM import HMM
from helper_file import *

emotions = ["anger", "boredom", "happiness", "sadness"]


def print_summary(summary_table):
    print()
    for emotion in summary_table.keys():
        string = emotion + '  :\t'
        for i in range(len(emotions)):
            string += emotions[i] + ": " + str(summary_table[emotion][emotions[i]]) + ',\t'
        string += "guessed: " + str(summary_table[emotion]["guessed"]) + ',\t'
        string += "tested: " + str(summary_table[emotion]["tested"]) + ',\t'
        string += "trained: " + str(summary_table[emotion]["trained"])
        print(string)


def create_summary_table():
    summary_table = {}
    for emotion in emotions:
        summary_table[emotion] = {}
        for result_emotion in emotions:
            summary_table[emotion][result_emotion] = 0
        summary_table[emotion]["guessed"] = 0
        summary_table[emotion]["tested"] = 0
        summary_table[emotion]["trained"] = 0

    return summary_table


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


def get_train_set(train_path_pattern, db_name, db_password, summary_table):
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
        for row in knn_db.train_knn_from_db(cursor, knn_db.pitch_train_set_name):
            all_pitch_features_vector.append([list(row[0: -1]), row[-1]])
            summary_table[row[-1]]["trained"] += 1

        print_progress_bar(2, 3, prefix='Training progress:', suffix='Complete', length=50)
        for row in knn_db.train_knn_from_db(cursor, knn_db.summary_pitch_train_set_name):
            all_summary_pitch_features_vector.append([list(row[0: -1]), row[-1]])

        print_progress_bar(3, 3, prefix='Training progress:', suffix='Complete', length=50)
        for row in knn_db.train_knn_from_db(cursor, knn_db.energy_train_set_name):
            all_energy_features_vector.append([list(row[0: -1]), row[-1]])

    else:
        print_debug("Train set doesn't exist, start computing data")

        test_set = build_file_set(train_path_pattern)
        num_files = len(test_set)
        for i in range(num_files):
            print_progress_bar(i + 1, num_files, prefix='Training progress:', suffix='Complete', length=50)
            file = test_set[i][0]
            emotion = test_set[i][1]
            summary_table[emotion]["trained"] += 1

            pitch_feature_vectors, energy_feature_vectors, summary_pitch_feature_vector \
                = get_feature_vectors(file)

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


def main_KNN(train_path_pattern, test_path_pattern, db_name, db_password):
    summary_table = create_summary_table()

    all_pitch_features_vector, all_energy_features_vector, all_summary_pitch_features_vector = \
        get_train_set(train_path_pattern, db_name, db_password, summary_table)

    pitch_knn_module = KNN(emotions, all_pitch_features_vector)
    summary_pitch_knn_module = KNN(emotions, all_summary_pitch_features_vector)
    energy_knn_module = KNN(emotions, all_energy_features_vector)

    print_debug("Computing emotions")
    test_set = build_file_set(test_path_pattern)
    num_files = len(test_set)

    for i in range(num_files):
        print_progress_bar(i+1, num_files, prefix='Computing progress:', suffix='Complete', length=50)
        file = test_set[i][0]
        emotion = test_set[i][1]

        pitch_feature_vectors, energy_feature_vectors, summary_pitch_feature_vector \
            = get_feature_vectors(file)

        possible_emotions = pitch_knn_module.compute_emotion(pitch_feature_vectors, 15)
        possible_emotions.extend(summary_pitch_knn_module.get_emotion(summary_pitch_feature_vector, 4))
        possible_emotions.extend(energy_knn_module.compute_emotion(energy_feature_vectors, 12))

        computed_emotion = get_most_frequent_emotion(possible_emotions, emotions)

        summary_table[emotion]["tested"] += 1
        if computed_emotion == emotion:
            summary_table[emotion]["guessed"] += 1
        else:
            summary_table[emotion][computed_emotion] += 1

    print_summary(summary_table)


def average_observations(feature_vector_set):

    min_features_vec, max_features_vec = normalize(feature_vector_set)

    kmeans = KMeans(n_clusters=1000).fit(
        [feature_vector_set[i][0] for i in range(len(feature_vector_set))])

    observations = (kmeans.cluster_centers_).tolist()

    return observations, min_features_vec, max_features_vec


def claster(vector, data):
    min_dist = euclidean_distance(vector, data[0])
    min_vector = data[0]
    for vec in data:
        dist = euclidean_distance(vector, vec)
        if dist < min_dist:
            min_dist = dist
            min_vector = vec

    return min_vector


def get_observations_vectors(file, feature, min_features_vec, max_features_vec, possible_observations):

    pitch_feature_vectors, energy_feature_vectors, summary_pitch_feature_vector \
        = get_feature_vectors(file)

    normalized_observations = []
    if feature == "pitch":
        for vec in pitch_feature_vectors:
            normalize_vector(vec, min_features_vec, max_features_vec)
            normalized_observations.append(claster(vec, possible_observations))
    else:
        for vec in energy_feature_vectors:
            normalize_vector(vec, min_features_vec, max_features_vec)
            normalized_observations.append(claster(vec, possible_observations))

    observation_sequence_vec = []
    #biorę 1,5 sek wektory obserwacji wypowiedzi
    for i in range(len(normalized_observations) - 2*6 - 2):
        observation = [normalized_observations[j + i] for j in range(0, 12, 2)]
        observation_sequence_vec.append(observation)

    return observation_sequence_vec


def get_possible_observations(train_path_pattern, db_name, db_password, summary_table):
    all_pitch_features_vector, all_energy_features_vector, all_summary_pitch_features_vector = \
        get_train_set(train_path_pattern, db_name, db_password, summary_table)

    possible_observations = {}
    min_max_features = {"pitch": {}, "energy": {}}

    possible_observations["pitch"], min_max_features["pitch"]["min"], min_max_features["pitch"]["max"] = \
        average_observations(all_pitch_features_vector)
    possible_observations["energy"], min_max_features["energy"]["min"], min_max_features["energy"]["max"] = \
        average_observations(all_energy_features_vector)

    return possible_observations, min_max_features


def get_hmm_train_set(train_path_pattern, features, min_max_features, possible_observations):
    file_set = build_file_set(train_path_pattern)
    num_files = len(file_set)
    train_set = {}
    for feature in features:
        train_set[feature] = {}
        for emotion in emotions:
            train_set[feature][emotion] = []

    for i in range(num_files):
        print_progress_bar(i + 1, num_files, prefix='Preparing:', suffix='Complete', length=50)
        file = file_set[i][0]
        trained_emotion = file_set[i][1]

        for feature in features:
            obs_vec = get_observations_vectors(file, feature, min_max_features[feature]["min"],
                                               min_max_features[feature]["max"], possible_observations[feature])
            for obs in obs_vec:
                train_set[feature][trained_emotion].append(obs)

    return train_set


def main_HMM(train_path_pattern, test_path_pattern, db_name, db_password):
    summary_table = create_summary_table()
    features = ["pitch", "energy"]
    possible_observations, min_max_features = get_possible_observations(train_path_pattern, db_name, db_password,
                                                                        summary_table)
    num_of_states = 6
    trasition_ppb = [[0.2, 0.8],
                     [0.2, 0.8],
                     [0.2, 0.8],
                     [0.2, 0.8],
                     [0.2, 0.8],
                     [0.2, 0]]

    HMM_modules = {}
    for f in features:
        HMM_modules[f] = {}

    for feature in features:
        for emotion in emotions:
            HMM_modules[feature][emotion] = HMM(trasition_ppb, num_of_states, possible_observations[feature])

    train_set = get_hmm_train_set(train_path_pattern, features, min_max_features, possible_observations)
    for feature in features:
        for emotion in emotions:
            if train_set[feature][emotion]:
                HMM_modules[feature][emotion].learn(train_set[feature][emotion], 0.001)

    file_set = build_file_set(test_path_pattern)
    num_files = len(file_set)
    for i in range(num_files):
        print_progress_bar(i + 1, num_files, prefix='Testing progress:', suffix='Complete', length=50)
        file = file_set[i][0]
        tested_emotion = file_set[i][1]
        possible_emotions = []

        for feature in features:
            obs_vec = get_observations_vectors(file, feature, min_max_features[feature]["min"],
                                               min_max_features[feature]["max"], possible_observations[feature])

            for obs in obs_vec:
                max_ppb = 0
                for emotion, hmm_module in HMM_modules[feature].items():
                    value = hmm_module.evaluate(obs)
                    if value > max_ppb:
                        max_ppb = value
                        most_ppb_emotion = emotion

                possible_emotions.append(most_ppb_emotion)

        computed_emotion = get_most_frequent_emotion(possible_emotions, emotions)
        summary_table[tested_emotion]["tested"] += 1
        summary_table[tested_emotion][computed_emotion] += 1

        if computed_emotion == tested_emotion:
            summary_table[tested_emotion]["guessed"] += 1

    print_summary(summary_table)


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
    main_HMM('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', db_name, db_password)

