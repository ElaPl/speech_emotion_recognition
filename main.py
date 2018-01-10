import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import sys
import pymysql
from warnings import filterwarnings

import knn_database as knn_db
from voice_module import VoiceModule
from KNN import KNN
from HMM import HMM

emotions = ["anger", "boredom", "happiness", "sadness"]


def build_file_set(pattern):
    train_set = []
    for path_and_file in glob.iglob(pattern, recursive=True):
        if path_and_file.endswith('.wav'):
            path, filename = os.path.split(path_and_file)
            emotion = os.path.basename(path)
            train_set.append([path_and_file, emotion])
    return train_set


def print2DArray(array_2D):
    for row in range(0, len(array_2D)):
        print(array_2D[row][0] + " " + array_2D[row][1])


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()


def get_most_frequent_emotion(possible_emotions):
    emotion_counter = {}
    for emotion in emotions:
        emotion_counter[emotion] = 0

    for emotion in possible_emotions:
        emotion_counter[emotion] += 1

    computed_emotion = emotions[0]
    computed_emotion_occurance = emotion_counter[computed_emotion]
    for emotion, num_occurance in emotion_counter.items():
        if num_occurance > computed_emotion_occurance:
            computed_emotion = emotion
            computed_emotion_occurance = num_occurance

    return computed_emotion


def draw_freq_histogram():
    train_set = build_file_set('Berlin_EmoDatabase/wav/' + sys.argv[1] + '/*.wav')
    frame_size = 512
    voice_m = VoiceModule()

    for i in range(0, len(train_set)):
        freq_vector = voice_m.get_freq_vector(train_set[i][0], 512)
        print(freq_vector)
        print()
        sample_rate = voice_m.get_sample_rate(train_set[i][0])

        bins = np.arange(0, len(freq_vector), 1)  # fixed bin size
        plt.plot(bins, freq_vector)
        plt.title('Fundamental freq of '+train_set[i][0])
        plt.xlabel('time, frame_length= ' + str(sample_rate/frame_size))
        plt.ylabel('Frequency')

        plt.show()


def print_feature_vector():
    train_set = build_file_set('Berlin_EmoDatabase/wav/' + sys.argv[1] + '/*.wav')
    frame_size = 512
    voice_m = VoiceModule(frame_size)

    for i in range(0, len(train_set)):
        feature_vector = voice_m.get_feature_vector(train_set[i][0], 512)
        for j in range(0, len(feature_vector)):
            print(feature_vector[j], end=" ")
        print("\n")

#
# def train(file):
#     train_set = build_file_set(file)
#
#     voice_m = VoiceModule(512)
#     knn_module = KNN(emotions, 8)
#
#     for i in range(0, len(train_set)):
#         print_progress_bar(i + 1, len(train_set), prefix='Training progress:', suffix='Complete', length=50)
#         feature_vector = voice_m.get_feature_vector(train_set[i][0])
#         for j in range(0, len(feature_vector)):
#             knn_module.train(feature_vector[j], train_set[i][1])
#
#     return knn_module

def are_equal(vecA, vecB):
    if len(vecA) != len(vecB):
        print("Not equal")
        return False
    else:
        for i in range(0, len(vecA)):
            if vecA[i] != vecB[i]:
                print("Not the same")
                return False

    return True

def compare_feature_vectors(file):
    knn_moduleA = train('Berlin_EmoDatabase/wav/a*/*/*.wav')
    knn_moduleB = train('Berlin_EmoDatabase/wav/a*/*/*.wav')

    for i in range(0, len(knn_moduleA.training_set)):
        if (False == are_equal(knn_moduleA.training_set[i]["training_vec"], knn_moduleB.training_set[i]["training_vec"])
            or False == are_equal(knn_moduleA.training_set[i]["norm_vec"], knn_moduleB.training_set[i]["norm_vec"])
            or knn_moduleA.training_set[i]["min"] != knn_moduleB.training_set[i]["min"]
            or knn_moduleA.training_set[i]["max"] != knn_moduleB.training_set[i]["max"]
            or (False == are_equal(knn_moduleA.training_set[i]["state"], knn_moduleB.training_set[i]["state"]))):
            print("Not equal")
            break
        print("Equal")


def draw_energy_histogram():
    train_set = build_file_set('Berlin_EmoDatabase/wav/' + sys.argv[1] + '/*.wav')
    frame_size = 512
    voice_m = VoiceModule()

    for i in range(0, len(train_set)):
        sample_rate = voice_m.get_sample_rate(train_set[i][0])
        energy_vector = voice_m.get_energy_vector(train_set[i][0], int(sample_rate/4))

        bins = np.arange(0, len(energy_vector), 1)
        plt.plot(bins, energy_vector)
        plt.title('Energy histogram of ' + train_set[i][0])
        plt.xlabel('Time')
        plt.ylabel('Energy')

        plt.show()


def train(train_path_pattern, voice_m, db_name, pitch_knn_module, summary_pitch_knn_module,
          energy_knn_module ,summary_table, frame_length):

    db = pymysql.connect(host="localhost", user="root", passwd="Mout", db=db_name)
    cursor = db.cursor()

    if knn_db.is_training_set_exists(cursor):
        print_progress_bar(1, 3, prefix='Training progress:', suffix='Complete', length=50)
        for row in knn_db.train_knn_from_db(cursor, knn_db.pitch_train_set_name):
            pitch_knn_module.train(row[0: -1], row[-1])
            summary_table[row[-1]]["trained"] += 1

        print_progress_bar(2, 3, prefix='Training progress:', suffix='Complete', length=50)
        for row in knn_db.train_knn_from_db(cursor, knn_db.summary_pitch_train_set_name):
            summary_pitch_knn_module.train(row[0: -1], row[-1])
            summary_table[row[-1]]["trained"] += 1

        print_progress_bar(3, 3, prefix='Training progress:', suffix='Complete', length=50)
        for row in knn_db.train_knn_from_db(cursor, knn_db.energy_train_set_name):
            energy_knn_module.train(row[0: -1], row[-1])
            summary_table[row[-1]]["trained"] += 1

    else:
        knn_db.create_training_set(db, cursor)
        train_set = build_file_set(train_path_pattern)

        for i in range(0, len(train_set)):
            print_progress_bar(i + 1, len(train_set), prefix='Training progress:', suffix='Complete', length=50)
            summary_table[train_set[i][1]]["trained"] += 1

            sample_rate = voice_m.get_sample_rate(train_set[i][0])
            pitch_feature_vectors = voice_m.get_pitch_feature_vector(train_set[i][0], frame_length)
            for j in range(0, len(pitch_feature_vectors)):
                pitch_knn_module.train(pitch_feature_vectors[j], train_set[i][1])
                knn_db.save_in_dbtable(db, cursor, pitch_feature_vectors[j], train_set[i][1],  knn_db.pitch_train_set_name)

            summary_pitch_feature_vector = voice_m.get_summary_pitch_feature_vector(pitch_feature_vectors)
            for j in range(0, len(summary_pitch_feature_vector)):
                summary_pitch_knn_module.train(summary_pitch_feature_vector, train_set[i][1])
                knn_db.save_in_dbtable(db, cursor, summary_pitch_feature_vector, train_set[i][1], knn_db.summary_pitch_train_set_name)

            energy_feature_vectors = voice_m.get_energy_feature_vector(train_set[i][0])
            for j in range(0, len(energy_feature_vectors)):
                energy_knn_module.train(energy_feature_vectors[j], train_set[i][1])
                knn_db.save_in_dbtable(db, cursor, energy_feature_vectors[j], train_set[i][1], knn_db.energy_train_set_name)

    return db


def main_KNN(train_path_pattern, test_path_pattern, db_name):
    summary_table = {}
    for i in range(0, len(emotions)):
        summary_table[emotions[i]] = {"neutral": 0, "anger": 0, "boredom": 0, "disgust": 0, "fear": 0, "happiness": 0,
                                      "sadness": 0, "guessed": 0, "tested": 0, "trained": 0}
    frame_length = 512

    pitch_knn_module = KNN(emotions)
    summary_pitch_knn_module = KNN(emotions)
    energy_knn_module = KNN(emotions)
    voice_m = VoiceModule()

    db = train(train_path_pattern, voice_m, db_name, pitch_knn_module, summary_pitch_knn_module, energy_knn_module,
               summary_table, frame_length)

    input_set = build_file_set(test_path_pattern)

    for i in range(0, len(input_set)):
        print_progress_bar(i + 1, len(input_set), prefix='Computing progress:', suffix='Complete', length=50)

        pitch_feature_vectors = voice_m.get_pitch_feature_vector(input_set[i][0], frame_length)
        summary_pitch_feature_vector = []
        if len(pitch_feature_vectors) > 0:
            pitch_possible_emotions = pitch_knn_module.compute_emotion(pitch_feature_vectors, 15)

            summary_pitch_feature_vector.extend(voice_m.get_summary_pitch_feature_vector(pitch_feature_vectors))
            summary_pitch_possible_emotions = summary_pitch_knn_module.get_emotion(summary_pitch_feature_vector, 4)

        energy_feature_vectors = voice_m.get_energy_feature_vector(input_set[i][0])
        energy_possible_emotions = []
        if len(energy_feature_vectors) > 0:
            energy_possible_emotions = energy_knn_module.compute_emotion(energy_feature_vectors, 12)

        possible_emotions = pitch_possible_emotions
        possible_emotions.extend(summary_pitch_possible_emotions)
        possible_emotions.extend(energy_possible_emotions)

        computed_emotion = get_most_frequent_emotion(possible_emotions)

        if computed_emotion == "":
            computed_emotion = pitch_possible_emotions[0]

        summary_table[input_set[i][1]]["tested"] += 1
        if computed_emotion == input_set[i][1]:
            summary_table[input_set[i][1]]["guessed"] += 1

        summary_table[input_set[i][1]][computed_emotion] += 1

    print_summary(summary_table)


def print_summary(summary_table):
    print()
    for i in range(0, len(emotions)):
        print("emo: %s\t, anger: %d\t, boredom: %d\t,  happiness: %d\t, "
              "sadness: %d\t, guessed:%d\t, tested: %d\t, trained: %d\n"
              % (emotions[i], summary_table[emotions[i]]["anger"],
                 summary_table[emotions[i]]["boredom"], summary_table[emotions[i]]["happiness"],
                 summary_table[emotions[i]]["sadness"], summary_table[emotions[i]]["guessed"],
                 summary_table[emotions[i]]["tested"], summary_table[emotions[i]]["trained"]))


def convert_vector_to_number(vector):
    num = ""
    for i in range(len(vector)):
        if int(vector[i] / 10) == 0:
            num += "0" + str(vector[i])
        else:
            num += str(vector[i])

    return num


def prepare_set(train_path_pattern, voice_m, frame_length):
    train_set = build_file_set(train_path_pattern)
    training_observations = {"anger": [], "boredom": [], "happiness": [], "sadness": []}

    for i in range(0, len(train_set)):
        print_progress_bar(i + 1, len(train_set), prefix='Computing progress:', suffix='Complete', length=50)
        pitch_feature_vectors = voice_m.get_pitch_feature_vector(train_set[i][0], frame_length)
        grouped_pitch_feature_vectors = []
        for j in range(0, len(pitch_feature_vectors)):
            grouped_vector = voice_m.create_grouped_pitch_veatures(
                pitch_feature_vectors[j][1:5] + pitch_feature_vectors[-1])
            grouped_pitch_feature_vectors.append(convert_vector_to_number(grouped_vector))

        grouped_pitch_feature_vectors_len = len(grouped_pitch_feature_vectors)
        if grouped_pitch_feature_vectors_len > 6:
            training_observations[train_set[i][1]].append(
                grouped_pitch_feature_vectors[grouped_pitch_feature_vectors_len - 6:grouped_pitch_feature_vectors_len])
        elif grouped_pitch_feature_vectors_len == 6:
            training_observations[train_set[i][1]].append(grouped_pitch_feature_vectors)
        else:
            for k in range(0, 6 - len(grouped_pitch_feature_vectors)):
                grouped_pitch_feature_vectors.append("0" * 10)
            training_observations[train_set[i][1]].append(grouped_pitch_feature_vectors)

    return training_observations


def main_HMM(train_path_pattern, test_path_pattern):
    voice_m = VoiceModule()
    frame_length = 512

    num_of_states = 6
    trasition_ppb = [[0.2, 0.8],
                     [0.2, 0.8],
                     [0.2, 0.8],
                     [0.2, 0.8],
                     [0.2, 0.8],
                     [0.2, 0]]

    print("Creating possible obserwations")
    possible_observations = voice_m.create_possible_observations()
    possible_observations.append("0"*10)

    HMM_modules = {}
    print("Creating HMM")
    HMM_modules["anger"] = HMM(trasition_ppb, num_of_states, possible_observations)
    HMM_modules["boredom"] = HMM(trasition_ppb, num_of_states, possible_observations)
    HMM_modules["happiness"] = HMM(trasition_ppb, num_of_states, possible_observations)
    HMM_modules["sadness"] = HMM(trasition_ppb, num_of_states, possible_observations)

    summary_table = {}
    for i in range(0, len(emotions)):
        summary_table[emotions[i]] = {"anger": 0, "boredom": 0, "happiness": 0, "sadness": 0,
                                      "guessed": 0, "tested": 0, "trained": 0}

    print("Creating training observations")
    training_observations = prepare_set(train_path_pattern, voice_m, frame_length)

    print("Traning HMM")
    for key, value in HMM_modules.items():
        print("Training %s hmm module" %(key))
        HMM_modules[key].learn(training_observations[key], 0.001)

    print("Preparing test observations")
    test_observations = prepare_set(test_path_pattern, voice_m, frame_length)

    print("Testing")
    for tested_emo in test_observations:
        for observation in test_observations[tested_emo]:
            most_ppb_emotion = tested_emo
            max_ppb = 0
            for emotion, hmm_module in HMM_modules.items():
                value = hmm_module.evaluate(observation)
                if value > max_ppb:
                    max_ppb = value
                    most_ppb_emotion = emotion

            summary_table[tested_emo]["tested"] += 1
            summary_table[tested_emo][most_ppb_emotion] += 1
            if most_ppb_emotion == tested_emo:
                summary_table[tested_emo]["guessed"] += 1

    print_summary(summary_table)
    # print(summary_table)


if len(sys.argv) > 1 and sys.argv[1] == 'KNN':
    filterwarnings('ignore', category = pymysql.Warning)
    main_KNN('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', knn_db.DB_NAME)
else:
    main_HMM('Berlin_EmoDatabase/train/male/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav')
    # possible_observations = [0, 1, 2]
    # num_of_states = 3
    # trasition_ppb = [[0.2, 0.8],
    #                  [0.2, 0.8],
    #                  [0.2, 0]]
    #
    # HMM_test = HMM(trasition_ppb, num_of_states, possible_observations)
    #
    # training_set = [[0, 1, 2],
    #                 [2, 1, 0]]
    #
    # HMM_test.learn(training_set, 0.00001)
    # print(HMM_test.evaluate(training_set[0]))
    # print(HMM_test.evaluate([2, 1, 0]))
    # print(HMM_test.evaluate([0, 0, 2]))
    # print(HMM_test.evaluate([1, 2, 0]))

    # for i in HMM_test.get_parameters():
    #     print(i)
    # print(HMM_test.evaluate(training_set[0]))
    # print(HMM_test.evaluate([4, 3, 5]))
    # print(HMM_test.evaluate([5, 4, 3]))


# main('emodb/train/female/*/*.wav', 'emodb/test/female/*/*.wav', knn_db.DB_FEMALE_NAME)
# print()
# main('emodb/train/male/*/*.wav', 'emodb/test/male/*/*.wav', knn_db.DB_MALE_NAME)

# draw_energy_histogram()
# draw_freq_histogram()
# print_feature_vector()
