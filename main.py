from voice_module import VoiceModule
from KNN import KNN
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import sys


emotions = ["neutral", "anger", "boredom", "disgust", "fear", "happiness", "sadness"]


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


def main(knn_pitch, knn_energy):
    print("\n MAIN  %d %d \n" %(knn_pitch, knn_energy))
    summary_table = {}
    for i in range(0, len(emotions)):
        summary_table[emotions[i]] = {"neutral": 0, "anger": 0, "boredom": 0, "disgust": 0, "fear": 0, "happiness": 0,
                                      "sadness": 0, "guessed": 0, "all": 0, "trained": 0}

    train_set = build_file_set('Berlin_EmoDatabase/wav/a*/*/*.wav')

    voice_m = VoiceModule()
    pitch_knn_module = KNN(emotions, knn_pitch)
    energy_knn_module = KNN(emotions, knn_energy)

    frame_length = 512
    for i in range(0, len(train_set)):
        print_progress_bar(i + 1, len(train_set), prefix='Training progress:', suffix='Complete', length=50)

        pitch_feature_vectors = voice_m.get_pitch_feature_vector(train_set[i][0], frame_length)
        for j in range(0, len(pitch_feature_vectors)):
            pitch_knn_module.train(pitch_feature_vectors[j], train_set[i][1])

        sample_rate = voice_m.get_sample_rate(train_set[i][0])
        energy_feature_vectors = voice_m.get_energy_feature_vector(train_set[i][0], int(sample_rate/4))
        for j in range(0, len(energy_feature_vectors)):
            energy_knn_module.train(energy_feature_vectors[j], train_set[i][1])

        summary_table[train_set[i][1]]["trained"] += 1

    # input_set = build_file_set("Berlin_EmoDatabase/wav/b02/happiness/*.wav")
    input_set = build_file_set('Berlin_EmoDatabase/wav/b*/*/*.wav')

    for i in range(0, len(input_set)):
        print_progress_bar(i + 1, len(input_set), prefix='Computing progress:', suffix='Complete', length=50)

        pitch_feature_vectors = voice_m.get_pitch_feature_vector(input_set[i][0], frame_length)
        if len(pitch_feature_vectors) > 0:
            pitch_possible_emotions = pitch_knn_module.compute_emotion(pitch_feature_vectors)

        sample_rate = voice_m.get_sample_rate(input_set[i][0])
        energy_feature_vectors = voice_m.get_energy_feature_vector(input_set[i][0], int(sample_rate / 4))
        if len(energy_feature_vectors) > 0:
            energy_possible_emotions = energy_knn_module.compute_emotion(energy_feature_vectors)

        # print("\n" + input_set[i][0])
        # print("pitch_possible_emotions")
        # print(pitch_possible_emotions)
        # print("energy_possible_emotions")
        # print(energy_possible_emotions)

        possible_emotions = []
        for stateA in pitch_possible_emotions:
            if stateA in energy_possible_emotions:
                possible_emotions.append(stateA)

        summary_table[input_set[i][1]]["all"] += 1

        computed_emotion = ""
        if len(possible_emotions) == 0:
            computed_emotion = pitch_possible_emotions[0]
        else:
            computed_emotion = possible_emotions[0]

        if computed_emotion == input_set[i][1]:
            summary_table[input_set[i][1]]["guessed"] += 1

        summary_table[input_set[i][1]][computed_emotion] += 1

    file = open('test_result/test_' + str(knn_pitch) + '_' + str(knn_energy) + '.txt', 'w')
    for i in range(0, len(emotions)):
        file.write("emo: %s\t, neutral: %d\t, anger: %d\t, boredom: %d\t, disgust: %d\t, fear: %d\t, happiness: %d\t, "
                   "sadness: %d\t, guessed:%d\t, all: %d\t, trained: %d\n"
                   % (emotions[i], summary_table[emotions[i]]["neutral"], summary_table[emotions[i]]["anger"],
                      summary_table[emotions[i]]["boredom"], summary_table[emotions[i]]["disgust"],
                      summary_table[emotions[i]]["fear"], summary_table[emotions[i]]["happiness"],
                      summary_table[emotions[i]]["sadness"], summary_table[emotions[i]]["guessed"],
                      summary_table[emotions[i]]["all"], summary_table[emotions[i]]["trained"]))


def draw_freq_histogram():
    train_set = build_file_set('Berlin_EmoDatabase/wav/' + sys.argv[1] + '/*.wav')
    frame_size = 512
    voice_m = VoiceModule(frame_size)

    for i in range(0, len(train_set)):
        freq_vector = voice_m.get_freq_vector(train_set[i][0])
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
        feature_vector = voice_m.get_feature_vector(train_set[i][0])
        for j in range(0, len(feature_vector)):
            print(feature_vector[j], end=" ")
        print("\n")


def train(file):
    train_set = build_file_set(file)

    voice_m = VoiceModule(512)
    knn_module = KNN(emotions, 8)

    for i in range(0, len(train_set)):
        print_progress_bar(i + 1, len(train_set), prefix='Training progress:', suffix='Complete', length=50)
        feature_vector = voice_m.get_feature_vector(train_set[i][0])
        for j in range(0, len(feature_vector)):
            knn_module.train(feature_vector[j], train_set[i][1])

    return knn_module

def are_equal(vecA, vecB):
    if len(vecA) != len(vecB):
        print("Not equal")
        return False
    else:
        for i in range(0, len(vecA)):
            if vecA[i] != vecB[i]:
                print("Not THe same")
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
            or False == are_equal(knn_moduleA.training_set[i]["state"], knn_moduleB.training_set[i]["state"])):
            print("Not equal")
            break
        print("Equal")


def draw_energy_histogram():
    train_set = build_file_set('Berlin_EmoDatabase/wav/' + sys.argv[1] + '/*.wav')
    frame_size = 512
    voice_m = VoiceModule(frame_size)

    for i in range(0, len(train_set)):
        energy_vector = voice_m.get_energy_han_vector(train_set[i][0])

        bins = np.arange(0, len(energy_vector), 1)  # fixed bin size
        plt.plot(bins, energy_vector)
        plt.title('Energy histogram of ' + train_set[i][0])
        plt.xlabel('time')
        plt.ylabel('Energy')

        plt.show()


# draw_energy_histogram()
# draw_freq_histogram()
# print_feature_vector()
# compare_feature_vectors("/home/ela/Documents/inz_proj/speech_emotion_recognition/Berlin_EmoDatabase/wav/a02/anger/*.wav")
# compare_feature_vectors("/home/ela/Documents/inz_proj/speech_emotion_recognition/Berlin_EmoDatabase/wav/a02/fear/08a02Ab.wav")
# compare_feature_vectors("/home/ela/Documents/inz_proj/speech_emotion_recognition/Berlin_EmoDatabase/wav/a02/happiness/13a02Fa.wav")
# compare_feature_vectors("/home/ela/Documents/inz_proj/speech_emotion_recognition/Berlin_EmoDatabase/wav/a02/anger/03a02Wb.wav")

for i in range(5, 15):
    for j in range(4, 15):
        main(i, j)

