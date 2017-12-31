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


def main():
    summary_table = {}
    for i in range(0, len(emotions)):
        summary_table[emotions[i]] = {"neutral": 0, "anger": 0, "boredom": 0, "disgust": 0, "fear": 0, "happiness": 0,
                                      "sadness": 0, "guessed": 0, "all": 0, "trained": 0}

    train_set = build_file_set('Berlin_EmoDatabase/wav/a*/*/*.wav')

    voice_m = VoiceModule(512)
    knn_module = KNN(emotions, 8)

    for i in range(0, len(train_set)):
        print_progress_bar(i + 1, len(train_set), prefix='Training progress:', suffix='Complete', length=50)
        feature_vector = voice_m.get_feature_vector(train_set[i][0])
        summary_table[train_set[i][1]]["trained"] += 1
        for j in range(0, len(feature_vector)):
            knn_module.train(feature_vector[j], train_set[i][1])

    input_set = build_file_set('Berlin_EmoDatabase/wav/b*/*/*.wav')

    # for i in range(0, len(emotions)):
    #     summary_table[emotions[i]] = {"all": 0, "guessed": 0}

    for i in range(0, len(input_set)):
        print_progress_bar(i + 1, len(input_set), prefix='Computing progress:', suffix='Complete', length=50)
        feature_vector = voice_m.get_feature_vector(input_set[i][0])
        if len(feature_vector) > 0:
            computed_emotion = knn_module.compute_emotion(feature_vector)
            summary_table[input_set[i][1]]["all"] += 1
            summary_table[input_set[i][1]][computed_emotion] += 1
            if computed_emotion == input_set[i][1]:
                summary_table[input_set[i][1]]["guessed"] += 1
    print()
    for i in range(0, len(emotions)):
        print("emo: %s\t, neutral: %d\t, anger: %d\t, boredom: %d\t, disgust: %d\t, fear: %d\t, happiness: %d\t, "
              "sadness: %d\t, guessed:%d\t, all: %d\t, trained: %d\n"
              %(emotions[i], summary_table[emotions[i]]["neutral"], summary_table[emotions[i]]["anger"],
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

# draw_freq_histogram()
# print_feature_vector()
main()
