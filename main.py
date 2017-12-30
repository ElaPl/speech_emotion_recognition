from voice_module import VoiceModule
from KNN import KNN
import os
import glob

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
    train_set = build_file_set('Berlin_EmoDatabase_tmp/wav/a*/*/*.wav')

    voice_m = VoiceModule(512)
    knn_module = KNN(emotions, 15)

    for i in range(0, len(train_set)):
        print_progress_bar(i + 1, len(train_set), prefix='Training progress:', suffix='Complete', length=50)
        feature_vector = voice_m.get_feature_vector(train_set[i][0])
        for j in range(0, len(feature_vector)):
            # print(feature_vector[j])
            knn_module.train(feature_vector[j], train_set[i][1])

    input_set = build_file_set('Berlin_EmoDatabase_tmp/wav/b*/*/*.wav')

    summary_table = {}
    for i in range(0, len(emotions)):
        summary_table[emotions[i]] = {"all": 0, "guessed": 0}

    for i in range(0, len(input_set)):
        print_progress_bar(i + 1, len(input_set), prefix='Computing progress:', suffix='Complete', length=50)
        feature_vector = voice_m.get_feature_vector(input_set[i][0])
        if len(feature_vector) > 0:
            summary_table[input_set[i][1]]["all"] += 1
            computed_emotion = knn_module.compute_emotion(feature_vector)
            if computed_emotion == input_set[i][1]:
                summary_table[input_set[i][1]]["guessed"] += 1
    print()
    for i in range(0, len(emotions)):
        print("emo: %s\t, all: %d \t guessed: %d\n" %(emotions[i], summary_table[emotions[i]]["all"], summary_table[emotions[i]]["guessed"]))


main()
