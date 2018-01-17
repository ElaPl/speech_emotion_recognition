from math import sqrt, pow
import glob
import os
import operator
import pymysql
from matplotlib import pyplot as plt


# If True than debug print will be displayed
debug = True

def print_debug(text):
    if debug is True:
        print(text)


# Oblicza odległosć euklidesową pomiedzy dwoma wektorami
def euclidean_distance(vec1, vec2):
    """ Funckja oblicza dystans euklidesowy pomiędzy wektorami
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


def normalize_vector(feature_vector, min_features, max_features):
    """ Normalizuje wektor testowy wartościami podanymi jako argumenty
    :param feature_vector: wektor cech do znormalizowania
    :param min_features: wektor najmniejszych wartości z każdej cechy, którymi należy znormalizować podany wektor
    :param max_features: wektor największych wartości z każdej cechy, którymi należy znormalizować podany wektor
    """
    for feature_id in range(len(feature_vector)):
        if (max_features[feature_id] != min_features[feature_id]):
            feature_vector[feature_id] = (feature_vector[feature_id] - min_features[feature_id]) / (max_features[feature_id] - min_features[feature_id])


def build_file_set(path_pattern):
    """ Funkcja tworzy listę plików znajdujących sie w katalogu path_pattern z rozszerzeniem wav.
        dla pliku ../anger/file.wav, tworzy parę [../anger/file.wav, anger]
    :param pattern: Ścieżka do pliku z którego maja być wyodrębione pliki wav
    :type pattern: basestring
    :return: list
    """
    train_set = []
    for path_and_file in glob.iglob(path_pattern, recursive=True):
        if path_and_file.endswith('.wav'):
            path, filename = os.path.split(path_and_file)
            emotion = os.path.basename(path)
            train_set.append([path_and_file, emotion])
    return train_set


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """ FUnckja rysuje pasek postępu
    :param iteration: Obecna iteracja
    :type iteration: int
    :param total: Liczba wszystkich operacji
    :type total: int
    :param prefix: Tekst na początku paska postępu
    :type prefix: str
    :param suffix: Tekst na końcu paska postępu
    :type suffix: str
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()

def get_most_frequently_occurring(emotions_set):
    """ Zwraca najczęściej występujacy element w liście
        :param emotions_set: lista elementów
        :type emotions_set: list
        :return element który występuje najczęściej
        """
    emotion_counter = {}

    for emotion in emotions_set:
        if emotion in emotion_counter:
            emotion_counter[emotion] += 1
        else:
            emotion_counter[emotion] = 1

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
    """Funckja łączy się z bazą danych jako root
    :param db_name nazwa bazy danych
    :type str
    :param db_password hasło do bazy danych
    :type str
    :return
        * db, cursor - jeżeli połączenie zostało nawiązane
        * None, None - w przeciwnym przypadku
    """
    try:
        db = pymysql.connect(host="localhost", user="root", passwd=db_password, db=db_name)
        cursor = db.cursor()
        return db, cursor
    except pymysql.Error as e:
        print_debug("Can't connect to db")
        return None, None


def normalize(feature_vector_set):
    """ Normalizuje zbiór listę wektórów postaci [data, target], nie naruszając ich kolejności

    :param: feature_vector_set: Zbiór wektorów cech do znormalizowania
    :return: * wektor najmniejszych wartości z każdej cechy
             * wektor największych wartości z każdej cechy
    """
    min_features_vector = []
    max_features_vector = []

    feature_vec_len = len(feature_vector_set[0][0])

    for feature_id in range(len(feature_vector_set[0][0])):
        min_features_vector.append(
            min(feature_vector_set[i][0][feature_id] for i in range(0, len(feature_vector_set))))
        max_features_vector.append(
            max(feature_vector_set[i][0][feature_id] for i in range(0, len(feature_vector_set))))

    for i in range(len(feature_vector_set)):
        for feature_id in range(feature_vec_len):
            if max_features_vector[feature_id] != min_features_vector[feature_id]:
                feature_vector_set[i][0][feature_id] = (feature_vector_set[i][0][feature_id] - min_features_vector[
                    feature_id]) / (max_features_vector[feature_id] - min_features_vector[feature_id])

    return min_features_vector, max_features_vector

