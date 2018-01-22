from helper_file import create_summary_table, connect_to_database, build_file_set, print_progress_bar, \
    get_most_frequently_occurring, euclidean_distance, print_summary, print_debug

from Markov_model import Markov_model
# from Markov_model_helper_file import get_all_possible_observations, hmm_features, get_nearest_neighbour
from voice_module import get_feature_vectors
from helper_file import normalize_vector
from hmm_main import *

def mm_main(train_path_pattern, test_path_pattern, db_name, db_password, emotions):
    """
    Główna funkcja algorytmu rozpoznawania emocji z głosu z urzyciem Modeli Markova. Dla każdego zestawu cech i
    kazdej emocji tworzy model HMM i trenuje go wektorami obserwacji pobranymi z bazie danych db_name jeżeli istnieją,
    lub w przeciwnym wypadku obliczonymi z plików znajdujacych sie w katalogu train_path_pattern.

    Następnie dla każdej wypowiedzi z katalogu test_path_pattern próbuje przewidzieć jaką emocję reprezentuje ta
    wypowiedź, w następujący sposób.
    Dla każdego pliku
        1) Dla każdego zestawu cech oblicz listę sekwencji obserwacji
        2) Dla każdej sekwencji obserwacji:
            2_1) Dla każdej emocji oblicza prawdopodobieństwo wygenerowania sekwencji obserwacji w modelu hmm
                reprezentującym emocje.
            2_2) Jako prawdopodobną emocję uznaje emocję reprezentującą przez model HMM, który zwrócił największe
                prawdopodobieńśtwo wygenerowania tej sekwencji obserwacji.
        3) Za emocję reprezentującą ten plik uznaje emocję, która wystąpiła największą ilosć razy

    :param train_train_path_pattern: Ścieżka do folderu zawierające pliki dźwiękowe, z których mają
        być wygenerowane dane trenujące
    :type train_path_pattern: basestring
    :param test_path_pattern: Ścieżka do folderu zawierające pliki dźwiękowe, z których mają
        być wygenerowane dane testujące
    :type test_path_pattern: basestring
    :param db_name: nazwa bazy danych
    :type db_name: basestring
    :param db_password: hasło do bazy danych
    :type db_password: basestring
    :param emotions: zbiór emocji do rozpoznania
    :type emotions: list
    """
    summary_table = create_summary_table(emotions);
    all_possible_observations, min_max_features = hmm_get_all_possible_observations(train_path_pattern, db_name,
                                                                                db_password, emotions)

    MM_modules = {}
    for f in hmm_features.keys():
        MM_modules[f] = {}

    for feature in hmm_features.keys():
        for emotion in emotions:
            MM_modules[feature][emotion] = Markov_model(all_possible_observations[feature])

    train_set = hmm_get_train_set(train_path_pattern, min_max_features, all_possible_observations, emotions,
                                 summary_table)

    num_of_hmm_models = len(hmm_features) * len(emotions)
    counter = 1

    for feature in hmm_features.keys():
        for emotion in emotions:
            print_progress_bar(counter, num_of_hmm_models, prefix='Training progress:', suffix='Complete', length=50)
            if train_set[feature][emotion]:
                MM_modules[feature][emotion].train(train_set[feature][emotion], 0.0001)
            counter += 1

    file_set = build_file_set(test_path_pattern)
    num_files = len(file_set)
    for i in range(num_files):
        print_progress_bar(i + 1, num_files, prefix='Testing progress:', suffix='Complete', length=50)
        file = file_set[i][0]
        tested_emotion = file_set[i][1]
        if tested_emotion in emotions:
            possible_emotions = []

            obs_vec = hmm_get_observations_vectors(file, min_max_features, all_possible_observations)
            for feature in hmm_features.keys():
                for obs in obs_vec[feature]:
                    max_ppb = 0
                    for emotion, mm_module in MM_modules[feature].items():
                        ppb = mm_module.evaluate(obs)
                        if ppb > max_ppb:
                            max_ppb = ppb
                            most_ppb_emotion = emotion

                        possible_emotions.append(most_ppb_emotion)

            computed_emotion = get_most_frequently_occurring(possible_emotions)
            summary_table[tested_emotion]["tested"] += 1
            summary_table[tested_emotion][computed_emotion] += 1

            if computed_emotion == tested_emotion:
                summary_table[tested_emotion]["guessed"] += 1

    print_summary(summary_table, emotions)
