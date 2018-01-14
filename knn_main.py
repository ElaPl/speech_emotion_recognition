from helper_file import create_summary_table, print_summary, connect_to_database, build_file_set, print_progress_bar, \
    get_most_frequently_occurring
from voice_module import  get_feature_vectors, get_summary_pitch_feature_vector
from KNN import KNN
import knn_database as knn_db

knn_features = {
    "pitch": {
        "db_table_name": knn_db.pitch_knn_train_db_table,
        "nearest_neighbour": 15
    },

    "energy": {
        "db_table_name": knn_db.energy_knn_train_db_table,
        "nearest_neighbour": 12
    },

    "pitch_summary": {
        "db_table_name": knn_db.summary_pitch_knn_train_db_table,
        "nearest_neighbour": 4
    }
}

def knn_main(train_path_pattern, test_path_pattern, db_name, db_password, emotions):
    """Główna funckja knn. Dla każdej emocji tworzy model KNN i trenuje go wektorami obserwacji
    pobranymi z bazie danych db_name jeżeli istnieją, lub w przeciwnym wypadku obliczonymi z plików znajdujacych sie w katalogu
    train_path_pattern.
    Następnie testuje ich działanie wektorami obserwacji obliczonymi z plików znajdujących się w test_path_pattern

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
    summary_table = create_summary_table(emotions)

    db, cursor = connect_to_database(db_name, db_password)
    if (cursor is not None) and knn_db.is_training_set_exists(cursor, knn_db.KNN_DB_TABLES):
        training_set = knn_get_training_feature_set_from_db(cursor)
    else:
        training_set = knn_get_training_feature_set_from_dir(train_path_pattern)
        if db is not None:

            knn_db.prepare_db_table(db, cursor, knn_db.KNN_DB_TABLES)
            for feature in knn_features:
                for i in range(len(training_set[feature])):
                    vector_to_save = list(training_set[feature][i][0])
                    vector_to_save.append(training_set[feature][i][1])
                    knn_db.save_in_dbtable(db, cursor, vector_to_save, knn_features[feature]["db_table_name"])

    for emotion in emotions:
        for vec in training_set["pitch"]:
            if vec[1] == emotion:
                summary_table[emotion]["trained"] += 1

    KNN_modules = {}
    for feature in knn_features:
        KNN_modules[feature] = KNN(emotions, training_set[feature])

    knn_compute_emotions(test_path_pattern, KNN_modules, summary_table, emotions)

    print_summary(summary_table, emotions)


def knn_compute_emotions(path_pattern, KNN_modules, summary_table, emotions):
    """Funckja dla każdego pliku z path_pattern, pobiera wektory obserwacji, a nastepnie testuje nimi
    każdy z modelów KNN w celu odganięcia najbardziej prawdopodobnej emocji jaką reprezentuje plik

    :param path_pattern: ściażka do katalogu z plikami z których należy wygenerować wektory cech
    :type path_pattern: basestring
    :param KNN_modules: Zbiór wytrenowanych obiektów KNN, dla każdej emocji jednej obiekt obiekt KNN
    :type KNN_modules: dictionary
    :param summary_table: Pomocnicza tablica do zapisywania wyników testów
    :type summary_table: dictionary
    :param summary_table: Lista emocji do przetestowania
    :type summary_table: list

    """

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
        for feature in knn_features.keys():
            possible_emotions.extend(KNN_modules[feature].compute_emotion(feature_vectors[feature],
                                     knn_features[feature]["nearest_neighbour"]))

        computed_emotion = get_most_frequently_occurring(possible_emotions)

        summary_table[emotion]["tested"] += 1
        if computed_emotion == emotion:
            summary_table[emotion]["guessed"] += 1
        else:
            summary_table[emotion][computed_emotion] += 1

    return summary_table

def knn_get_training_feature_set_from_db(cursor):
    """Funkcja dla każdego z dla każdego zbioru cech tworzy wektor cech z plików w katalogu "path_pattern"

    :param path_pattern: ściażka do katalogu z plikami z których należy wygenerować wektory cech
    :type path_pattern: basestring

    :return: Dla każdego zbioru cech lista [wektor cech , emocja]
    :rtype: dicttionary
    """
    features_set = {}
    for feature in knn_features.keys():
        features_set[feature] = []

    for feature in knn_features.keys():
            for row in knn_db.select_all_from_db(cursor, knn_features[feature]["db_table_name"]):
                features_set[feature].append([list(row[0: -1]), row[-1]])

    return features_set

def knn_get_training_feature_set_from_dir(path_pattern):
    """Funkcja dla każdego z dla każdego zbioru cech pobiera wektor cech oraz emocję jaką reprezentuje
    z bazy danych na którą wskazuje cursor

       :param: path_pattern: kursor na bazę danych

       :return: Dla każdego zbioru cech lista [wektor cech , emocja]
       :rtype: dictionary
       """
    features_set = {}
    for feature in knn_features.keys():
        features_set[feature] = []

    test_set = build_file_set(path_pattern)
    num_files = len(test_set)
    for i in range(num_files):
        print_progress_bar(i + 1, num_files, prefix='Training progress:', suffix='Complete', length=50)
        file = test_set[i][0]
        emotion = test_set[i][1]

        pitch_feature_vectors, energy_feature_vectors = get_feature_vectors(file)
        summary_pitch_feature_vector = get_summary_pitch_feature_vector(pitch_feature_vectors)

        for vec in pitch_feature_vectors:
            features_set["pitch"].append([vec, emotion])

        for vec in energy_feature_vectors:
            features_set["energy"].append([vec, emotion])

        features_set["pitch_summary"].append([summary_pitch_feature_vector, emotion])

    return features_set
