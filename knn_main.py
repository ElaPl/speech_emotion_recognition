from helper_file import create_summary_table, print_summary, build_file_set, print_progress_bar, \
    get_most_frequently_occurring
from voice_module import  get_feature_vectors
from KNN import KNN
import database_module as knn_db
from database_module import connect_to_database


knn_features = {
    "features": {
        "db_table_name": knn_db.knn_train_db_table,
        "nearest_neighbour": 17
    }
}

def knn_main(train_path_pattern, test_path_pattern, db_name, emotions):
    """Główna funckja knn. Dla każdej emocji tworzy model KNN i trenuje go wektorami obserwacji
    pobranymi z bazie danych db_name jeżeli istnieją, lub w przeciwnym wypadku obliczonymi z plików znajdujacych sie w katalogu
    train_path_pattern.
    Następnie testuje ich działanie wektorami obserwacji obliczonymi z plików znajdujących się w test_path_pattern

    :param train_path_pattern: Ścieżka do folderu zawierające pliki dźwiękowe, z których mają
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

    con, cursor = connect_to_database(db_name)
    if (cursor is not None) and knn_db.is_training_set_exists(cursor, knn_db.KNN_DB_TABLES):
        training_set = knn_get_training_feature_set_from_db(cursor)
    else:
        training_set = knn_get_training_feature_set_from_dir(train_path_pattern, emotions)
        if con is not None:
            knn_db.prepare_db_table(con, cursor, knn_db.KNN_DB_TABLES)
            for feature in knn_features:
                for i in range(len(training_set[feature])):
                    vector_to_save = list(training_set[feature][i][0])
                    vector_to_save.append(training_set[feature][i][1])
                    knn_db.save_in_dbtable(con, cursor, vector_to_save, knn_features[feature]["db_table_name"])

    if con:
        con.close()

    KNN_modules = {}
    for feature in knn_features:
        KNN_modules[feature] = KNN(training_set[feature])

    summary_table = create_summary_table(emotions)
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

        if emotion in emotions:
            feature_vectors = {}
            feature_vectors["pitch"], feature_vectors["energy"], feature_vectors["features"] = get_feature_vectors(file)

            possible_emotions = []
            for feature in knn_features:
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

def knn_get_training_feature_set_from_dir(path_pattern, emotions):
    """Funkcja dla każdego z dla każdego zbioru cech pobiera wektor cech oraz emocję jaką reprezentuje
    z bazy danych na którą wskazuje cursor

   :param path_pattern: kursor na bazę danych

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

        if emotion in emotions:
            pitch_feature_vectors, energy_feature_vectors, feature_vectors = get_feature_vectors(file)

            for vec in feature_vectors:
                features_set["features"].append([vec, emotion])

    return features_set


