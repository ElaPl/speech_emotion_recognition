from helper_file import create_summary_table, connect_to_database, build_file_set, print_progress_bar, \
    get_most_frequently_occurring, euclidean_distance, print_summary, print_debug

from HMM import HMM
import knn_database as knn_db
from voice_module import get_feature_vectors
from sklearn.cluster import KMeans
from helper_file import normalize_vector


hmm_features = {
    "pitch": {
        "db_table_name": knn_db.pitch_hmm_train_db_table,
    },

    "energy": {
        "db_table_name": knn_db.energy_hmm_train_db_table,
    }
}


def hmm_main(train_path_pattern, test_path_pattern, db_name, db_password, emotions):
    """Główna funkcja hmm. Dla każdego zestawu cech i kazdej emocji tworzy model HMM i trenuje go wektorami obserwacji
    pobranymi z bazie danych db_name jeżeli istnieją, lub w przeciwnym wypadku obliczonymi z plików znajdujacych sie w katalogu
    train_path_pattern.

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

    summary_table = create_summary_table(emotions)

    possible_observations, min_max_features = hmm_get_all_possible_observations(train_path_pattern, db_name,
                                                                                db_password)
    num_of_states = 6
    trasition_ppb = [[0.2, 0.8],
                     [0.2, 0.8],
                     [0.2, 0.8],
                     [0.2, 0.8],
                     [0.2, 0.8],
                     [0.2, 0]]

    HMM_modules = {}
    for f in hmm_features.keys():
        HMM_modules[f] = {}

    for feature in hmm_features.keys():
        for emotion in emotions:
            HMM_modules[feature][emotion] = HMM(trasition_ppb, num_of_states, possible_observations[feature])

    train_set = hmm_get_train_set(train_path_pattern, min_max_features, possible_observations, emotions, summary_table)

    num_of_hmm_models = len(hmm_features) * len(emotions)
    counter = 1
    for feature in hmm_features.keys():
        for emotion in emotions:
            print_progress_bar(counter, num_of_hmm_models, prefix='Training progress:', suffix='Complete', length=50)
            if train_set[feature][emotion]:
                HMM_modules[feature][emotion].learn(train_set[feature][emotion], 0.0001)
            counter += 1

    file_set = build_file_set(test_path_pattern)
    num_files = len(file_set)
    for i in range(num_files):
        print_progress_bar(i + 1, num_files, prefix='Testing progress:', suffix='Complete', length=50)
        file = file_set[i][0]
        tested_emotion = file_set[i][1]
        possible_emotions = []

        obs_vec = hmm_get_observations_vectors(file, min_max_features, possible_observations)
        for feature in hmm_features.keys():
            for obs in obs_vec[feature]:
                max_ppb = 0
                for emotion, hmm_module in HMM_modules[feature].items():
                    ppb = hmm_module.evaluate(obs)
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


def hmm_get_train_set(path_pattern, min_max_features, all_possible_observations, emotions, summary_table):
    """Funkcja dla każdego zestawu cech i każdej emocji tworzy zbiór sekwencji obserwacji do trenowania obiektów HMM.

    :param path_pattern: Ścieżka do folderu zawierające pliki dźwiękowe, z których mają być wygenerowane dane trenujące
    :type path_pattern: string
    :param min_max_features: Parametry potrzebne do normalizacji zbioru cech wektorów
    :type min_max_features: dictionary
    :param all_possible_observations: Wszystkie możliwe zbiory cech wektorów
    :type all_possible_observations: string
    :param emotions: Lista emocji
    :type emotions: list
    :return: Słownik, zwierający dla każdego zestawu cech i każdej emocji, listę sekwencji obserwacji (wektorów cech) z
        wszystkich plików z katalogu "path_pattern".
    :rtype: dictionary
    """

    train_set = {}
    for feature in hmm_features.keys():
        train_set[feature] = {}
        for emotion in emotions:
            train_set[feature][emotion] = []

    file_set = build_file_set(path_pattern)
    num_files = len(file_set)
    for i in range(num_files):
        print_progress_bar(i + 1, num_files, prefix='Preparing training set:', suffix='Complete', length=50)
        file = file_set[i][0]
        trained_emotion = file_set[i][1]
        summary_table[trained_emotion]["trained"] += 1
        obs_vec = hmm_get_observations_vectors(file, min_max_features, all_possible_observations)
        for feature in hmm_features.keys():
            for obs in obs_vec[feature]:
                train_set[feature][trained_emotion].append(obs)

    return train_set


def hmm_get_observations_vectors(file, min_max_features_vec, all_possible_observations):
    """Funkcja dla każdego zestawu cech tworzy zbiór wektorów cech wyliczonych z pliku "file".
    Każdy wektor normalizuje i przypisuje mu najbliższego sąsiada z wszystkich możliwych obserwacji.

    :param file: ścieżka do pliku z którego mają być pobrane zestawy cech
    :type file: basestring
    :param min_max_features: Parametry potrzebne do normalizacji zbioru cech wektorów
    :type min_max_features: dictionary
    :param all_possible_observations: Dla każdej cechy wszystkie możliwe w HMM zbiory cech wektorów
    :type all_possible_observations: vector[vector]

    :return: Słownik zawierający dla każdego zestawu cech listę sekwencję obserwacji wygenerowanych z pliku "file".
        Każda sekwencja obserwacji jest ok 1,5sek wypowiedzią i składa się z 6 wektorów cech,
        z których każdy reprezentuje 0,25s wypowiedzi.
    :rtype: dictionary
    """

    feature_vectors = {}
    feature_vectors["pitch"], feature_vectors["energy"] = get_feature_vectors(file)

    normalized_observations = {}
    for feature in hmm_features.keys():
        normalized_observations[feature] = []
        for vec in feature_vectors[feature]:
            normalize_vector(vec, min_max_features_vec[feature]["min"], min_max_features_vec[feature]["max"])
            normalized_observations[feature].append(hmm_get_nearest_neighbour(vec, all_possible_observations[feature]))

    observation_sequence_vec = {}
    for feature in hmm_features.keys():
        observation_sequence_vec[feature] = []
        #biorę 1,5 sek wektory obserwacji wypowiedzi
        for i in range(len(normalized_observations[feature]) - 2*6 - 2):
            observation = [normalized_observations[feature][j + i] for j in range(0, 12, 2)]
            observation_sequence_vec[feature].append(observation)

    return observation_sequence_vec


def hmm_get_nearest_neighbour(vec, data):
    """Funkcja porównuje dystans pomiędzy wektorem vec a każdym z wektórów "data".

    :param vec: wektor cech
    :type vec: vector
    :param data: wszystkie akceptowalne wektory cech
    :type data: list[feature_vector]

    :return: Wektor z "data" dla którego dystans do wektora vec jest najmniejszy.
    :rtype: vector
    """

    min_dist = euclidean_distance(vec, data[0])
    min_vector = data[0]
    for neighbour in data:
        dist = euclidean_distance(vec, neighbour)
        if dist < min_dist:
            min_dist = dist
            min_vector = neighbour

    return min_vector


def hmm_claster(feature_vector_set):
    """Funkcja normalizuje i za pomocą algorytmu K-means klasteryzuje podany zestaw wektorów cech.

    :param feature_vector_set: zbiór wektoróch cech
    :type feature_vector_set: list[vector]
    :return:
        * lista sklasteryzowancyh wektorów cech
        * wektor najmniejszych wartości z każdej cechy
        * wektor największych wartości z każdej cechy
    """

    min_feature_vec, max_feature_vec = hmm_normalize(feature_vector_set)

    kmeans = KMeans(n_clusters=1000).fit(feature_vector_set)

    observations = (kmeans.cluster_centers_).tolist()

    return observations, min_feature_vec, max_feature_vec


def hmm_normalize(feature_vector_set):
    """Funkcja normalizuje podany zbiór wektorów cech

    :param: feature_vector_set: Zbiór wektorów cech do znormalizowania
    :type: feature_vector_set: list[vector]
    :return:
             * wektor najmniejszych wartości każdej cechy
             * wektor największych wartości każdej cechy
    """
    min_features_vector = []
    max_features_vector = []

    feature_vector_set_len = len(feature_vector_set)
    feature_vec_len = len(feature_vector_set[0])

    for feature_id in range(feature_vec_len):
        min_features_vector.append(min(feature_vector_set[i][feature_id] for i in range(feature_vector_set_len)))
        max_features_vector.append(max(feature_vector_set[i][feature_id] for i in range(feature_vector_set_len)))

    for i in range(feature_vector_set_len):
        for feature_id in range(feature_vec_len):
            if max_features_vector[feature_id] != min_features_vector[feature_id]:
                feature_vector_set[i][feature_id] = \
                    (feature_vector_set[i][feature_id] - min_features_vector[feature_id]) / \
                    (max_features_vector[feature_id] - min_features_vector[feature_id])

    return min_features_vector, max_features_vector


def hmm_get_all_possible_observations(train_path_pattern, db_name, db_password):
    """Funkcja dla każdego zestawu cech tworzy zbiór wszystkich możliwych wektorów cech, wyliczony z plików w katalogu
    train_path_pattern, oraz klasteryzuje je w celu uzyskania ograniczonego zbioru obserwacji

    :param train_path_pattern: ścieżka do katalogu z plikami, z których mają być wyliczone obserwacje
    :type train_path_pattern: basestring
    :param db: baza danych do zapisu
    :param cursor: kursor na bazę danych
    :param summary_table: tablica podsumowująca wyniki
    :param emotions: lista emocji do wykrycia
    :type emotions: list

    :return:
        * dla każdego zestawu cech lista możliwych obserwacji
        * dla każdego zestawu cech wektor najmniejszych i największych wartości każdej z cech
    """

    db, cursor = connect_to_database(db_name, db_password)
    if (cursor is not None) and knn_db.is_training_set_exists(cursor, knn_db.HMM_DB_TABLES):
        feature_set = hmm_get_features_vectors_from_db(cursor)
    else:
        feature_set = hmm_get_features_vector_from_dir(train_path_pattern)
        if db is not None:
            knn_db.prepare_db_table(db, cursor, knn_db.HMM_DB_TABLES)
            for feature in hmm_features:
                for i in range(len(feature_set[feature])):
                    knn_db.save_in_dbtable(db, cursor, feature_set[feature][i], hmm_features[feature]["db_table_name"])

    min_max_features = {}
    possible_observations = {}

    for feature in hmm_features.keys():
        min_max_features[feature] = {}

    for feature in hmm_features.keys():
        possible_observations[feature], min_max_features[feature]["min"], min_max_features[feature]["max"] = \
            hmm_claster(feature_set[feature])

    return possible_observations, min_max_features

def hmm_get_features_vectors_from_db(cursor):
    """Funkcja dla każdego z dla każdego zbioru cech pobiera wektor cech z bazy danych, którą wskazuje cursor

    :param: path_pattern: kursor na bazę danych

    :return: Dla każdego zbioru cech, zbiór wektorów cech
    :rtype: dictionary
    """

    feature_set = {}
    for feature in hmm_features.keys():
        feature_set[feature] = []

    for feature in hmm_features.keys():
        for row in knn_db.select_all_from_db(cursor, hmm_features[feature]["db_table_name"]):
            feature_set[feature].extend([list(row)])

    return feature_set


def hmm_get_features_vector_from_dir(path_pattern):
    """Funkcja dla każdego z dla każdego zbioru cech tworzy wektor cech z plików w katalogu "path_pattern"

    :param path_pattern: ściażka do katalogu z plikami z których należy wygenerować wektory cech
    :type path_pattern: basestring

    :return: Dla każdego zbioru cech, lista wektorów cech
    :rtype: dictionary
    """

    features_set = {}
    for feature in hmm_features.keys():
        features_set[feature] = []

    test_set = build_file_set(path_pattern)
    num_files = len(test_set)
    for i in range(num_files):
        print_progress_bar(i + 1, num_files, prefix='Preparing observations:', suffix='Complete', length=50)
        file = test_set[i][0]

        feature_vectors = {}
        feature_vectors["pitch"], feature_vectors["energy"] = get_feature_vectors(file)

        for feature in hmm_features.keys():
            features_set[feature].extend(feature_vectors[feature])

    return features_set
