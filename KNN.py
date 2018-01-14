from helper_file import euclidean_distance, normalize_vector


class KNN:
    """Ta klasa implementuje algorytm K najbliższych sąsiadów dla problemu znajodwania najbliższych sąsiadów

    Dany jest zbiór uczący zawieracjący obserwację (wektory cech), z któżych każda ma przypisaną emocję jaką dany wektor
    reprezentuje. Zbiór uczący zostaje znormalizowany a zmienne użyte do normalizacje zapisane jako parametry obiektu.

    Dany jest zbiór obserwacji C = (c_1, c_2 ... c_k}. Celem jest na podstawie informacji z zbioru uczącego
     przewidzenie jaką emocję reprezentuje dany zbiór obserwacji.

    S = [] - zbiór stanów wynikowych

    Algorytm predycji:
    Dla każdej obserwacji c_i :
        * c_i zostaje znormalizowane wartościami którymi znormalizowany został zbiór uczący.
        * Obliczana jest odległość euklidesowa pomiedczy c_i a każdym wektorem z zbioru uczącego
        * Z zbioru uczącego wybierane jest k wektorów, których odległość do c_i jest najmniejsza.
        * Sumowane są stany które reprezentują zbiór k wektorów.
        * Stany które wystąpiły najczęściej dodawane są do S
    Stany które wystąpiły najczęściej w S są zwracane jako możliwe stany reprezentujace dany zbiór obserwacji
    """
    def __init__(self, train_set):
        """Konstruktor klasy.
        Normalizuje i zapisuje zbiór uczący

        :param list train_set: zbiór uczący dany model KNN -> lista wektorów postaci
            [wektor_cech, emocja jaką reprezentuje]"""
        self.min_features, self.max_features = self.normalize(train_set)

        self.training_set = []
        for row in train_set:
            self.training_set.append({'training_vec': row[0], 'emotion': row[1]})

    def get_emotion(self, test_vector, num_of_nearest_neighbour):
        """Funkcja porównuje podany wektor emocji z każdym z zbioru trenującego i wybiera k najbliższych.

        :param vector test_vector: wektor, którego stan należy odgadnąć
        :param int num_num_of_nearest_neighbour: liczba najbliższych sąsiadów, z których należy wziąć stan do porównania.

        :return lista stanów których wektory pojawiły sie najczęściej w grupie k najbliższych wektorów.
        """
        dist_table = []
        normalize_vector(test_vector, self.min_features, self.max_features)

        for train_vec in self.training_set:
            dist = euclidean_distance(train_vec['training_vec'], test_vector)
            dist_table.append([dist, train_vec['emotion']])

        dist_table.sort(key=lambda x: x[0])

        if num_of_nearest_neighbour > len(dist_table):
            num_of_nearest_neighbour = len(dist_table)

        emotion_counter = {}
        for i in range(num_of_nearest_neighbour):
            if dist_table[i][1] in emotion_counter:
                emotion_counter[dist_table[i][1]] += 1
            else:
                emotion_counter[dist_table[i][1]] = 1

        max_num_of_occurrence = max(value for value in emotion_counter.values())
        return [key for key in emotion_counter if emotion_counter[key] == max_num_of_occurrence]

    def compute_emotion(self, obs_sequence, num_of_nearest_neighbour):
        """Funkcja dla każdego wektora z zbioru obserwacji, sumuje stany jakie reprezentują.

        :param list obs_sequence: lista obserwacji (wektorów) reprezentujących wypowiedź, której stan emocjonalny trzeba
            rozpoznać
        :param int num_num_of_nearest_neighbour: liczba najbliższych sąsiadów.

        :return stany najczęściej występujące w podanej sekwencji obserwacji.
        """

        if not isinstance(obs_sequence[0], list):
            return self.get_emotion(obs_sequence, num_of_nearest_neighbour)

        emotion_counter = {}
        for observation in obs_sequence:
            emotions = self.get_emotion(observation, num_of_nearest_neighbour)
            for emotion in emotions:
                if emotion in emotion_counter:
                    emotion_counter[emotion] += 1
                else:
                    emotion_counter[emotion] = 0

        max_num_of_occurrence = max(value for value in emotion_counter.values())
        return [key for key in emotion_counter if emotion_counter[key] == max_num_of_occurrence]

    @staticmethod
    def normalize(feature_vector_set):
        """ Normalizuje zbiór listę wektórów postaci [wektor_cecg, emocja]

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
