from helper_file import euclidean_distance, normalize_vector


class KNN:
    def __init__(self, states, train_set):
        self.states = states
        self.min_features, self.max_features = self.normalize(train_set)

        self.training_set = []
        for row in train_set:
            self.training_set.append({'training_vec': row[0], 'emotion': row[1]})

    def get_emotion(self, test_vector, num_of_nearest_neighbour):
        dist_table = []
        normalize_vector(test_vector, self.min_features, self.max_features)

        for train_vec in self.training_set:
            dist = euclidean_distance(train_vec['training_vec'], test_vector)
            dist_table.append([dist, train_vec['emotion']])

        dist_table.sort(key=lambda x: x[0])

        emotion_counter = {}
        for state in self.states:
            emotion_counter[state] = 0

        if num_of_nearest_neighbour > len(dist_table):
            num_of_nearest_neighbour = len(dist_table)

        for i in range(num_of_nearest_neighbour):
            emotion_counter[dist_table[i][1]] += 1

        max_num_of_occurrence = max(value for value in emotion_counter.values())

        possible_emotions = []
        for state, num_occurrence in emotion_counter.items():
            if num_occurrence == max_num_of_occurrence:
                possible_emotions.append(state)

        return possible_emotions

    def compute_emotion(self, obs_sequence, num_of_nearest_neighbour):
        if not isinstance(obs_sequence[0], list):
            return self.get_emotion(obs_sequence, num_of_nearest_neighbour)
        # if len(obs_sequence) == 1:
        #     return self.get_emotion(obs_sequence, num_of_nearest_neighbour)

        emotion_counter = {}
        for emotion in self.states:
            emotion_counter[emotion] = 0

        for observation in obs_sequence:
            emotions = self.get_emotion(observation, num_of_nearest_neighbour)
            for emotion in emotions:
                emotion_counter[emotion] += 1

        max_num_of_occurrence = max(value for value in emotion_counter.values())
        possible_emotions = []
        for emotion, num_occurrence in emotion_counter.items():
            if num_occurrence == max_num_of_occurrence:
                possible_emotions.append(emotion)

        return possible_emotions

    @staticmethod
    def normalize(feature_vector_set):
        """
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

