from math import pow, sqrt


class KNN:
    training_set = []

    def __init__(self, states, k):
        self.states = states
        self.k = k

    # Oblicza odległosć euklidesową pomiedzy dwoma wektorami
    def dist_eu(self, vec1, vec2):
        dist = 0
        for i in range(0, len(vec1)):
            dist += pow(vec1[i] - vec2[i], 2)

        return sqrt(dist)

    # Normalizuje wektor testowy wartościami z wektora treningowego
    def norm_test_vec(self, test_vec, train_min, train_max):
        norm_vect = []
        for i in range(0, len(test_vec)):
            norm_vect.append(((test_vec[i] - train_min) / (train_max - train_min)))

        return norm_vect

    # Zwraca najczęściej występujący stan
    def get_most_frequent_state(self, states_counter):
        max_state, max_state_counter = states_counter.popitem()
        for key, value in states_counter.items():
            if value > max_state_counter:
                max_state_counter = value
                max_state = key

        return max_state

    def train(self, training_vec, state):
        min_value = training_vec[0]
        max_value = training_vec[0]
        for i in range(1, len(training_vec)):
            if training_vec[i] < min_value:
                min_value = training_vec[i]

            if training_vec[i] > max_value:
                max_value = training_vec[i]

        norm_vec = []
        for i in range(0, len(training_vec)):
            norm_vec.append((training_vec[i] - min_value) / (max_value - min_value))

        self.training_set.append({'training_vec': training_vec, 'norm_vec': norm_vec, 'min': min_value, 'max': max_value,
                                  'state': state})

    def get_emotion(self, test_vec):
        dist_table = []

        for train_vec in self.training_set:
            norm_test_vec = []
            norm_test_vec.extend(self.norm_test_vec(test_vec, train_vec['min'], train_vec['max']))
            dist = self.dist_eu(train_vec['norm_vec'], norm_test_vec)
            dist_table.append([dist, train_vec['state']])

        dist_table.sort(key=lambda x: x[0])

        # for i in range(0, 9):
        #     print(dist_table[i][0], end=" ")
        #     print(dist_table[i][1])

        states_counter = {}
        for state in self.states:
            states_counter[state] = 0

        neightbour_num = self.k
        if self.k > len(dist_table):
            neightbour_num = len(dist_table)

        for i in range(0, neightbour_num):
            states_counter[dist_table[i][1]] += 1
        #
        # for key, value in states_counter.items():
        #     print(key, end=" : ")
        #     print(value)

        return self.get_most_frequent_state(states_counter)

    def compute_emotion(self, obs_sequence):
        states_counter = []
        for state in self.states:
            states_counter[state] = 0

        for observation in obs_sequence:
            states_counter[self.get_emotion(observation)] += 1

        return self.get_most_frequent_state(states_counter)