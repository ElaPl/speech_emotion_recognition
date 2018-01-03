from math import pow, sqrt


class KNN:
    def __init__(self, states, k, training_set=[]):
        self.states = states
        self.k = k
        self.training_set = training_set

    # Oblicza odległosć euklidesową pomiedzy dwoma wektorami
    def dist_eu(self, vec1, vec2):
        dist = 0
        for i in range(0, len(vec1)):
            dist += pow(vec1[i] - vec2[i], 2)

        return sqrt(dist)

    def cosine_similarity(self, vecA, vecB):
        dotProduct = 0
        normA = 0
        normB = 0
        for i in range(0, len(vecA)):
            dotProduct += vecA[i] * vecB[i]
            normA += pow(vecA[i], 2)
            normB += pow(vecB[i], 2)
        normA = sqrt(normA)
        normB = sqrt(normB)

        return dotProduct/(normA*normB)

    # Normalizuje wektor testowy wartościami z wektora treningowego
    def norm_test_vec(self, test_vec, train_min, train_max):
        if train_min == train_max:
            return test_vec

        norm_vect = []
        for i in range(0, len(test_vec)):
            norm_vect.append(((test_vec[i] - train_min) / (train_max - train_min)))
        return norm_vect

    # Zwraca najczęściej występujący stan
    def get_most_frequent_state(self, states_counter):
        max_state_counter = states_counter[self.states[0]]
        max_state = self.states[0]
        for state in self.states:
            if states_counter[state] > max_state_counter:
                max_state_counter = states_counter[state]
                max_state = state

        return max_state

    def train(self, training_vec, state):
        min_value = training_vec[0]
        max_value = training_vec[0]
        for i in range(1, len(training_vec)):
            if training_vec[i] < min_value:
                min_value = training_vec[i]

            if training_vec[i] > max_value:
                max_value = training_vec[i]

        if max_value != min_value:
            norm_vec = []
            for i in range(0, len(training_vec)):
                norm_vec.append((training_vec[i] - min_value) / (max_value - min_value))

            self.training_set.append({'training_vec': training_vec, 'norm_vec': norm_vec, 'min': min_value,
                                      'max': max_value, 'state': state})
        else:
            self.training_set.append({'training_vec': training_vec, 'norm_vec': training_vec, 'min': min_value,
                                      'max': max_value, 'state': state})

    def get_emotion(self, test_vec):
        dist_table = []

        for train_vec in self.training_set:
            norm_test_vec = self.norm_test_vec(test_vec, train_vec['min'], train_vec['max'])
            dist = self.dist_eu(train_vec['norm_vec'], norm_test_vec)
            dist_table.append([dist, train_vec['state']])

        dist_table.sort(key=lambda x: x[0])

        states_counter = {}
        for state in self.states:
            states_counter[state] = 0

        neighbour_num = self.k
        if self.k > len(dist_table):
            neighbour_num = len(dist_table)

        for i in range(0, neighbour_num):
            states_counter[dist_table[i][1]] += 1

        max_state = self.get_most_frequent_state(states_counter)
        max_occurance = states_counter[max_state]
        possible_states = []
        for state, num_occurence in states_counter.items():
            if num_occurence == max_occurance:
                possible_states.append(state)

        return possible_states

    def compute_emotion(self, obs_sequence):
        states_counter = {}
        for state in self.states:
            states_counter[state] = 0

        for observation in obs_sequence:
            states = self.get_emotion(observation)
            for state in states:
                states_counter[state] += 1

        max_state = self.get_most_frequent_state(states_counter)
        max_occurance = states_counter[max_state]
        possible_states = []
        for state, num_occurence in states_counter.items():
            if num_occurence == max_occurance:
                possible_states.append(state)

        return possible_states
