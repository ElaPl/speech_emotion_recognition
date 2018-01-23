import numpy


class Markov_model:
    """Klasa implementująca algorytm Ukryte Modle Markova dla problemu rozpoznawania emocji z głosu.

    W tym modelu każda obserwacja tworzy jeden stan modelu. Z Każdeg stanu można przejść do każdego innego
    z pewnym pradopodobieństwem.

    Na podstawie danych z zbióru uczącego zawieracjącego obserwację (wektory cech), obliczonane jest prawdopodobieństwo
    przejścia pomiędzy stanami.


    Dla danej obserwacji X = [x_1, x_2,..., x_n]
    Prawdopodobieństwo wygenerowania obserwacji X w tym modelu :
    P(X|M) = initial_ppb[x_1] * iloczyn(transition_ppb[x_i-1][x_i])

    :param int states_num: liczba wszystkich możliwych obserwacji (stanów modelu)
    :param matrix[hidden_states_num][hidden_states_num] transition_ppb: tablica prawdopodobieństw przejść pomiedzy
    stanami. matrix[i][j] - prawdopodobieństwo przejśćia z stanu i do stanu j
    :param dict states: słownik zawierający dla każdej obserawacji i jej index w transition_ppb
    :param list[hidden_states_num] initial_ppb: lista prawdopodobieństw przejsć z stanu początkowego dostanu każdego
        z pozostałych stanów.

    """
    def __init__(self, observations):
        self.states = self.create_states(observations)
        self.transition_ppb = self.create_transition_ppb(self.states)
        self.states_num = len(observations)
        self.initial_ppb = self.create_initial_ppb(self.states_num)


    def create_initial_ppb(self, states_num):
        """Funkcja tworzy wektor prawdopodobieństw przejść ze stanu początkowe do każdego z ukrytych stanów

        :return list[states_num]
        """
        init_ppb = numpy.zeros(shape=states_num)
        for i in range(0, states_num):
            init_ppb[i] = 1/states_num

        return init_ppb

    def create_transition_ppb(self, states):
        """
         :param int states_num: liczba stanów
         :param matrix[state_num][2] given_transition_ppb: tablica prawdopodobieństw przejść pomiedzy kolejnymi stanami

         :return matrix[states_num][states_num]: macierz prawdopodobieństw przejść pomiędzy stanami
        """
        states_num = len(states)
        transition_ppb = numpy.zeros(shape=(states_num, states_num))
        for i in range(states_num):
            for j in range(states_num):
                transition_ppb[i][j] = 1 / states_num

        return transition_ppb

    def create_states(self, observations):
        """Funkcja tworzy słownik obserwacji. Każdą obserwacje zamienia na string i przypisuje unikatowy
        numer z przedziału [0, len(observations)-1].

        :param list observations: lista obserwacji (wektorów cech)
        :return słownik obserwacji
        """
        observation_dict = {}
        observation_id = 0
        if isinstance(observations[0], str):
            for obs in observations:
                observation_dict[obs] = observation_id
                observation_id += 1
        else:
            for obs in observations:
                observation_dict[str(obs)] = observation_id
                observation_id += 1

        return observation_dict


    def reestimate(self, observations, laplance_smoothing = 0.001):
        """Implementacja algorytmu do reestymacji parametrów modelu Makova.
        Jako estymator tablicy przejść wybrano liczbę przejść pomiedży stanami:

        N[i][j] - liczba przejść z stanu i do stanu j

        transition_ppb[i][j] = N[i][j] / suma(N[i][k] dla każdego stanu k)
        Do transition_ppb dodawana jest również wartość laplance_smoothing aby prawdopodobieństwo przejśćia pomiedzy stanami było
        zawsze nie zerowe.

        :param list observations: lista sekwencji obserwacji
        :param int observations_num: liczba sekwencji obserwacji
        :param int obs_seq_len: długość każdej z sekwencji obserwacji
        :param laplance_smoothing: minimalne pradopodobieństwo wyrzucenia obserwacji

        """

        transition_counter = numpy.zeros(shape=(self.states_num, self.states_num))
        init_counter = numpy.zeros(shape=self.states_num)

        for observation_seq in observations:
            obs_sequence_id = [self.states[observation_seq[i]] for i in range(0, len(observation_seq))]
            init_counter[obs_sequence_id[0]] += 1
            for i in range(len(obs_sequence_id) - 1):
                transition_counter[obs_sequence_id[i]][obs_sequence_id[i+1]] += 1

        sum_init = sum(init_counter[i] for i in range(self.states_num))
        #reeestimate self.initial_ppb
        for state_id in range(self.states_num):
            self.initial_ppb[state_id] = (init_counter[state_id] + laplance_smoothing) / \
                                         (sum_init + laplance_smoothing * self.states_num)

        # reestimate transition_ppb
        for state_from in range(self.states_num):
            all_transition_from = sum(transition_counter[state_from][state] for state in range(self.states_num))
            for state_to in range(self.states_num):
                self.transition_ppb[state_from][state_to] = \
                    (transition_counter[state_from][state_to] + laplance_smoothing) \
                    / (all_transition_from + (laplance_smoothing * self.states_num))


    def create_observation_seq_str(self, observation_seq):
        """
        Zamienia każdą obserwację z listy na string
        :param observation_seq: sekwencja obserwacji dowolnego typu
        :return: sekwencja obserwacji typu string
        """

        if not isinstance(observation_seq[0], str):
            return list(map(str, observation_seq))

        return observation_seq

    def train(self, training_observations_sequences, laplance_smoothing=0.001):
        """Funkcja trenuje model HMM za pomocą podanego zbioru uczącego

        :param list training_set: zbiór uczący postaci lista seqwencji obserwacji.
        :param float laplance_smoothing: minimalne prawdopodobieństwo wygenerowania obserwacji przez dany model

        Ponieważ obserwacje modelu są typu string, najpierw zamienia każdą obserwację na elementy typu string.
        Następnie powtarza algorytm Bauma-Welcha na zbiorze uczącym, określoną ilość razy, lub dopóki różnica
        prawdopodobieństw wygenerowania zbioru uczącego w starym modelu i nowym będzie mniejsze niż epsilon.
        """

        training_observation_seq_str = []

        for observation_seq in training_observations_sequences:
            training_observation_seq_str.append(self.create_observation_seq_str(observation_seq))

        self.reestimate(training_observation_seq_str, laplance_smoothing)

    def print_params(self):
        """Funkcja wypisuje parametry modelu HMM"""
        print("\nInitial ppb")
        print(self.initial_ppb)

        print("\nTransition")
        print(self.transition_ppb)

        print("\nstates_num")
        print(self.states_num)
        print("\nstates")
        print(self.states)
        print()

    def evaluate(self, observation_seq):
        """
        :param observation_seq: sekwencja obserwacji
        :return int: prawdopodobieństwo wygenerowania podanej obserwacji w tym modelu
        """
        observations_seq_len = len(observation_seq)

        observation_seq_str = self.create_observation_seq_str(observation_seq)
        obs_sequence_id = [self.states[observation_seq_str[i]] for i in range(0, observations_seq_len)]

        likehood = self.initial_ppb[obs_sequence_id[0]]

        for i in range(0, observations_seq_len - 1):
            likehood *= self.transition_ppb[obs_sequence_id[i]][obs_sequence_id[i+1]]

        return likehood

