import numpy
from math import log
import glob

debug = 0

class HMM:
    """Klasa implementująca algorytm Ukryte Modele Markowa dla problemu rozpoznawania emocji z głosu.

        Dany jest zbiór uczący zawieracjący obserwację (wektory cech), z któżych każda ma przypisaną emocję jaką dany wektor

        :param int hidden_states_num:  liczba ukrytych modeli Markowa
        :param int observations_num: liczba wszystkich możliwych obserwacji
        :param dict observation_dict: słownik zawierający dla każdej obserawacji jej index w tablicy emission_ppb
        :param matrix[hidden_states_num][observations_num] emission_ppb: tablica zawierająca dla każdego stanu S
            i każdej obserawcji O prawdopodobieństwo wygenerowanie B w stanie O
        :param matrix[hidden_states_num][hidden_states_num] transition_ppb: tablica prawdopodobieństw przejść pomiedzy
            stanami. matrix[i][j] - prawdopodobieństwo przejśćia z stanu i do stanu j
        :param list[hidden_states_num] initial_ppb: lista prawdopodobieństw przejsć z stanu początkowego dostanu każdego
            z ukrytych stanów.

    """
    def __init__(self, transition_ppb, states_num, observations):
        """Konstruktor klasy HMM.

        Tworzy tablicę przejść pomiędzy stanami, transition_ppb.
        Dla dowolnych 2 stanów s_i i s_j, prawdopodobnieństwo przejścia z stanu s_i do stanu s_j: p(s_i, s_j) jest równe:

            * transition_ppb[i][0] jeżeli i == j
            * transition_ppb[i][1] jeżeli i == j-1
            * 0 w przeciwnym przypadku

        Tworzy tablicę emisji: emission_ppb. Na początku dla każdego stanu S i każdej obserwacji O prawdopodobieństwo
        przejścia emisji O w stanie S jest równe.

        Tworzy tablicę: initial_ppb. Na początku dla każdego stanu S prawdopodobieństwo przejścia ze stanu początkowego
        do stanu S jest równe.

        :param matrix[states_num][2] transition_ppb: macierz prawdopodobieństw przejść pomiędzy stanami.

            * matrix[i][0] - zawiera pradopodobieńśwo przejśćia z stanu i do stanu i,
            * matrix[i][1] - zawiera pradopodobieńśwo przejśćia z stanu i do stanu (i+1)
        :param int states_num: liczba ukrytych stanow
        :param list observations: lista wszystkich możliwych obserwacji

        """
        self.hidden_states_num = states_num
        self.observations_num = len(observations)
        self.observation_dict = self.create_observation_dict(observations)

        # transition_ppb[i][j] - ppb of transition from state i to state j
        self.transition_ppb = self.create_transition_ppb(states_num, transition_ppb)

        # emission_ppb[j][o1] - ppb that at state i observation o1 will be produced
        self.emission_ppb = self.create_emission_ppb()

        # initial_ppb[i] - ppb of transition from initial state to state i
        self.initial_ppb = self.create_initial_ppb(states_num)

    def create_observation_dict(self, observations):
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

    def create_transition_ppb(self, states_num, given_transition_ppb):
        """
        :param int states_num: liczba stanów
        :param matrix[state_num][2] given_transition_ppb: tablica prawdopodobieństw przejść pomiedzy kolejnymi stanami

        :return matrix[states_num][states_num]: macierz prawdopodobieństw przejść pomiędzy stanami
        """
        transition_ppb = numpy.zeros(shape=(states_num, states_num))
        for state in range(0, states_num):
            transition_ppb[state][state] = given_transition_ppb[state][0]
            if state + 1 != states_num:
                transition_ppb[state][state + 1] = given_transition_ppb[state][1]

        return transition_ppb

    def create_emission_ppb(self):
        """Funckcja dla każdej obseracji i każdego stanu tworzy tablicę prawodopodobieństw wyrzucenia obserwacji w
        danym stanie

        :return matrix[state_num][observation_num]: macierz emisji prawdopodobieństw obserwacji
        """

        emission_ppb = numpy.zeros(shape=(self.hidden_states_num, self.observations_num))
        for state in range(0, self.hidden_states_num):
            for observation_id in self.observation_dict.values():
                emission_ppb[state][observation_id] = 1 / self.observations_num

        return emission_ppb

    def create_initial_ppb(self, states_num):
        """Funkcja tworzy wektor prawdopodobieństw przejść ze stanu początkowe do każdego z ukrytych stanów

        :return list[states_num] """
        initial_ppb = numpy.zeros(shape=states_num)
        for i in range(0, states_num):
            initial_ppb[i] = 1 / states_num

        return initial_ppb

    def get_parameters(self):
        """Funkcja zwraca parametry obiektu

        :return
            * transiton_ppb
            * emission_ppb
            * initial_ppb
            * observation_dict
        """
        return self.transition_ppb, self.emission_ppb, self.initial_ppb, self.observation_dict

    def forward_algorithm(self, observation_seq):
        """Implementacja algorytmu prefiksowego (forward algorithm).

        :param list observation_seq: sekwencja obserwacji

        :return matrix[hidden_states_num][len(observation_seq)], matrix[i][t]

        Opis algorytmu:
        Dane:
        Y = [y_0, y_1, ... , y_n] - observation_seq
        X = [x_1, x_1, ... , x_k] - ukryte stany markowa

        Cel:
        macierz alfa[[hidden_states_num][n]) taka, że:
            alfa[i][t] = P(Y[0] = y_0, Y[1] = y_1, ..., Y[t] = y_t | X_t = i) - prawdopodobienstwo wygenerowania
                y(0:t) przy założeniu, że w czasie t byliśmy w stanie i.

        Algorytm:

            * alfa[i][0] = initial_ppb[i] * emission_ppb[i][y_0]
            * alfa[j][t] = [\sum_{i=1}^{k} (alfa[i][t-1])*transition_ppb[i][j]] * emission_ppb[j][y_t]

        """
        observation_len = len(observation_seq)

        alfa = numpy.zeros(shape=(self.hidden_states_num, observation_len))
        for i in range(0, self.hidden_states_num):
            alfa[i][0] = self.initial_ppb[i] * self.emission_ppb[i][observation_seq[0]]

        for t in range(1, observation_len):
            for state_to in range(0, self.hidden_states_num):
                state_from = state_to
                ppb_to = alfa[state_from][t - 1] * self.transition_ppb[state_from][state_to]
                if state_to > 0:
                    state_from = state_to - 1
                    ppb_to += alfa[state_from][t - 1] * self.transition_ppb[state_from][state_to]

                alfa[state_to][t] = ppb_to * self.emission_ppb[state_to][observation_seq[t]]

        return alfa


    def backward_algorithm(self, ob_sequence):
        """Implementacja algorytmu sufiksowego (backward algorithm)

        :param list ob_sequence: sekwencja obserwacji

        :return matrix[hidden_states_num][len(observation_seq)], matrix[i][t]

            Opis algorytmu:
        Dane:
        Y = [y_0, y_1, ... , y_n] - observation_seq
        X = [x_1, x_1, ... , x_k] - ukryte stany markowa

            Cel:
        macierz beta[[hidden_states_num][n]) taka, że:
        beta[i][t] = P(Y[t+1] = y_t+1, Y[t+1] = y_t+1, ..., Y[n] = y_n | X_t = i) - prawdopodobienstwo
        zaobserwowania obserwacji y(t+1:n) zaczynając w punkcie i w czasie t.

            Algorytm:

        * beta[i][n] = 1
        * beta[i][t] = [\sum_{j=1}^{k} (emission_ppb[j][y_t+1] * beta[j][t+1] * transition_ppb[i][j]]

        """
        observation_len = len(ob_sequence)

        beta = numpy.zeros(shape=(self.hidden_states_num, observation_len))

        for state in range(0, self.hidden_states_num):
            beta[state][observation_len-1] = 1

        for t in range(observation_len - 2, -1, -1):
            for state_from in range(0, self.hidden_states_num):
                state_to = state_from
                beta[state_from][t] = beta[state_to][t + 1] * self.transition_ppb[state_from][state_to] \
                                      * self.emission_ppb[state_to][ob_sequence[t + 1]]
                if debug and t == 1 and state_from == 2:
                    print(beta[state_from][t])

                if state_from + 1 < self.hidden_states_num:
                    state_to = state_from + 1
                    beta[state_from][t] += beta[state_to][t + 1] * self.transition_ppb[state_from][state_to] * \
                                           self.emission_ppb[state_to][ob_sequence[t + 1]]
        return beta

    def baum_welch_algorithm(self, observations, observations_num, obs_seq_len, laplance_smoothing = 0.001):
        """Implementacja algorytmu bauma-welcha z użyciem równania Levinsona, dla N niezależnych sekwencji obserwacji.
        Algorytm służy to reestymacji parametrów ukrytych modeli Markowa

        :param list observations: lista sekwencji obserwacji
        :param int observations_num: liczba sekwencji obserwacji
        :param int obs_seq_len: długość każdej z sekwencji obserwacji
        :param laplance_smoothing: minimalne pradopodobieństwo wyrzucenia obserwacji

        """

        # Tablica prawdopodobieństw emisji,
        # gamma[i][t][k] = ppb że w czasie t w stanie i zostanie wyrzucona obserwacja obs_seq[k][t]
        gamma = numpy.zeros(shape=(self.hidden_states_num, obs_seq_len, observations_num))

        # Tablica prawdopodobienstw przejść pomiedzy stanami
        # trajectory_ppb[t][i][j][k] - prawdopodobieńswo, bycia w czasie t w stanie "i" i przejścia w czasie (t+1)
        # do stanu "j", produkując przy tym obserwację (observations[k][0] : observations[k][t+1])
        trajectory_ppb = numpy.zeros(
            shape=(obs_seq_len - 1, self.hidden_states_num, self.hidden_states_num, observations_num))

        observation_id = 0
        for obs_sequence in observations:
            obs_sequence_id = [self.observation_dict[obs_sequence[i]] for i in range(0, obs_seq_len)]

            alfa = self.forward_algorithm(obs_sequence_id)
            beta = self.backward_algorithm(obs_sequence_id)

            # P(O|M) - prawdopodobieństwo wygenerowania obserwacji przez model
            ppb_of_any_path = sum(alfa[i][obs_seq_len - 1] for i in range(0, self.hidden_states_num))

            # Oblicz tablcę gamma, opis na początku funcji
            for state in range(0, self.hidden_states_num):
                for t in range(0, obs_seq_len):
                    gamma[state][t][observation_id] += (alfa[state][t] * beta[state][t]) / ppb_of_any_path

            # Oblicz trajectory_ppb, opis na początku funkcji
            for t in range(0, obs_seq_len - 1):
                # prawdopodobieństwo przejśćia w czasie t z dowolnego stanu i do dowolnego stanu j
                for state_from in range(0, self.hidden_states_num):
                    if state_from + 1 == self.hidden_states_num:
                        trajectory_ppb[t][state_from][state_from][observation_id] \
                            = (alfa[state_from][t] * self.transition_ppb[state_from][state_from] *
                               self.emission_ppb[state_from][obs_sequence_id[t + 1]] * beta[state_from][t + 1]) / ppb_of_any_path
                    else:
                        for state_to in [state_from, state_from + 1]:
                            trajectory_ppb[t][state_from][state_to][observation_id] \
                                = (alfa[state_from][t] * self.transition_ppb[state_from][state_to] *
                                   self.emission_ppb[state_to][obs_sequence_id[t + 1]] * beta[state_to][t + 1]) / ppb_of_any_path
            observation_id += 1

        # Reestymacja initial_ppb
        for state in range(0, self.hidden_states_num):
            exp = sum(gamma[state][0][k] for k in range(observations_num))
            self.initial_ppb[state] = (exp + laplance_smoothing) / \
                                      (1/observations_num + self.hidden_states_num * laplance_smoothing)

        # gamma_sum[i] - czestość pobytów w stanie "i" w czasie 0, T-2
        gamma_sum = numpy.zeros(shape=self.hidden_states_num)
        for state in range(0, self.hidden_states_num):
            gamma_sum[state] = sum(sum(gamma[state][t][k] for t in range(0, obs_seq_len - 1))
                                   for k in range(0, observations_num))

        # Reestymacja transition_ppb
        for state_from in range(0, self.hidden_states_num):
            if state_from + 1 != self.hidden_states_num:
                state_to = state_from + 1

                # Czestośc przejsć z stanu i do stanu j
                exp_num = sum(sum(trajectory_ppb[t][state_from][state_to][k] for t in range(0, obs_seq_len - 1))
                              for k in range(0, observations_num))

                self.transition_ppb[state_from][state_to] = (exp_num + laplance_smoothing) / \
                                                            (gamma_sum[state_from] + 2 * laplance_smoothing)
                self.transition_ppb[state_from][state_from] = 1 - self.transition_ppb[state_from][state_to]

            else:
                # Czestośc przejsć z stanu i do stanu i
                exp_num = sum(sum(trajectory_ppb[t][state_from][state_from][k] for t in range(0, obs_seq_len - 1))
                              for k in range(0, observations_num))

                self.transition_ppb[state_from][state_from] = (exp_num + laplance_smoothing) / \
                                                              (gamma_sum[state_from] + 2 * laplance_smoothing)

        # gamma_sumB[i] - czestość pobytów w stanie "i" w czasie 0, T-1
        gamma_sumB = numpy.zeros(shape=self.hidden_states_num)
        for state in range(0, self.hidden_states_num):
            gamma_sumB[state] = sum(sum(gamma[state][t][k] for t in range(0, obs_seq_len))
                                    for k in range(0, observations_num))

        # Reestymacja emission_ppb
        for state in range(0, self.hidden_states_num):
            for observation, observation_id in self.observation_dict.items():
                # val - expected number of times that we were in state "state" and saw symbol "observation"
                val = 0.0
                for k in range(0, observations_num):
                    for t in range(0, obs_seq_len):
                        if observations[k][t] == observation:
                            val += gamma[state][t][k]

                self.emission_ppb[state][observation_id] = \
                    (val + laplance_smoothing) / (gamma_sumB[state] + self.observations_num * laplance_smoothing)

    def print_params(self):
        """Funkcja wypisuje parametry modelu HMM"""
        print("\nInitial")
        print(self.initial_ppb)
        print("Transition")
        print(self.transition_ppb)
        print("emission_ppb")
        print(self.emission_ppb)

    # Baum-Welch algorithm
    def learn(self, training_set, laplance_smoothing = 0.001):
        """Funkcja trenuje model HMM za pomocą podanego zbioru uczącego

        :param list training_set: zbiór uczący postaci lista seqwencji obserwacji.
        :param float laplance_smoothing: minimalne prawdopodobieństwo wygenerowania obserwacji przez dany model

        Ponieważ obserwacje modelu są typu string, najpierw zamienia każdą obserwację na elementy typu string.
        Następnie powtarza algorytm Bauma-Welcha na zbiorze uczącym, określoną ilość razy, lub dopóki różnica
        prawdopodobieństw wygenerowania zbioru uczącego w starym modelu i nowym będzie mniejsze niż epsilon.
        """
        observations_num = len(training_set)
        obs_seq_len = len(training_set[0])

        # Ponieważ obseracje modelu są typu string, najpierw zamień każdą obserwację na string
        training_set_str = []
        if not isinstance(training_set[0][0], str):
            for obs_seq in training_set:
                obs_seq_str = list(map(str, obs_seq))
                training_set_str.append(obs_seq_str)
        else:
            training_set_str = training_set

        for estimation_iteration in range(0, 20):
            old_likehood = sum(log(self.evaluate(obs)) for obs in training_set)
            old_likehood /= observations_num

            # file: Training Hidden Markov Models with Multiple Observations – A Combinatorial Method
            self.baum_welch_algorithm(training_set_str, observations_num, obs_seq_len, laplance_smoothing)

            new_likehood = sum(log(self.evaluate(obs)) for obs in training_set)
            new_likehood /= observations_num

            reliability = abs(old_likehood - new_likehood)
            if reliability < .00001:
                break

    # Oblicza prawdopodobieństwo, żę dana sekwencja obserwacji została wyprodukowana przez ten model
    def evaluate(self, obs_sequence):
        """Funckcja oblicza prawdopodobieństwo, że dana sekwencja obserwacji została wyprodukowana przez ten model.

        :param: list obs_sequence: lista obserwacji"""
        if not isinstance(obs_sequence, str):
            obs_sequence_str = list(map(str, obs_sequence))
        else:
            obs_sequence_str = obs_sequence

        obs_sequence_id = [self.observation_dict[obs_sequence_str[i]] for i in range(0, len(obs_sequence_str))]

        alfa = self.forward_algorithm(obs_sequence_id)

        ppb = sum(alfa[state][len(obs_sequence)-1] for state in range(self.hidden_states_num))
        return ppb