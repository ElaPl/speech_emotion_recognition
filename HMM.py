import numpy
from math import log

debug = 0

class HMM:
    def __init__(self, transition_ppb, states_num, observations):
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
        transition_ppb = numpy.zeros(shape=(states_num, states_num))
        for state in range(0, states_num):
            transition_ppb[state][state] = given_transition_ppb[state][0]
            if state + 1 != states_num:
                transition_ppb[state][state + 1] = given_transition_ppb[state][1]

        return transition_ppb

    def create_emission_ppb(self):
        emission_ppb = numpy.zeros(shape=(self.hidden_states_num, self.observations_num))
        for state in range(0, self.hidden_states_num):
            for observation_id in self.observation_dict.values():
                emission_ppb[state][observation_id] = 1 / self.observations_num

        return emission_ppb

    def create_initial_ppb(self, state_num):
        initial_ppb = numpy.zeros(shape=state_num)
        for i in range(0, state_num):
            initial_ppb[i] = 1 / state_num

        return initial_ppb

    def get_parameters(self):
        return self.transition_ppb, self.emission_ppb, self.initial_ppb, self.observation_dict

    #  Compute the probability of a state at a certain time, given the history of evidence.
    def forward_algorithm(self, observation_seq):
        observation_len = len(observation_seq)

        alfa = numpy.zeros(shape=(self.hidden_states_num, observation_len))
        for i in range(0, self.hidden_states_num):
            alfa[i][0] = self.initial_ppb[i] * self.emission_ppb[i][observation_seq[0]]

        for t in range(1, observation_len):
            for state_to in range(0, self.hidden_states_num):
                state_from = state_to
                ppb_to = alfa[state_from][t - 1] * self.transition_ppb[state_from][state_to]
                if debug:
                    print("time, %d, state_to %d" %(t, state_to))
                    print("line 65 : %lf * %lf " %(alfa[state_from][t - 1], self.transition_ppb[state_from][state_to]))

                if state_to > 0:
                    state_from = state_to - 1
                    ppb_to += alfa[state_from][t - 1] * self.transition_ppb[state_from][state_to]
                    if debug:
                        print("line 70 : %lf * %lf " %(alfa[state_from][t - 1],self.transition_ppb[state_from][state_to]))

                alfa[state_to][t] = ppb_to * self.emission_ppb[state_to][observation_seq[t]]
                if debug:
                    print("line 75 : alfa %lf * %lf "  %(ppb_to, self.emission_ppb[state_to][observation_seq[t]]))

        return alfa

    # b_k(i) stores the probability of observing the rest of the sequence after time step i.
    # Given that at time step i we are in state k in the HMM.
    def backward_algorithm(self, ob_sequence):
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

    def baum_welch_algorithm(self, observations, observations_num, obs_seq_len, laplance_smoothing):
        # ppb table that at time t at state i observation_sequency[t] will be produced
        gamma = numpy.zeros(shape=(self.hidden_states_num, obs_seq_len, observations_num))

        # The probability of a trajectory being in state xi at time t and making the
        # transition to sj at t + 1 given the observation sequence and mode
        trajectory_ppb = numpy.zeros(
            shape=(obs_seq_len - 1, self.hidden_states_num, self.hidden_states_num, observations_num))

        # print(observations_num)
        # Expectation
        observation_id = 0
        for obs_sequence in observations:
            # print(len(observations))
            # print(observation_id)
            # print(obs_sequence)
            # for obs in obs_sequence:
            #     print(type(obs))
            obs_sequence_id = [self.observation_dict[obs_sequence[i]] for i in range(0, obs_seq_len)]

            # alfa => [n][len(obs_sequence)]
            alfa = self.forward_algorithm(obs_sequence_id)
            # beta => [n][len(obs_sequence)]
            beta = self.backward_algorithm(obs_sequence_id)

            # P(O|M) - prawdopodobieństwo wygenerowania obserwacji przez model
            ppb_of_any_path = sum(alfa[i][obs_seq_len - 1] for i in range(0, self.hidden_states_num))

            # Compute gamma[i][t] = prawdopodobnieństwo byćia w stanie i w czasie t
            for state in range(0, self.hidden_states_num):
                for t in range(0, obs_seq_len):
                    gamma[state][t][observation_id] += (alfa[state][t] * beta[state][t]) / ppb_of_any_path

            if debug:
                print("Alfa")
                print(alfa)
                print("Beta")
                print(beta)
                print("ppb_any")
                print(ppb_of_any_path)
                print("Gamma")
                print(gamma)

            # Compute trajectory -
            # trajectory_ppb[t][stateA][stateB] - prawdopodobieństwo bycia w casie t w stanie A i przejścia do stanu
            # B w czasie t+1
            for t in range(0, obs_seq_len - 1):
                # prawdopodobieństwo przejśćia w czasie t z dowolnego stanu i do dowolnego stanu j
                for state_from in range(0, self.hidden_states_num):
                    if debug:
                        print("Time: %d, state_from: %d" %(t, state_from))
                    if state_from + 1 == self.hidden_states_num:
                        trajectory_ppb[t][state_from][state_from][observation_id] \
                            = (alfa[state_from][t] * self.transition_ppb[state_from][state_from] *
                               self.emission_ppb[state_from][obs_sequence_id[t + 1]] * beta[state_from][t + 1]) / ppb_of_any_path
                        if debug:
                            print("153: %lf %lf %lf %lf / %lf" %(alfa[state_from][t], self.transition_ppb[state_from][state_from],
                                                                 self.emission_ppb[state_from][obs_sequence_id[t + 1]], beta[state_from][t + 1], ppb_of_any_path))
                    else:
                        for state_to in [state_from, state_from + 1]:
                            trajectory_ppb[t][state_from][state_to][observation_id] \
                                = (alfa[state_from][t] * self.transition_ppb[state_from][state_to] *
                                   self.emission_ppb[state_to][obs_sequence_id[t + 1]] * beta[state_to][t + 1]) / ppb_of_any_path
                            if debug:
                                print("160: %lf %lf %lf %lf / %lf" %(alfa[state_from][t], self.transition_ppb[state_from][state_to],
                                                                     self.emission_ppb[state_to][obs_sequence_id[t + 1]], beta[state_to][t + 1], ppb_of_any_path))
            observation_id += 1

        # Maksimalization
        # Levinson's training equations
        # http://vision.gel.ulaval.ca/~parizeau/Publications/P971225.pdf

        # Reestimate initial pppb
        for state in range(0, self.hidden_states_num):
            exp = sum(gamma[state][0][k] for k in range(observations_num))
            self.initial_ppb[state] = (exp + laplance_smoothing) / \
                                      (1/observations_num + self.hidden_states_num * laplance_smoothing)

        if debug:
            print("Initial state")
            print(self.initial_ppb)
            print(sum(self.initial_ppb[i] for i in range(self.hidden_states_num)))

        # gamma_sum[A] - czestość pobytów w stanie i w czasie 0, T-2
        gamma_sum = numpy.zeros(shape=self.hidden_states_num)
        for state in range(0, self.hidden_states_num):
            gamma_sum[state] = sum(sum(gamma[state][t][k] for t in range(0, obs_seq_len - 1))
                                   for k in range(0, observations_num))

        for state_from in range(0, self.hidden_states_num):
            if state_from + 1 != self.hidden_states_num:
                state_to = state_from + 1

                # Czestośc przejsć z stanu i do staju j
                exp_num = sum(sum(trajectory_ppb[t][state_from][state_to][k] for t in range(0, obs_seq_len - 1))
                              for k in range(0, observations_num))

                self.transition_ppb[state_from][state_to] = (exp_num + laplance_smoothing) / \
                                                            (gamma_sum[state_from] + 2 * laplance_smoothing)
                self.transition_ppb[state_from][state_from] = 1 - self.transition_ppb[state_from][state_to]

            else:
                # Czestośc przejsć z stanu i do staju i
                exp_num = sum(sum(trajectory_ppb[t][state_from][state_from][k] for t in range(0, obs_seq_len - 1))
                              for k in range(0, observations_num))

                self.transition_ppb[state_from][state_from] = (exp_num + laplance_smoothing) / \
                                                              (gamma_sum[state_from] + 2 * laplance_smoothing)
        if debug:
            print("Transition ppb")
            print(self.transition_ppb)

        gamma_sumB = numpy.zeros(shape=self.hidden_states_num)
        for state in range(0, self.hidden_states_num):
            gamma_sumB[state] = sum(sum(gamma[state][t][k] for t in range(0, obs_seq_len))
                                    for k in range(0, observations_num))

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
        print("\nInitial")
        print(self.initial_ppb)
        print("Transition")
        print(self.transition_ppb)
        print("emission_ppb")
        print(self.emission_ppb)

    # Baum-Welch algorithm
    def learn(self, training_set, laplance_smoothing):
        observations_num = len(training_set)
        obs_seq_len = len(training_set[0])

        training_set_str = []
        if not isinstance(training_set[0][0], str):
            for obs_seq in training_set:
                obs_seq_str = list(map(str, obs_seq))
                training_set_str.append(obs_seq_str)
        else:
            training_set_str = training_set

        for estimation_iteration in range(0, 7):
            old_likehood = sum(log(self.evaluate(obs)) for obs in training_set)
            old_likehood /= observations_num

            # file: Training Hidden Markov Models with Multiple Observations – A Combinatorial Method
            self.baum_welch_algorithm(training_set_str, observations_num, obs_seq_len, laplance_smoothing)
            #
            new_likehood = sum(log(self.evaluate(obs)) for obs in training_set)
            new_likehood /= observations_num

            reliability = abs(old_likehood - new_likehood)
            if reliability < .00001:
                break

    # Oblicza prawdopodobieństwo, żę dana sekwencja obserwacji została wyprodukowana przez ten model
    def evaluate(self, obs_sequence):
        # print("Evaluate")

        # obs_sequence_str = []
        if not isinstance(obs_sequence, str):
            obs_sequence_str = list(map(str, obs_sequence))
        else:
            obs_sequence_str = obs_sequence

        obs_sequence_id = [self.observation_dict[obs_sequence_str[i]] for i in range(0, len(obs_sequence_str))]

        alfa = self.forward_algorithm(obs_sequence_id)

        ppb = sum(alfa[state][len(obs_sequence)-1] for state in range(self.hidden_states_num))
        return ppb