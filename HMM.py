import numpy

class HMM:
    def __init__(self, transition_ppb, states_num, observation_num):
        self.hidden_states_num = states_num
        self.transition_ppb = numpy.zeros(shape=(self.hidden_states_num, self.hidden_states_num))
        for i in range(0, self.hidden_states_num):
            self.transition_ppb[i][i] = transition_ppb[i][0]
            self.transition_ppb[i][i+1] = transition_ppb[i][1]

        self.emission_ppb = numpy.zeros(shape=(self.hidden_states_num, observation_num))
        self.observations = []
        self.observations_num = 0
        self.initial_ppb = numpy.zeros(shape=(self.hidden_states_num))

    #  Compute the probability of a state at a certain time, given the history of evidence.
    def forward_algorithm(self, observation_seq):

        observation_len = len(observation_seq)

        alfa = numpy.zeros(shape=(self.hidden_states_num, observation_len))
        for i in range(0, self.hidden_states_num):
            alfa[i][0] = self.initial_ppb[i] * self.emission_ppb[i][observation_seq[0]]

        for t in range(1, observation_len):
            for state_to in range(0, self.hidden_states_num):
                ppb_to = 0
                if state_to == 0:
                    ppb_to = ppb_to + alfa[state_to][t - 1] * self.transition_ppb[state_to][state_to]
                else:
                    ppb_to = sum(alfa[state_from][t - 1] * self.transition_ppb[state_from][state_to]
                                 for state_from in [state_to - 1, state_to])
                alfa[state_to][t] = ppb_to + self.emission_ppb[state_to][observation_seq[t]]

        return alfa

    # C
    def backward_algorithm(self, ob_sequence):
        observation_len = len(ob_sequence)

        beta = numpy.zeros(shape=(self.hidden_states, observation_len))

        for state in range(0, self.hidden_states_num):
            beta[state][observation_len-1] = 1

        for t in range(observation_len - 2, -1, -1):
            for state_from in range(0, self.hidden_states_num):
                if state_from + 1 == self.hidden_states_num:
                    beta[t][state_from] += beta[t + 1][state_from] * self.transition_ppb[state_from][state_from] * \
                                           self.emission_ppb[state_from][ob_sequence[t + 1]]
                else:
                    for state_to in [state_from, state_from + 1]:
                        beta[t][state_from] += beta[t+1][state_to] * self.transition_ppb[state_from][state_to] * \
                                               self.emission_ppb[state_to][ob_sequence[t+1]]

        return beta

    # Baum-Welch algorithm
    def learn(self, observations):
        observations_len = len(observations)
        obs_seq_len = len(observations[0])

        for estimation_iteration in range(0, 50) :
            # ppb table that at time t at state i observation_sequency[t] will be produced
            gamma = numpy.zeros(shape=(self.hidden_states_num, obs_seq_len, observations_len))

            # The probability of a trajectory being in state xi at time t and making the
            # transition to sj at t + 1 given the observation sequence and mode
            trajectory_ppb = numpy.zeros(shape=(obs_seq_len - 1, self.hidden_states_num, self.hidden_states_num, observations_len))

            # Expectation
            observation_id = 0
            for obs_sequence in observations:
                # alfa => [n][n]
                alfa = self.forward_algorithm(obs_sequence)
                # beta => [n][len(obs_sequence)]
                beta = self.backward_algorithm(obs_sequence)

                # P(O|M) - ppb of any path
                ppb_of_any_path = sum(alfa[i][obs_seq_len-1] for i in range(0, obs_seq_len))

                # Compute gamma = ppb that at time t at state i observation_sequency[t] will be produced
                for t in range(0, obs_seq_len-1):
                    ppb_of_any_path = 0
                    for state in range(0, self.hidden_states_num):
                        gamma[state][t][observation_id] += (alfa[state][t] * beta[state][t]) / ppb_of_any_path

                # Compute trajectory - The probability of a trajectory being in state xi at time t and making the
                # transition to sj at t + 1 given the observation sequence and mode
                for t in range(0, obs_seq_len - 1):
                    for state_from in range(0, self.hidden_states_num):
                        if state_from + 1 == self.hidden_states_num:
                            trajectory_ppb[t][state_from][state_from][observation_id] \
                                = (alfa[state_from][t] * self.transition_ppb[state_from][state_from] *
                                   self.emission_ppb[state_from][obs_sequence[t + 1]] * beta[state_from][t + 1]) \
                                  / ppb_of_any_path

                        for state_to in range(state_from, state_from + 1):
                            trajectory_ppb[t][state_from][state_to][observation_id] \
                                = (alfa[state_from][t] * self.transition_ppb[state_from][state_to] *
                                   self.emission_ppb[state_to][obs_sequence[t+1]] * beta[state_to][t+1]) / ppb_of_any_path

                observation_id += 1

            # Maksimalization
            # Levinson's training equations
            # http://vision.gel.ulaval.ca/~parizeau/Publications/P971225.pdf
            for state in range(0, self.hidden_states_num):
                self.initial_ppb[state] = 1/observations_len * sum(gamma[state][0][k] for k in range(0, observations_len))

            old_transition_ppb = self.transition_ppb
            for state_from in range(0, self.hidden_states_num):
                for state_to in range(0, self.hidden_states_num):
                    self.transition_ppb[state_from][state_to] = \
                        sum(sum(trajectory_ppb[t][state_from][state_to][k] for t in range(0, obs_seq_len - 1)) for k in range(0, observations_len)) / \
                        sum(sum(gamma[state_from][t][k] for t in range(0, obs_seq_len - 1)) for k in range(0, observations_len))

            old_emission_ppb = self.emission_ppb
            for state in range(0, self.hidden_states_num):
                for observation in self.observations:
                    val = 0.0
                    for k in range(0, observations_len):
                        for t in range(0, obs_seq_len):
                            if observations[k][t] == observation:
                                val += gamma[state][t][k]
                    val /= sum(sum(gamma[state][t][k] for t in range(0, obs_seq_len)) for k in range(0, observations_len))
                    self.emission_ppb[state][observation] = val

            # https: // www.mimuw.edu.pl / ~pzwiernik / docs / hmm.pdf
            # Porównanie ( za sugestią Levisona,Rabinera,Sondhi )
            if abs(numpy.dot(old_transition_ppb, old_emission_ppb) - numpy.dot(self.transition_ppb, self.emission_ppb)) < .00001:
                break


        