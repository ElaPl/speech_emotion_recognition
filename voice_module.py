import wave
import numpy as np
from math import sqrt
from wav_iterator import WavIterator
from hanning_window import HanningWindow

def print_signal(signal):
    for i in range(0, len(signal)):
        print(signal[i], end="  ")
    print()

class VoiceModule:
    def __init__(self, sample_count):
        self.window = HanningWindow(sample_count)
        self.sample_count = sample_count

    def get_feature_vector(self, filename):
        try:
            wav_file = wave.open(filename, 'rb')
        except IOError:
            print("Can't open file " + filename)
            return []

        wav_iter = WavIterator(wav_file, self.sample_count)

        fundamental_freq_array = []
        energy_deviation_array = []
        for time_domain_vector in wav_iter:
            if len(time_domain_vector) == self.sample_count:
                signal = self.window.plot(time_domain_vector)
            elif len(time_domain_vector) > 1:
                tmp_hanning_module = HanningWindow(len(time_domain_vector))
                signal = tmp_hanning_module.plot(time_domain_vector)
            else:
                break
            frequency_domain_vector = np.fft.rfft(signal)
            fundamental_freq_array.append(self.get_fundamental_freq(frequency_domain_vector, wav_file.getframerate()))
            energy_deviation_array.append(self.get_std_deviation(time_domain_vector))

        wav_file.close()

        pitch_features_vector = self.get_pitch_features(fundamental_freq_array)
        energy_features_vector = self.get_energy_features(energy_deviation_array)
        pitch_features_vector.extend(energy_features_vector)

        return pitch_features_vector

    def get_freq_vector(self, filename):
        try:
            wav_file = wave.open(filename, 'rb')
        except IOError:
            print("Can't open file " + filename)
            return []

        wav_iter = WavIterator(wav_file, self.sample_count)

        fundamental_freq_array = []
        energy_deviation_array = []
        for time_domain_vector in wav_iter:
            if len(time_domain_vector) == self.sample_count:
                signal = self.window.plot(time_domain_vector)
            elif len(time_domain_vector) > 1:
                tmp_hanning_module = HanningWindow(len(time_domain_vector))
                signal = tmp_hanning_module.plot(time_domain_vector)
            else:
                break
            frequency_domain_vector = np.fft.rfft(signal)
            fundamental_freq_array.append(self.get_fundamental_freq(frequency_domain_vector, wav_file.getframerate()))

        return fundamental_freq_array

    def get_fundamental_freq(self, freq_domain_vect, sample_rate):
        max_magnitude = sqrt(pow(np.real(freq_domain_vect[1]), 2) + pow(np.imag(freq_domain_vect[1]), 2))
        max_magnitude_ind = 1
        for i in range(1, len(freq_domain_vect)):
            magnitude_i = sqrt(pow(np.real(freq_domain_vect[i]), 2) + pow(np.imag(freq_domain_vect[i]), 2))

            if magnitude_i > max_magnitude:
                max_magnitude = magnitude_i
                max_magnitude_ind = i

        # print ("sample rate: %d  | len_freq_domain_vec: %d  | mangitude_ind: %d   \n" %(sample_rate, len(freq_domain_vect), max_magnitude_ind))

        return (sample_rate/len(freq_domain_vect)) * max_magnitude_ind

    def get_std_deviation(self, time_domain_vect):
        avg_energy = 0
        for i in range(0, len(time_domain_vect)):
            avg_energy += time_domain_vect[i]
        avg_energy /= len(time_domain_vect)

        sum_power = 0
        for i in range(0, len(time_domain_vect)):
            sum_power += pow(time_domain_vect[i]-avg_energy,2)

        return sqrt((sum_power/len(time_domain_vect)))

    def get_pitch_features(self, fundamental_freq_array):
        max_freq = fundamental_freq_array[0]
        min_freq = fundamental_freq_array[0]
        sum_freq = fundamental_freq_array[0]
        dynamic_tones_counter = 0
        rising_tones_counter = 0
        falling_tones_counter = 0

        for i in range(1, len(fundamental_freq_array)):
            sum_freq += fundamental_freq_array[i]

            if fundamental_freq_array[i] > fundamental_freq_array[i-1]+50:
                dynamic_tones_counter += 1

            if fundamental_freq_array[i] > fundamental_freq_array[i-1]:
                rising_tones_counter += 1

            if fundamental_freq_array[i] < fundamental_freq_array[i-1]:
                falling_tones_counter += 1

            if fundamental_freq_array[i] > max_freq:
                max_freq = fundamental_freq_array[i]

            if fundamental_freq_array[i] < min_freq:
                min_freq = fundamental_freq_array[i]

        vocal_range = max_freq - min_freq
        avg_frequency = sum_freq/len(fundamental_freq_array)
        percent_of_dynamic_tones = 100 * dynamic_tones_counter / len(fundamental_freq_array)
        percent_of_rising_tones = 100 * rising_tones_counter / len(fundamental_freq_array)
        percent_of_falling_tones = 100 * falling_tones_counter / len(fundamental_freq_array)

        # compute standard deviation
        std_sum = 0
        for i in range(0, len(fundamental_freq_array)):
            std_sum += pow(fundamental_freq_array[i] - avg_frequency, 2)

        std_sum /= len(fundamental_freq_array)

        standard_deviation_frequency = sqrt(std_sum)

        return [vocal_range, max_freq, min_freq, avg_frequency, percent_of_dynamic_tones, percent_of_falling_tones,
                percent_of_rising_tones, standard_deviation_frequency]

    def get_energy_features(self, energy_deviation_array):
        avg_deviation = 0
        max_deviation = energy_deviation_array[0]
        min_deviation = energy_deviation_array[0]
        for i in range(0, len(energy_deviation_array)):
            avg_deviation += energy_deviation_array[i]

            if energy_deviation_array[i] > max_deviation:
                max_deviation = energy_deviation_array[i]

            if energy_deviation_array[i] < min_deviation:
                min_deviation = energy_deviation_array[i]

        avg_deviation_energy = avg_deviation/len(energy_deviation_array)

        return [avg_deviation_energy , max_deviation, min_deviation]