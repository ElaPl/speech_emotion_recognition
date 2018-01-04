import wave
import numpy as np
from math import sqrt
from wav_iterator import WavIterator
from hanning_window import HanningWindow

voice_freq_scale = [
    {"id": 1, "min": 0, "max": 150},
    {"id": 2, "min": 151, "max": 300},
    {"id": 3, "min": 300, "max": 500},
    {"id": 4, "min": 501, "max": 700},
    {"id": 5, "min": 701, "max": 900},
    {"id": 6, "min": 901, "max": 1300},
    {"id": 7, "min": 1301, "max": 2000},
    {"id": 8, "min": 2001, "max": 4000},
    {"id": 9, "min": 4001, "max": 6000},
    {"id": 10, "min": 6001, "max": 30000},
]

def print_signal(signal):
    for i in range(0, len(signal)):
        print(signal[i], end="  ")
    print()


class VoiceModule:

    def get_pitch_feature_vector(self, filename, frame_length):
        window = HanningWindow(frame_length)

        try:
            wav_file = wave.open(filename, 'rb')
        except IOError:
            print("Can't open file " + filename)
            return []

        sample_rate = wav_file.getframerate()
        sample_in_sec_quarter = sample_rate / 4
        frames_num = sample_in_sec_quarter/frame_length
        if sample_in_sec_quarter % frame_length != 0:
            frames_num += 1

        features_vectors_container = []
        frame_counter = 0
        wav_iter = WavIterator(wav_file, frame_length)
        fundamental_freq_array = []
        for time_domain_vector in wav_iter:
            time_domain_vect_size = len(time_domain_vector)
            if time_domain_vect_size == frame_length:
                signal = window.plot(time_domain_vector)
                frame_counter += 1
            elif time_domain_vect_size >= 128:
                tmp_hanning_module = HanningWindow(len(time_domain_vector))
                signal = tmp_hanning_module.plot(time_domain_vector)
                frame_counter += 1

            if time_domain_vect_size != 0:
                frequency_domain_vector = np.fft.rfft(signal)
                fundamental_freq_array.append(self.get_fundamental_freq(frequency_domain_vector, wav_file.getframerate(),
                                                                        len(time_domain_vector)))

            if frame_counter >= frames_num or (time_domain_vect_size == 0 and frame_counter != 0):
                frame_counter = 0
                features_vector = self.get_pitch_features(fundamental_freq_array)
                features_vectors_container.append(features_vector)
                fundamental_freq_array = []

        wav_file.close()
        return features_vectors_container

    def get_energy_feature_vector(self, filename, frame_length):
        window = HanningWindow(frame_length)

        try:
            wav_file = wave.open(filename, 'rb')
        except IOError:
            print("Can't open file " + filename)
            return []

        features_vectors_container = []
        wav_iter = WavIterator(wav_file, frame_length)
        for time_domain_vector in wav_iter:
            if len(time_domain_vector) == frame_length:
                signal = window.plot(time_domain_vector)
            elif len(time_domain_vector) > 100:
                tmp_hanning_module = HanningWindow(len(time_domain_vector))
                signal = tmp_hanning_module.plot(time_domain_vector)
            else:
                break

            features_vectors_container.append(self.get_energy_features(signal))

        wav_file.close()
        return features_vectors_container

    @staticmethod
    def get_file_info(filename):
        try:
            wav_file = wave.open(filename, 'rb')
        except IOError:
            print("Can't open file " + filename)
            return []

        file_params = wav_file.getparams()

        wav_file.close()
        return file_params

    @staticmethod
    def get_sample_rate(filename):
        try:
            wav_file = wave.open(filename, 'rb')
        except IOError:
            print("Can't open file " + filename)
            return []

        sample_rate = wav_file.getframerate()

        wav_file.close()
        return sample_rate

    def get_freq_vector(self, filename, frame_length):
        window = HanningWindow(frame_length)

        try:
            wav_file = wave.open(filename, 'rb')
        except IOError:
            print("Can't open file " + filename)
            return []

        wav_iter = WavIterator(wav_file, frame_length)

        fundamental_freq_array = []
        for time_domain_vector in wav_iter:
            if len(time_domain_vector) == frame_length:
                signal = window.plot(time_domain_vector)
            elif len(time_domain_vector) > 1:
                tmp_hanning_module = HanningWindow(len(time_domain_vector))
                signal = tmp_hanning_module.plot(time_domain_vector)
            else:
                break
            frequency_domain_vector = np.fft.rfft(signal)
            fundamental_freq_array.append(self.get_fundamental_freq(frequency_domain_vector, wav_file.getframerate(),
                                                                    len(time_domain_vector)))

        wav_file.close()

        return fundamental_freq_array

    def get_energy_vector(self, filename, frame_length):
        window = HanningWindow(frame_length)

        try:
            wav_file = wave.open(filename, 'rb')
        except IOError:
            print("Can't open file " + filename)
            return []

        wav_iter = WavIterator(wav_file, frame_length)

        energy_array = []
        for time_domain_vector in wav_iter:
            if len(time_domain_vector) == frame_length:
                signal = window.plot(time_domain_vector)
            elif len(time_domain_vector) > 1:
                tmp_hanning_module = HanningWindow(len(time_domain_vector))
                signal = tmp_hanning_module.plot(time_domain_vector)
            else:
                break
            energy_array.extend(signal)

        wav_file.close()

        return energy_array

    @staticmethod
    def get_fundamental_freq(freq_domain_vect, sample_rate, sample_length):
        max_magnitude = sqrt(np.power(np.real(freq_domain_vect[1]), 2) + np.power(np.imag(freq_domain_vect[1]), 2))
        max_magnitude_ind = 1
        for i in range(1, len(freq_domain_vect)):
            magnitude_i = sqrt(np.power(np.real(freq_domain_vect[i]), 2) + np.power(np.imag(freq_domain_vect[i]), 2))

            if magnitude_i > max_magnitude:
                max_magnitude = magnitude_i
                max_magnitude_ind = i

        # print ("sample rate: %d  | len_freq_domain_vec: %d  | mangitude_ind: %d   \n" %(sample_rate, len(freq_domain_vect), max_magnitude_ind))

        return (sample_rate/sample_length) * max_magnitude_ind

    @staticmethod
    def get_scale_id(freq):
        for scale in voice_freq_scale:
            if scale['min'] <= freq <= scale['max']:
                return scale['id']

    def get_pitch_features(self, fundamental_freq_array):
        max_freq = fundamental_freq_array[0]
        min_freq = fundamental_freq_array[0]
        sum_freq = fundamental_freq_array[0]
        dynamic_tones_counter = 0
        rising_tones_counter = 0
        falling_tones_counter = 0
        frames_with_dynamic_tones = []
        freq_scale_counter = [0] * len(voice_freq_scale)

        for i in range(1, len(fundamental_freq_array)):
            sum_freq += fundamental_freq_array[i]

            freq_scale_counter[self.get_scale_id(fundamental_freq_array[i])-1] += 1
            # for scale in voice_freq_scale:
            #     if scale['min'] <= fundamental_freq_array[i] <= scale['max']:
            #         freq_scale_counter[scale['id']] += 1

            if fundamental_freq_array[i] > fundamental_freq_array[i-1] + 3000:
                frames_with_dynamic_tones.append(i)
                dynamic_tones_counter += 1

            if fundamental_freq_array[i] > fundamental_freq_array[i-1]:
                rising_tones_counter += 1

            if fundamental_freq_array[i] < fundamental_freq_array[i-1]:
                falling_tones_counter += 1

            if fundamental_freq_array[i] > max_freq:
                max_freq = fundamental_freq_array[i]

            if fundamental_freq_array[i] < min_freq:
                min_freq = fundamental_freq_array[i]

        max_dynamic_freq_id = 0
        max_dynamic_freq_occurance = freq_scale_counter[max_dynamic_freq_id]

        for i in range(1, len(freq_scale_counter)):
            if freq_scale_counter[i] > max_dynamic_freq_occurance:
                max_dynamic_freq_id = i
                max_dynamic_freq_occurance = freq_scale_counter[i]

        # To avoid 0
        max_dynamic_freq_id += 1

        vocal_range = max_freq - min_freq
        avg_frequency = sum_freq/len(fundamental_freq_array)
        dynamic_tones_frequency = dynamic_tones_counter / len(fundamental_freq_array)
        percent_of_rising_tones = 100 * (rising_tones_counter / len(fundamental_freq_array))
        percent_of_falling_tones = 100 * (falling_tones_counter / len(fundamental_freq_array))

        # compute standard deviation
        variance = 0
        for i in range(0, len(fundamental_freq_array)):
            variance += pow(fundamental_freq_array[i] - avg_frequency, 2)

        variance /= len(fundamental_freq_array)

        standard_deviation_frequency = sqrt(variance)

        return [vocal_range, self.get_scale_id(max_freq), self.get_scale_id(min_freq), max_dynamic_freq_id,
                dynamic_tones_frequency, percent_of_falling_tones, percent_of_rising_tones,
                standard_deviation_frequency, variance]

    def get_summary_pitch_feature_vector(self, pitch_feature_vectors):
        pitch_feature_vectors_size = len(pitch_feature_vectors)
        max_freq = pitch_feature_vectors[0][1]
        min_freq = pitch_feature_vectors[0][2]
        avg_freq = pitch_feature_vectors[0][3]
        falling_tones_counter = 0
        rising_tones_counter = 0
        dynamic_tones_freq = 0
        if pitch_feature_vectors[0][4] != 0:
            dynamic_tones_freq = 1

        for i in range(1, pitch_feature_vectors_size):
            avg_freq += pitch_feature_vectors[i][3]

            if pitch_feature_vectors[i][1] > max_freq:
                max_freq = pitch_feature_vectors[i][1]

            if pitch_feature_vectors[i][2] < min_freq:
                min_freq = pitch_feature_vectors[i][2]

            if pitch_feature_vectors[i][4] > 0:
                dynamic_tones_freq += 1

            if pitch_feature_vectors[i][3] > pitch_feature_vectors[i-1][3]:
                rising_tones_counter += 1

            if pitch_feature_vectors[i][3] < pitch_feature_vectors[i-1][3]:
                falling_tones_counter += 1


        avg_freq /= pitch_feature_vectors_size
        freq_range = max_freq - min_freq
        dynamic_tones_freq /= pitch_feature_vectors_size
        percent_of_falling_tones = (falling_tones_counter/pitch_feature_vectors_size) * 100
        percent_of_rising_tones = (rising_tones_counter/pitch_feature_vectors_size) * 100

        variance = 0
        for i in range(1, pitch_feature_vectors_size):
            variance += pow(pitch_feature_vectors[i][3] - avg_freq, 2)

        std_deviation = sqrt(variance)

        return [freq_range, max_freq, min_freq, avg_freq, dynamic_tones_freq, percent_of_falling_tones,
                percent_of_rising_tones, std_deviation, variance]


    @staticmethod
    def get_energy_features(time_domain_signal):
        crossing_rate = 0
        min_value = time_domain_signal[0]
        max_value = time_domain_signal[0]
        sum = time_domain_signal[0]

        for i in range(1, len(time_domain_signal)):
            sum += time_domain_signal[i]

            if time_domain_signal[i] < min_value:
                min_value = time_domain_signal[i]

            if time_domain_signal[i] > max_value:
                max_value = time_domain_signal[i]

            if (time_domain_signal[i-1] < 0 and time_domain_signal[i] > 0 or
                time_domain_signal[i-1] == 0 and time_domain_signal[i] != 0 or
                time_domain_signal[i-1] > 0 and time_domain_signal[i] < 0):
                crossing_rate += 1

        range_energy = max_value - min_value
        avg_value = sum/len(time_domain_signal)

        variance = 0
        for i in range(1, len(time_domain_signal)):
            variance += pow(time_domain_signal[i] - avg_value, 2)

        variance /= len(time_domain_signal)
        standard_deviation = sqrt(variance)

        return [standard_deviation, variance, 100 * crossing_rate/len(time_domain_signal)]
