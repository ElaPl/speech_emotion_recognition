import wave
import numpy as np
from math import sqrt, log10
from wav_iterator import WavIterator
from hanning_window import HanningWindow

voice_freq_scale = [
    {"id": 1, "min": 0, "max": 100},
    {"id": 2, "min": 101, "max": 200},
    {"id": 3, "min": 201, "max": 300},
    {"id": 4, "min": 301, "max": 500},
    {"id": 5, "min": 501, "max": 700},
    {"id": 6, "min": 701, "max": 900},
    {"id": 7, "min": 901, "max": 1300},
    {"id": 8, "min": 1301, "max": 2000},
    {"id": 9, "min": 2001, "max": 4000},
    {"id": 10, "min": 4001, "max": 6000},
    {"id": 11, "min": 6001, "max": 30000},
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
        sample_in_sec_quarter = sample_rate / 2
        frames_num = sample_in_sec_quarter/frame_length
        if sample_in_sec_quarter % frame_length != 0:
            frames_num += 1

        features_vectors_container = []
        frame_counter = 0
        wav_iter = WavIterator(wav_file, frame_length)
        fundamental_freq_array = []
        for time_domain_vector in wav_iter:
            time_domain_vect_size = len(time_domain_vector)
            signal = []
            if time_domain_vect_size == frame_length:
                signal = window.plot(time_domain_vector)
                frame_counter += 1
            elif time_domain_vect_size >= 128:
                tmp_hanning_module = HanningWindow(len(time_domain_vector))
                signal = tmp_hanning_module.plot(time_domain_vector)
                frame_counter += 1

            if signal and time_domain_vect_size != 0:
                frequency_domain_vector = np.fft.rfft(signal)
                fundamental_freq_array.append(
                    self.get_fundamental_freq(frequency_domain_vector, wav_file.getframerate(),
                                              len(time_domain_vector)))

            if frame_counter >= frames_num or (time_domain_vect_size == 0 and frame_counter != 0):
                frame_counter = 0
                pitch_feature_vec = self.get_pitch_features(fundamental_freq_array)
                if len(pitch_feature_vec) != 0:
                    features_vector = pitch_feature_vec
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

            energy_feature_vec = self.get_energy_features(signal)
            if len(energy_feature_vec) > 0:
                features_vectors_container.append(energy_feature_vec)

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

    @staticmethod
    def get_energy_vector(filename, frame_length):
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

        return (sample_rate/sample_length) * max_magnitude_ind

    @staticmethod
    def get_scale_id(freq):
        for scale in voice_freq_scale:
            if scale['min'] <= freq <= scale['max']:
                return scale['id']

    def get_pitch_features(self, fundamental_freq_array):
        if len(fundamental_freq_array) == 0:
            return []
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

        if sum_freq == 0:
            return []

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

    @staticmethod
    def get_summary_pitch_feature_vector(pitch_feature_vectors):
        pitch_feature_vectors_size = len(pitch_feature_vectors)
        max_freq_range = 1
        min_freq_range = len(voice_freq_scale)
        dynamic_tones_freq = 0
        freq_scale_counter = [0] * len(voice_freq_scale)
        avg_range = 0

        if pitch_feature_vectors[0][4] != 0:
            dynamic_tones_freq = 1

        for i in range(0, pitch_feature_vectors_size):
            freq_scale_counter.append(pitch_feature_vectors[i][3])
            avg_range += pitch_feature_vectors[i][3]

            if pitch_feature_vectors[i][1] > max_freq_range:
                max_freq_range = pitch_feature_vectors[i][1]

            if pitch_feature_vectors[i][2] < min_freq_range:
                min_freq_range = pitch_feature_vectors[i][2]

            if pitch_feature_vectors[i][4] > 0:
                dynamic_tones_freq += 1

        avg_range /= pitch_feature_vectors_size
        freq_range = max_freq_range - min_freq_range
        dynamic_tones_freq /= pitch_feature_vectors_size

        variance = 0
        for i in range(1, pitch_feature_vectors_size):
            variance += pow(pitch_feature_vectors[i][3] - avg_range, 2)

        std_deviation = sqrt(variance)

        return [freq_range, max_freq_range, min_freq_range, avg_range, 100*dynamic_tones_freq, std_deviation, variance]

    @staticmethod
    def get_energy_features(time_domain_signal):
        time_domain_signal_len = len(time_domain_signal)
        if time_domain_signal_len == 0:
            return []

        peak = time_domain_signal[0]
        sound_vol_rms = 0
        sound_vol_avg = 0
        amplitude_ration_signal = [0] * time_domain_signal_len
        zero_crossing_rate = 0

        for i in range(0, time_domain_signal_len):
            sound_vol_rms += pow(time_domain_signal[i], 2)
            peak = max(peak, time_domain_signal[i])

            if time_domain_signal[i] != 0:
                amplitude_ration_signal[i] = 10 * log10(pow(time_domain_signal[i], 2))
                sound_vol_avg += amplitude_ration_signal[i]

            if i > 0:
                if (amplitude_ration_signal[i-1] < 0 < amplitude_ration_signal[i]
                    or amplitude_ration_signal[i-1] == 0 and amplitude_ration_signal[i] != 0
                    or amplitude_ration_signal[i-1] > 0 > amplitude_ration_signal[i]):
                    zero_crossing_rate += 1

        if sound_vol_rms == 0:
            return []

        sound_vol_rms = sqrt(sound_vol_rms / time_domain_signal_len)
        rms_db = 10 * log10(sound_vol_rms)
        peak_db = 10 * log10(pow(peak, 2) / pow(sound_vol_rms, 2))
        zero_crossing_rate /= (time_domain_signal_len - 1)
        sound_vol_avg /= time_domain_signal_len

        variance = 0
        for i in range(0, time_domain_signal_len):
            variance += pow(amplitude_ration_signal[i] - sound_vol_avg, 2)

        variance /= time_domain_signal_len
        standard_deviation = sqrt(variance)

        return [standard_deviation, variance, 100 * zero_crossing_rate, sound_vol_rms, rms_db, peak_db]
