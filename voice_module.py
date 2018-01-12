import wave
import numpy as np
from math import sqrt, log10
from wav_iterator import WavIterator
from hanning_window import HanningWindow
import struct


# Odczytaj określoną ilość próbek z jednego channelu z pliku wav
def read_from_wav_file(wav_file, length):
    sizes = {1: 'B', 2: 'h', 4: 'i'}
    fmt_size = sizes[wav_file.getsampwidth()]

    fmt = "<" + fmt_size * length * wav_file.getnchannels()

    decoded = struct.unpack(fmt, wav_file.readframes(length))
    decoded_on_channel = []

    for i in range(0, len(decoded), wav_file.getnchannels()):
        decoded_on_channel.append(decoded[i])
    return decoded_on_channel

# Dziwięk dzielimy na kawałki o długości ~0,25ms
# Zwraca listę features z każdych 25ms pliku
def get_feature_vectors(file):

    try:
        wav_file = wave.open(file, 'rb')
    except IOError:
        print("Can't open file " + file)
        return []

    # liczba sampli / sek
    frame_rate = wav_file.getframerate()

    frame_length = 512

    # liczba ramek w 0,25 ms
    frame_num = int(frame_rate / 4 / frame_length)

    # ilosć sampli w 0,25 ms
    sample_len = frame_num * frame_length

    try:
        sample = read_from_wav_file(wav_file, sample_len)
    except wave.Error:
        return []

    pitch_window = HanningWindow(frame_length)
    energy_window = HanningWindow(sample_len)

    pitch_feature_vectors = [get_pitch_feature_vector(sample, frame_length, pitch_window, frame_rate)]
    energy_feature_vectors = [get_energy_feature_vector(sample, energy_window)]

    sample_len = int(sample_len / 2)

    while wav_file.tell() + sample_len < wav_file.getnframes():
        sample_next = sample[sample_len:]
        sample_next.extend(read_from_wav_file(wav_file, sample_len))
        pitch_feature_vectors.append(get_pitch_feature_vector(sample_next, frame_length, pitch_window, frame_rate))
        energy_feature_vectors.append(get_energy_feature_vector(sample_next, energy_window))
        sample = sample_next

    summary_pitch_feature_vectors = get_summary_pitch_feature_vector(pitch_feature_vectors)

    return pitch_feature_vectors, energy_feature_vectors, summary_pitch_feature_vectors

def get_pitch_feature_vector(sample, frame_length, window, frame_rate):
    fundamental_freq_array = []
    sample_len = len(sample)
    sample_pointer = 0
    while sample_pointer + frame_length < sample_len:
        frame = sample[sample_pointer:(sample_pointer + frame_length)]
        sample_pointer += frame_length
        signal = window.plot(frame)
        frequency_domain_vector = np.fft.rfft(signal)
        fundamental_freq_array.append(
            get_fundamental_freq(frequency_domain_vector, frame_rate, frame_length))

    return get_pitch_features(fundamental_freq_array)


def get_file_info(filename):
    try:
        wav_file = wave.open(filename, 'rb')
    except IOError:
        print("Can't open file " + filename)
        return []

    file_params = wav_file.getparams()

    wav_file.close()
    return file_params


def get_sample_rate(filename):
    try:
        wav_file = wave.open(filename, 'rb')
    except IOError:
        print("Can't open file " + filename)
        return []

    sample_rate = wav_file.getframerate()

    wav_file.close()
    return sample_rate

def get_freq_vector(filename, frame_length):
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
        fundamental_freq_array.append(get_fundamental_freq(frequency_domain_vector, wav_file.getframerate(),
                                                           len(time_domain_vector)))

    wav_file.close()

    return fundamental_freq_array


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


def get_fundamental_freq(freq_domain_vect, sample_rate, sample_length):
    max_magnitude = sqrt(np.power(np.real(freq_domain_vect[1]), 2) + np.power(np.imag(freq_domain_vect[1]), 2))
    max_magnitude_ind = 1
    for i in range(1, len(freq_domain_vect)):
        magnitude_i = sqrt(np.power(np.real(freq_domain_vect[i]), 2) + np.power(np.imag(freq_domain_vect[i]), 2))

        if magnitude_i > max_magnitude:
            max_magnitude = magnitude_i
            max_magnitude_ind = i

    return (sample_rate/sample_length) * max_magnitude_ind


def get_scale_id(scale_dict, value):
    for scale in scale_dict:
        if scale['min'] <= value < scale['max']:
            return scale['id']


def get_pitch_features(fundamental_freq_array):
    if len(fundamental_freq_array) == 0:
        return []
    max_freq = fundamental_freq_array[0]
    min_freq = fundamental_freq_array[0]
    sum_freq = 0
    rising_tones_counter = 0
    falling_tones_counter = 0

    for i in range(0, len(fundamental_freq_array)):
        sum_freq += fundamental_freq_array[i]

        max_freq = max(max_freq, fundamental_freq_array[i])
        min_freq = min(min_freq, fundamental_freq_array[i])

        if fundamental_freq_array[i] > fundamental_freq_array[i-1]:
            rising_tones_counter += 1

        if fundamental_freq_array[i] < fundamental_freq_array[i-1]:
            falling_tones_counter += 1

    if sum_freq == 0:
        return []

    # print(sum_freq)
    vocal_range = max_freq - min_freq
    avg_frequency = sum_freq/len(fundamental_freq_array)
    percent_of_rising_tones = 100 * (rising_tones_counter / len(fundamental_freq_array))
    percent_of_falling_tones = 100 * (falling_tones_counter / len(fundamental_freq_array))

    dynamic_tones_percent = 0
    variance = 0
    for i in range(0, len(fundamental_freq_array)):
        variance += pow(fundamental_freq_array[i] - avg_frequency, 2)
        if fundamental_freq_array[i] >= avg_frequency + 3000:
            dynamic_tones_percent += 1

    dynamic_tones_percent = (dynamic_tones_percent / len(fundamental_freq_array)) * 100
    standard_deviation_frequency = sqrt(variance/(len(fundamental_freq_array)-1))
    relative_std_deviation = (standard_deviation_frequency/avg_frequency) * 100
    # print(avg_frequency)
    return [vocal_range, max_freq, min_freq, avg_frequency, dynamic_tones_percent, percent_of_falling_tones,
            percent_of_rising_tones, relative_std_deviation]


def get_summary_pitch_feature_vector(pitch_feature_vectors):
    pitch_feature_vectors_size = len(pitch_feature_vectors)
    max_freq_range = pitch_feature_vectors[0][3]
    min_freq_range = pitch_feature_vectors[0][3]
    avg_range = 0
    dynamic_tones_percent = 0

    for i in range(0, pitch_feature_vectors_size):
        avg_range += pitch_feature_vectors[i][3]

        if pitch_feature_vectors[i][1] > max_freq_range:
            max_freq_range = pitch_feature_vectors[i][1]

        if pitch_feature_vectors[i][2] < min_freq_range:
            min_freq_range = pitch_feature_vectors[i][2]

        dynamic_tones_percent += pitch_feature_vectors[i][4] / 100

    dynamic_tones_percent = (dynamic_tones_percent / pitch_feature_vectors_size) * 100
    avg_range /= pitch_feature_vectors_size
    freq_range = max_freq_range - min_freq_range

    variance = 0
    for i in range(1, pitch_feature_vectors_size):
        variance += pow(pitch_feature_vectors[i][3] - avg_range, 2)

    std_deviation = sqrt(variance/(len(pitch_feature_vectors)-1))
    relative_std_deviation = (std_deviation / avg_range) * 100

    return [freq_range, max_freq_range, min_freq_range, avg_range, dynamic_tones_percent, relative_std_deviation]

def get_energy_feature_vector(sample, window):
    time_domain_signal = window.plot(sample)
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
    zero_crossing_rate = (zero_crossing_rate/ (time_domain_signal_len - 1)) * 100
    sound_vol_avg /= time_domain_signal_len

    standard_deviation = 0
    for i in range(0, time_domain_signal_len):
        standard_deviation += pow(amplitude_ration_signal[i] - sound_vol_avg, 2)

    standard_deviation = sqrt(standard_deviation/(time_domain_signal_len-1))
    relative_std_deviation = (standard_deviation/sound_vol_avg) * 100
    return [relative_std_deviation, zero_crossing_rate, rms_db, peak_db]