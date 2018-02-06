import wave
import numpy as np
from math import sqrt, log10
from hanning_window import HanningWindow
import struct


def read_from_wav_file(wav_file, length):
    """Funkcja odczytuje z pliku wav określoną ilość próbek pochodzących z jednego kanału.

    :param wav_file: wskaźnik na plik wav
    :param int length: liczba sampli jaką chcemy odczytać

    :return: lista sampli długości length, pochodzących z jednego kanału.
    :rtype: list
    Opis:

    Sample w plikach wav z więcej niż jednym kanałem są ustawione naprzemiennie. Najpierw jest pierszy sampel kanału 1,
     następnie pierwszy sampel kanału 2 itd. dopiero później są sample dla kanału 2.

    Aby więc odczytać informację o długości n należy odczytać długość sampla * liczbę kanałów * rządana długość, a
    następnie z odczytanych danych wziąć elemnty z określonego kanału. Na przykład dla trzek kanałów należy
    wziać co trzeci element.
    """
    sizes = {1: 'B', 2: 'h', 4: 'i'}
    fmt_size = sizes[wav_file.getsampwidth()]

    fmt = "<" + fmt_size * length * wav_file.getnchannels()

    decoded = struct.unpack(fmt, wav_file.readframes(length))
    decoded_on_channel = []

    for i in range(0, len(decoded), wav_file.getnchannels()):
        decoded_on_channel.append(decoded[i])
    return decoded_on_channel


def get_feature_vectors(file):
    """ Funckja otwiera plik .wav a nastepnie co 0,125ms z pliku odczytuje próbkę dźwięku od długości ok 0,25 ms.
    Z każdej próbki oblicza wektor cech częstotliwości i energii, tworząc wektory cech.

    :param str file: ścieżka do pliku z którego mają być wygenerowane wektory cech

    :return:
        * lista wektorów cech częstotliwości
        * lista wektorów cech energii
        * lista wektorów cech częstotliwości i energii
    :rtype: dictionary
    """


    try:
        wav_file = wave.open(file, 'rb')
    except IOError:
        print("Can't open file " + file)
        return []

    frame_rate = wav_file.getframerate()

    frame_length = 512

    frame_num = int(frame_rate / 4 / frame_length)

    sample_len = frame_num * frame_length

    try:
        sample = read_from_wav_file(wav_file, sample_len)
    except wave.Error:
        return []

    pitch_window = HanningWindow(frame_length)
    energy_window = HanningWindow(sample_len)

    pitch_feature_vectors = [get_pitch_features(get_fundamental_freq_form_time_domain
                                                (sample, frame_length, pitch_window, frame_rate))]
    energy_feature_vectors = [get_energy_feature_vector(sample, energy_window)]

    all_feature_vector = [pitch_feature_vectors[0] + energy_feature_vectors[0]]

    sample_len = int(sample_len / 2)

    while wav_file.tell() + sample_len < wav_file.getnframes():
        sample_next = sample[sample_len:]
        sample_next.extend(read_from_wav_file(wav_file, sample_len))

        pitch_vector = get_pitch_features(get_fundamental_freq_form_time_domain
                                          (sample_next, frame_length, pitch_window, frame_rate))
        pitch_feature_vectors.append(pitch_vector)

        energy_vector = get_energy_feature_vector(sample_next, energy_window)
        energy_feature_vectors.append(energy_vector)

        all_vector = pitch_vector + energy_vector
        all_feature_vector.append(all_vector)

        sample = sample_next

    return pitch_feature_vectors, energy_feature_vectors, all_feature_vector


def get_file_info(filename):
    """
    Zwraca informacje o podanym pliku

    :param str filename: Ścieżka do pliku z rozszerzeniem wav

    :return: parametry pliku
    :rtype: dictionary
    """
    try:
        wav_file = wave.open(filename, 'rb')
    except IOError:
        print("Can't open file " + filename)
        return []

    file_params = wav_file.getparams()

    wav_file.close()
    return file_params


def get_sample_rate(filename):
    """

    :param str filename: Ścieżka do pliku z rozszerzeniem wav

    :return: częstotliwość samplowania
    :rtype: int
    """
    try:
        wav_file = wave.open(filename, 'rb')
    except IOError:
        print("Can't open file " + filename)
        return []

    sample_rate = wav_file.getframerate()

    wav_file.close()
    return sample_rate


def get_fundamental_freq(freq_domain_vect, sample_rate, sample_length):
    """
    Funkcja dla podanej jako argument funkcji w domenie częstotliwości, oblicza wartość tonu podstawowego

    :param vector freq_domain_vect: wektor reprezentujacy funkcję w domenie częstotliowści
    :param int sample_rate: częstotliwość próbkowania dźwięku z którego pochodzi funkcja
    :param int sample_length: długosć ramki z której została wygenerowana funkcja

    :return: wartość tonu podstawowego obliczonego z podanej funkcji
    :rtype: float
    """
    max_magnitude = sqrt(np.power(np.real(freq_domain_vect[1]), 2) + np.power(np.imag(freq_domain_vect[1]), 2))
    max_magnitude_ind = 1
    for i in range(1, len(freq_domain_vect)):
        magnitude_i = sqrt(np.power(np.real(freq_domain_vect[i]), 2) + np.power(np.imag(freq_domain_vect[i]), 2))

        if magnitude_i > max_magnitude:
            max_magnitude = magnitude_i
            max_magnitude_ind = i

    return (sample_rate/sample_length) * max_magnitude_ind


def get_pitch_features(fundamental_freq_array):
    """
    Funkcja na podstawie podanej listy wartości tonów podstawowych oblicza wektor cech dla tych danych

    :param vector fundamental_freq_array: wektor częstotliowści bazowych

    :return: lista cech obliczonych na podstawie podanej jako argument funckji
    :rtype: list
    """

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

    vocal_range = max_freq - min_freq
    avg_frequency = sum_freq/len(fundamental_freq_array)
    percent_of_rising_tones = 100 * (rising_tones_counter / len(fundamental_freq_array))
    percent_of_falling_tones = 100 * (falling_tones_counter / len(fundamental_freq_array))

    dynamic_tones_percent = 0
    variance = 0
    for i in range(0, len(fundamental_freq_array)):
        variance += pow(fundamental_freq_array[i] - avg_frequency, 2)
        if fundamental_freq_array[i] >= avg_frequency + 700:
            dynamic_tones_percent += 1

    dynamic_tones_percent = (dynamic_tones_percent / len(fundamental_freq_array)) * 100
    standard_deviation_frequency = sqrt(variance/(len(fundamental_freq_array)-1))
    relative_std_deviation = (standard_deviation_frequency/avg_frequency) * 100

    return [vocal_range, max_freq, min_freq, avg_frequency, percent_of_falling_tones,
            percent_of_rising_tones, relative_std_deviation]


def get_summary_pitch_feature_vector(pitch_feature_vectors):
    """
    Funkcja na podstawie danych oblicza wektor cech częstotliwośći bazowych

    :param pitch_feature_vectors: lista wektorów cech częstotliowśći bazowych

    :return: wektor cech
    :rtype: list
    """

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
    for i in range(0, pitch_feature_vectors_size):
        variance += pow(pitch_feature_vectors[i][3] - avg_range, 2)

    std_deviation = sqrt(variance/(len(pitch_feature_vectors)-1))
    relative_std_deviation = (std_deviation / avg_range) * 100

    return [freq_range, max_freq_range, min_freq_range, avg_range, dynamic_tones_percent, relative_std_deviation]


def get_energy_feature_vector(sample, window):
    """
    Funkcja na podstawie podanej listy amplitude w domenie czasu oblicza wektor cech dla tych danych

    :param vector sample: lista zmian energii w domenie czasu
    :param window: funkcja okna

    :return: lista cech na podstawie wprowadzonych danych
    :rtype: list
    """
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


def get_energy_history(file):
    """
    Funkcja otwiera plik podany w ścieżce, oraz oblicza rozkłąd wartości nateżęnia w czasie w tym pliku.

    :param str file: Ścieżka do pliku

    :return: lista zawierająca wartości natężenia w czasie w podanym pliku
    :rtype: list
    """

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

    energy_window = HanningWindow(frame_length)

    energy_history = []

    while wav_file.tell() + sample_len < wav_file.getnframes():
        sample = read_from_wav_file(wav_file, frame_length)
        energy_history.append(compute_rms_db(sample, energy_window))

    return energy_history

def compute_rms_db(time_domain_signal, window):
    """Funkcja mnoży podany sygnał przez podene okno i oblicza średnią
    wadratową wartości natężenia tego sygnału

    :param list time_domain_singal: sygnał w domenie czasu
    :param window: objekt reprezentujący funkcję okna.

    :return: średnia kwadratowa wartości natężenia
    :rtype: double
    """
    time_domain_signal = window.plot(time_domain_signal)
    rms_db = 0
    for mag in time_domain_signal:
        rms_db += pow(mag, 2)

    rms_db = sqrt(rms_db / len(time_domain_signal))
    rms_db = 10 * log10(rms_db)
    return rms_db


def get_freq_history(file):
    """ Funckja otwiera plik wav i dzieli go na kawałki o długości ~0,25s i z każdego fragmentu oblicza wartość tonu
    podstawowego

    :param str file: ścieżka do pliku z którego mają być wygenerowane wektory cech

    :return:
        * lista zawierająca wartości tonu podstawego w czasie w podanym pliku
    :rtype: list
    """


    try:
        wav_file = wave.open(file, 'rb')
    except IOError:
        print("Can't open file " + file)
        return []

    frame_rate = wav_file.getframerate()

    frame_length = 512

    frame_num = int(frame_rate / 4 / frame_length)

    sample_len = frame_num * frame_length

    pitch_window = HanningWindow(frame_length)

    frequency_feature_time_domain = []

    while wav_file.tell() + sample_len < wav_file.getnframes():
        sample = read_from_wav_file(wav_file, sample_len)
        frequency_feature_time_domain.extend(get_fundamental_freq_form_time_domain(sample, frame_length,
                                                                                   pitch_window, frame_rate))

    return frequency_feature_time_domain


def get_fundamental_freq_form_time_domain(sample, frame_length, window, frame_rate):
    """
    Funkcja dla każdej rammki z sample, dłguośći frame_length przygotowuje fft i oblicza z tego częstotliwość bazową.
    Następnie tworzy listę częstotliwości bazowych.

    :param sample: lista sampli, z ktorych ma być obliczony wetor cech częstotliwości
    :param frame_length: Dłguosć ramki do fft
    :param window: Funkcja okna
    :param frame_rate: Czestotliwość samplowania

    :return: Lista czestotliwości bazowych
    :rtype: list
    """
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

    return fundamental_freq_array
