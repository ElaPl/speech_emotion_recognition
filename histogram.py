from voice_module import get_freq_history, get_energy_history
import matplotlib.pyplot as plt
import os


def show_subplot_histogram(file_set):
    """
    Funkcja wyświetla przebieg częstotliwości bazowych, oraz wartości natężenia dla każdego
    z plików podanych jako argument
    :param file_set: lista plików
    :return: None
    """
    if file_set:
        columns = len(file_set)
        f, axarr = plt.subplots(2, columns)
        for i in range(len(file_set)):
            path, filename = os.path.split(file_set[i])
            emotion = os.path.basename(path)

            freq_array = get_freq_history(file_set[i])
            x = [i for i in range(0, len(freq_array))]

            axarr[0, i].plot(x, freq_array, label=emotion)
            axarr[0, i].set_title(emotion)

            energy_array = get_energy_history(file_set[i])
            x = [i for i in range(0, len(energy_array))]
            axarr[1, i].plot(x, energy_array, label=emotion)

    plt.show()


