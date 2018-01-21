from voice_module import get_freq_history, get_energy_history
import matplotlib.pyplot as plt
import os
import numpy as np

def show_freq_histogram(file_set):
    if file_set:
        for i in range(len(file_set)):
            path, filename = os.path.split(file_set[i])
            emotion = os.path.basename(path)
            freq_array = get_freq_history(file_set[i])
            plt.plot(freq_array, label=emotion)

        plt.title("Pitch")
        leg = plt.legend(loc='upper right', ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.show()


def show_energy_histogram(file_set):
    if file_set:
        for i in range(len(file_set)):
            path, filename = os.path.split(file_set[i])
            emotion = os.path.basename(path)
            energy_array = get_energy_history(file_set[i])
            plt.plot(energy_array, label=emotion)

        plt.title("Energy")
        leg = plt.legend(loc='upper right', ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.show()


def show_subplot_histogram(file_set):
    if file_set:
        columns = len(file_set)
        f, axarr = plt.subplots(2, columns)
        for i in range(len(file_set)):
            path, filename = os.path.split(file_set[i])
            emotion = os.path.basename(path)

            freq_array = get_freq_history(file_set[i])
            x = [i for i in range(0, len(freq_array))]
            print(x)
            print(len(x))
            print(len(freq_array))
            print(freq_array)

            axarr[0, i].plot(x, freq_array, label=emotion)
            axarr[0, i].set_title(emotion)

            energy_array = get_energy_history(file_set[i])
            x = [i for i in range(0, len(energy_array))]
            axarr[1, i].plot(x, energy_array, label=emotion)

    plt.show()


