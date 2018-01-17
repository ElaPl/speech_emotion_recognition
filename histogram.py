from voice_module import get_freq_history
import matplotlib.pyplot as plt
import os


def show_freq_histogram(file_set):
    if file_set:
        for i in range(len(file_set)):
            path, filename = os.path.split(file_set[i])
            emotion = os.path.basename(path)
            freq_array = get_freq_history(file_set[i])
            plt.plot(freq_array, label=emotion)

        leg = plt.legend(loc='upper right', ncol=2, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)

        plt.show()