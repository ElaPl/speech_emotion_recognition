#!/usr/bin/python3.5

"""
.. module:: project_documentation
    :platform: OS X
    :synopsis: module illustrating how to document python source code

.. moduleauthor:: Patrick Kennedy <patkennedy79@gmail.com>
"""

from matplotlib import pyplot as plt
import sys
import pymysql
from warnings import filterwarnings

from hmm_main import hmm_main
from knn_main import knn_main
from voice_module import *
from helper_file import *

emotions = ["anger", "boredom", "happiness", "sadness"]


def draw_freq_histogram():
    train_set = build_file_set('Berlin_EmoDatabase/wav/' + sys.argv[1] + '/*.wav')
    frame_size = 512

    for i in range(0, len(train_set)):
        freq_vector = get_freq_vector(train_set[i][0], 512)
        print(freq_vector)
        print()
        sample_rate = get_sample_rate(train_set[i][0])

        bins = np.arange(0, len(freq_vector), 1)  # fixed bin size
        plt.plot(bins, freq_vector)
        plt.title('Fundamental freq of ' + train_set[i][0])
        plt.xlabel('time, frame_length= ' + str(sample_rate / frame_size))
        plt.ylabel('Frequency')

        plt.show()


def draw_energy_histogram():
    train_set = build_file_set('Berlin_EmoDatabase/wav/' + sys.argv[1] + '/*.wav')

    for i in range(0, len(train_set)):
        sample_rate = get_sample_rate(train_set[i][0])
        energy_vector = get_energy_vector(train_set[i][0], int(sample_rate / 4))

        bins = np.arange(0, len(energy_vector), 1)
        plt.plot(bins, energy_vector)
        plt.title('Energy histogram of ' + train_set[i][0])
        plt.xlabel('Time')
        plt.ylabel('Energy')

        plt.show()


if __name__ == "__main__":
    filterwarnings('ignore', category=pymysql.Warning)
    db_name = knn_db.DB_NAME
    db_password = "Mout"

    if len(sys.argv) == 1:
        print("Use either KNN or HMM option")

    if len(sys.argv) > 4:
        db_name = sys.argv[2]
        db_password = sys.argv[3]

    if sys.argv[1] == 'KNN':
        knn_main('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', db_name, db_password,
                 emotions)
    else:
        hmm_main('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', db_name, db_password,
                 emotions)

