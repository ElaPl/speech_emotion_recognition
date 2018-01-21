#!/usr/bin/python3.5

import sys
import pymysql
from warnings import filterwarnings

from hmm_main import hmm_main
from knn_main import knn_main
from mm_main import mm_main
import knn_database as knn_db
from FeatureImportance import feature_importance
from histogram import show_freq_histogram, show_energy_histogram, show_subplot_histogram

emotions = ["anger", "boredom", "happiness", "sadness"]

if __name__ == "__main__":
    filterwarnings('ignore', category=pymysql.Warning)
    db_name = knn_db.DB_NAME
    db_password = "Mout"

    if len(sys.argv) == 4:
        db_name = sys.argv[2]
        db_password = sys.argv[3]

    if sys.argv[1] == 'KNN':
        knn_main('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', db_name, db_password,
                 emotions)
    elif sys.argv[1] == 'HMM':
        hmm_main('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', db_name, db_password,
                 emotions)
    elif sys.argv[1] == 'MM':
        mm_main('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', db_name, db_password,
                emotions)
    elif sys.argv[1] == 'FI':
        feature_importance('Berlin_EmoDatabase/train/*/*/*.wav', db_name, db_password)
    elif sys.argv[1] == 'FH':
        show_freq_histogram(sys.argv[2:])
    elif sys.argv[1] == 'EH':
        show_energy_histogram(sys.argv[2:])
    elif sys.argv[1] == 'AS':
        show_subplot_histogram(['Berlin_EmoDatabase/train/male/anger/03b01Wa.wav',
                                'Berlin_EmoDatabase/train/male/sadness/12b01Ta.wav',
                                'Berlin_EmoDatabase/train/male/boredom/03b01Lb.wav',
                                'Berlin_EmoDatabase/train/male/happiness/03b01Fa.wav'])
    else:
        print("Nieznana komenda. Dozwolone komendy KNN, HMM, MM")
