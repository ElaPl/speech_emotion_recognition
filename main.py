#!/usr/bin/python3.5

import sys
import pymysql
from warnings import filterwarnings

from hmm_main import hmm_main
from knn_main import knn_main
from mm_main import mm_main
import knn_database as knn_db

emotions = ["anger", "boredom", "happiness", "sadness"]

if __name__ == "__main__":
    filterwarnings('ignore', category=pymysql.Warning)
    db_name = knn_db.DB_NAME
    db_password = "Mout"

    if len(sys.argv) == 1:
        print("Use either KNN or HMM option")

    if len(sys.argv) > 2:
        db_name = sys.argv[2]
        db_password = sys.argv[3]

    if sys.argv[1] == 'KNN':
        knn_main('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', db_name, db_password,
                 emotions)
    elif sys.argv[1] == 'HMM':
        hmm_main('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', db_name, db_password,
                 emotions)
    else:
        mm_main('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', db_name, db_password,
                emotions)

