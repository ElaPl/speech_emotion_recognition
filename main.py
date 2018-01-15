#!/usr/bin/python3.5

import sys
import pymysql
from warnings import filterwarnings

from hmm_main import hmm_main
from knn_main import knn_main
from helper_file import *

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
    else:
        hmm_main('Berlin_EmoDatabase/train/*/*/*.wav', 'Berlin_EmoDatabase/test/*/*/*.wav', db_name, db_password,
                 emotions)

