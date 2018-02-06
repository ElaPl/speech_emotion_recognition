import sys
import sqlite3 as lite

DB_NAME = 'speech_emotion_recognition'

knn_train_db_table = 'knn_train_set'

KNN_DB_TABLES = {}
KNN_DB_TABLES[knn_train_db_table] = {}

KNN_DB_TABLES[knn_train_db_table]['create'] = (
    "CREATE TABLE knn_train_set ("
    "vocal_range DOUBLE, "
    "max_freq DOUBLE, "
    "min_freq DOUBLE, "
    "avg_freq DOUBLE, "
    "percent_of_falling_tones DOUBLE, "
    "percent_of_rising_tones DOUBLE, "
    "relative_standard_deviation_freq DOUBLE, "
    "relative_std_deviation_energy DOUBLE,"
    "zero_crossing_rate DOUBLE,"
    "rms_db DOUBLE, "
    "peaK_db DOUBLE,"
    "emotion TEXT);"
)

KNN_DB_TABLES[knn_train_db_table]['insert'] = (
    "INSERT INTO knn_train_set("
    "vocal_range,"
    "max_freq,"
    "min_freq,"
    "avg_freq,"
    "percent_of_falling_tones,"
    "percent_of_rising_tones,"
    "relative_standard_deviation_freq,"
    "relative_std_deviation_energy,"
    "zero_crossing_rate,"
    "rms_db, "
    "peaK_db,"
    "emotion)"
    "VALUES (%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %s)"
)

hmm_observation_db_table = 'hmm_observation_vectors'

HMM_DB_TABLES = {}
HMM_DB_TABLES[hmm_observation_db_table] = {}

HMM_DB_TABLES[hmm_observation_db_table]['create'] = (
    "CREATE TABLE hmm_observation_vectors ("
    "vocal_range DOUBLE, "
    "max_freq DOUBLE, "
    "min_freq DOUBLE, "
    "avg_freq DOUBLE, "
    "percent_of_falling_tones DOUBLE, "
    "percent_of_rising_tones DOUBLE, "
    "relative_standard_deviation_freq DOUBLE, "
    "relative_std_deviation_energy DOUBLE,"
    "zero_crossing_rate DOUBLE,"
    "rms_db DOUBLE, "
    "peaK_db DOUBLE);"
)

HMM_DB_TABLES[hmm_observation_db_table]['insert'] = (
    "INSERT INTO hmm_observation_vectors("
    "vocal_range,"
    "max_freq,"
    "min_freq,"
    "avg_freq,"
    "percent_of_falling_tones,"
    "percent_of_rising_tones,"
    "relative_standard_deviation_freq,"
    "relative_std_deviation_energy,"
    "zero_crossing_rate,"
    "rms_db, "
    "peaK_db)"
    "VALUES (%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf)"
)


def connect_to_database(db_name):
    """Funckja łączy się z bazą danych jako root

    :param db_name: nazwa bazy danych
    :type str
    :param db_password" hasło do bazy danych
    :type str

    :return:
        * db, cursor - jeżeli połączenie zostało nawiązane
        * None, None - w przeciwnym przypadku
    """
    try:
        con = lite.connect(db_name)
        cur = con.cursor()
        return con, cur

    except lite.Errora as e:
        return None, None


def prepare_db_table(db, cursor, table_names):
    """Tworzy w bazie danych tablice o podanych nazwach

    :param db: obiekt połączenia, który reprezentuje bazę danych
    :param cursor: kursor
    :param list table_names: lista nazw tablic, do utworzenia
    """
    for name, ddl in table_names.items():
        try:
            cursor.execute("DROP TABLE IF EXISTS %s;" % name)
            db.commit()
            cursor.execute(ddl['create'])
            db.commit()
        except lite.Error as e:
            db.rollback()
            sys.exit('[ERROR] %d: %s\n' % (e.args[0], e.args[1]))


def is_training_set_exists(cursor, table):
    """Funkcja sprawdza, czy wszystkie tablice podane w table istnieją.

    :param cursor: kursor
    :param dictionary table: tablica, której kluczami są nazwy tablic, których istnienie należy sprawdzić


    :return True - jeżeli wszystkie tablice w table.kesy istnieją
            False - w.p.p."""
    for name in table.keys():
        try:
            cursor.execute("SELECT * FROM %s;" % name)
        except lite.Error as e:
            return False

        result = cursor.fetchall()
        if len(result) == 0:
            return False

    return True


def select_all_from_db(cursor, table_name):
    """Funckja pobiera z bazy danych z tablicy table_name wszystkie wartości i zwraca w postaci listy wektorów.

    :param cursor: kursonr
    :param string table_name: Nazwa tablicy, z której mają być pobrane dane
    """
    try:
        cursor.execute("SELECT * FROM %s;" % table_name)
    except lite.Error as e:
        sys.exit('[ERROR] % d: % s\n' % (e.args[0], e.args[1]))

    result = cursor.fetchall()
    return result


def save_in_dbtable(db, cursor, vect, tbname):
    """Funkcja zapisuje w bazie danych w tablicy tbname, wartości z wektora vect

    :param db: obiekt połączenia, który reprezentuje bazę danych
    :param cursor: kursor
    :param vector vect: wektor który ma być zapisany w bazie danych
    :param string tbname: nazwa tablicy w bazie danych, do ktorej mają być zapisane wartości z wektora
    """
    try:

        if tbname == knn_train_db_table:
            cursor.execute(KNN_DB_TABLES[tbname]['insert'] % (vect[0], vect[1], vect[2], vect[3], vect[4], vect[5],
                                                              vect[6], vect[7], vect[8], vect[9], vect[10],
                                                              "'" + vect[11] + "'"))
        elif tbname == hmm_observation_db_table:
            cursor.execute(HMM_DB_TABLES[tbname]['insert'] % (vect[0], vect[1], vect[2], vect[3], vect[4], vect[5],
                                                              vect[6], vect[7], vect[8], vect[9], vect[10]))
        else:
            print("Unknown table name")
        db.commit()
    except lite.Error as e:
        sys.exit('[ERROR] % d: % s\n' % (e.args[0], e.args[1]))
