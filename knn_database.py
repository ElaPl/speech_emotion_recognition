import pymysql, sys

DB_NAME = 'speech_emotion_recognition'

DB_FEMALE_NAME = 'female_speech_emo'
DB_MALE_NAME = 'male_speech_emo'

pitch_knn_train_db_table = 'knn_pitch_train_vectors'
summary_pitch_knn_train_db_table = 'knn_summary_pitch_vectors'
energy_knn_train_db_table = 'knn_energy_train_vectors'

KNN_DB_TABLES = {}
KNN_DB_TABLES[pitch_knn_train_db_table] = {}
KNN_DB_TABLES[summary_pitch_knn_train_db_table] = {}
KNN_DB_TABLES[energy_knn_train_db_table] = {}

KNN_DB_TABLES[pitch_knn_train_db_table]['create'] = (
    "CREATE TABLE knn_pitch_train_vectors ("
    "vocal_range DOUBLE, "
    "max_freq DOUBLE, "
    "min_freq DOUBLE, "
    "avg_freq DOUBLE, "
    "dynamic_tones_freq DOUBLE, "
    "percent_of_falling_tones DOUBLE, "
    "percent_of_rising_tones DOUBLE, "
    "standard_deviation_freq DOUBLE, "
    "emotion TEXT);"
)

KNN_DB_TABLES[pitch_knn_train_db_table]['insert'] = (
    "INSERT INTO knn_pitch_train_vectors("
    "vocal_range,"
    "max_freq,"
    "min_freq,"
    "avg_freq,"
    "dynamic_tones_freq,"
    "percent_of_falling_tones,"
    "percent_of_rising_tones,"
    "standard_deviation_freq,"
    "emotion)"
    "VALUES (%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf, %s)"
)

KNN_DB_TABLES[summary_pitch_knn_train_db_table]['create'] = (
    "CREATE TABLE knn_summary_pitch_vectors ("
    "freq_range DOUBLE,"
    "max_freq DOUBLE,"
    "min_freq DOUBLE,"
    "avg_freq DOUBLE,"
    "dynamic_tones_freq DOUBLE,"
    "standard_deviation_freq DOUBLE,"
    "emotion TEXT);"
)

KNN_DB_TABLES[summary_pitch_knn_train_db_table]['insert'] = (
    "INSERT INTO knn_summary_pitch_vectors("
    "freq_range,"
    "max_freq,"
    "min_freq,"
    "avg_freq,"
    "dynamic_tones_freq,"
    "standard_deviation_freq,"
    "emotion)"
    "VALUES (%lf, %lf, %lf, %lf, %lf, %lf, %s);"
)

KNN_DB_TABLES[energy_knn_train_db_table]['create'] = (
    "CREATE TABLE knn_energy_train_vectors ("
    "standard_deviation_freq DOUBLE,"
    "crossing_rate DOUBLE,"
    "rms_db DOUBLE,"
    "peak_db DOUBLE,"
    "emotion TEXT);"
)

KNN_DB_TABLES[energy_knn_train_db_table]['insert'] = (
    "INSERT INTO knn_energy_train_vectors("
    "standard_deviation_freq,"
    "crossing_rate,"
    "rms_db,"
    "peak_db,"
    "emotion)"
    "VALUES (%lf, %lf, %lf, %lf, %s);"
)

pitch_hmm_train_db_table = 'hmm_pitch_observation_vectors'
energy_hmm_train_db_table = 'hmm_energy_observation_vectors'

HMM_DB_TABLES = {}
HMM_DB_TABLES[pitch_hmm_train_db_table] = {}
HMM_DB_TABLES[energy_hmm_train_db_table] = {}

HMM_DB_TABLES[pitch_hmm_train_db_table]['create'] = (
    "CREATE TABLE hmm_pitch_observation_vectors ("
    "vocal_range DOUBLE, "
    "max_freq DOUBLE, "
    "min_freq DOUBLE, "
    "avg_freq DOUBLE, "
    "dynamic_tones_freq DOUBLE, "
    "percent_of_falling_tones DOUBLE, "
    "percent_of_rising_tones DOUBLE, "
    "standard_deviation_freq DOUBLE);"
)

HMM_DB_TABLES[pitch_hmm_train_db_table]['insert'] = (
    "INSERT INTO hmm_pitch_observation_vectors("
    "vocal_range,"
    "max_freq,"
    "min_freq,"
    "avg_freq,"
    "dynamic_tones_freq,"
    "percent_of_falling_tones,"
    "percent_of_rising_tones,"
    "standard_deviation_freq)"
    "VALUES (%lf, %lf, %lf, %lf, %lf, %lf, %lf, %lf)"
)

HMM_DB_TABLES[energy_hmm_train_db_table]['create'] = (
    "CREATE TABLE hmm_energy_observation_vectors ("
    "standard_deviation_freq DOUBLE,"
    "crossing_rate DOUBLE,"
    "rms_db DOUBLE,"
    "peak_db DOUBLE);"
)

HMM_DB_TABLES[energy_hmm_train_db_table]['insert'] = (
    "INSERT INTO hmm_energy_observation_vectors ("
    "standard_deviation_freq,"
    "crossing_rate,"
    "rms_db,"
    "peak_db)"
    "VALUES (%lf, %lf, %lf, %lf);"
)

def prepare_db_table(db, cursor, table):
    for name, ddl in table.items():
        try:
            cursor.execute("DROP TABLE IF EXISTS %s;" % name)
            db.commit()
            cursor.execute(ddl['create'])
            db.commit()
        except pymysql.Error as e:
            db.rollback()
            sys.exit('[ERROR] %d: %s\n' % (e.args[0], e.args[1]))


def is_training_set_exists(cursor, table):
    for name in table.keys():
        try:
            cursor.execute("SELECT * FROM %s;" % name)
        except pymysql.Error as e:
            return False

        result = cursor.fetchall()
        if len(result) == 0:
            return False

    return True


def select_all_from_db(cursor, table_name):
    try:
        cursor.execute("SELECT * FROM %s;" % table_name)
    except pymysql.Error as e:
        sys.exit('[ERROR] % d: % s\n' % (e.args[0], e.args[1]))

    result = cursor.fetchall()
    return result


def save_in_dbtable(db, cursor, vect, tbname):
    try:

        if tbname == pitch_knn_train_db_table:
            cursor.execute(KNN_DB_TABLES[tbname]['insert'] % (vect[0], vect[1], vect[2], vect[3], vect[4], vect[5],
                                                              vect[6], vect[7], "'" + vect[8] + "'"))
        elif tbname == summary_pitch_knn_train_db_table:
            cursor.execute(KNN_DB_TABLES[tbname]['insert'] % (vect[0], vect[1], vect[2], vect[3], vect[4], vect[5],
                                                              "'" + vect[6] + "'"))
        elif tbname == energy_knn_train_db_table:
            cursor.execute(KNN_DB_TABLES[tbname]['insert'] % (vect[0], vect[1], vect[2], vect[3], "'" + vect[4] + "'"))
        elif tbname == energy_hmm_train_db_table:
            cursor.execute(HMM_DB_TABLES[tbname]['insert'] % (vect[0], vect[1], vect[2], vect[3]))
        else:
            cursor.execute(HMM_DB_TABLES[tbname]['insert'] % (vect[0], vect[1], vect[2], vect[3], vect[4], vect[5],
                                                              vect[6], vect[7]))
        db.commit()
    except pymysql.Error as e:
        sys.exit('[ERROR] % d: % s\n' % (e.args[0], e.args[1]))
