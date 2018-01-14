import pymysql, sys

DB_NAME = 'speech_emotion_recognition'
DB_FEMALE_NAME = 'female_speech_emo'
DB_MALE_NAME = 'male_speech_emo'
pitch_train_set_name = 'pitch_train_set'
summary_pitch_train_set_name = 'summary_pitch_train_set'
energy_train_set_name = 'energy_train_set'

KNN_DB_TABLES = {}
KNN_DB_TABLES[pitch_train_set_name] = {}
KNN_DB_TABLES[summary_pitch_train_set_name] = {}
KNN_DB_TABLES[energy_train_set_name] = {}

KNN_DB_TABLES[pitch_train_set_name]['create'] = (
    "CREATE TABLE pitch_train_set ("
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

KNN_DB_TABLES[pitch_train_set_name]['insert'] = (
    "INSERT INTO pitch_train_set("
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

KNN_DB_TABLES[summary_pitch_train_set_name]['create'] = (
    "CREATE TABLE summary_pitch_train_set ("
    "freq_range DOUBLE,"
    "max_freq DOUBLE,"
    "min_freq DOUBLE,"
    "avg_freq DOUBLE,"
    "dynamic_tones_freq DOUBLE,"
    "standard_deviation_freq DOUBLE,"
    "emotion TEXT);"
)

KNN_DB_TABLES[summary_pitch_train_set_name]['insert'] = (
    "INSERT INTO summary_pitch_train_set("
    "freq_range,"
    "max_freq,"
    "min_freq,"
    "avg_freq,"
    "dynamic_tones_freq,"
    "standard_deviation_freq,"
    "emotion)"
    "VALUES (%lf, %lf, %lf, %lf, %lf, %lf, %s);"
)

KNN_DB_TABLES[energy_train_set_name]['create'] = (
    "CREATE TABLE energy_train_set ("
    "standard_deviation_freq DOUBLE,"
    "crossing_rate DOUBLE,"
    "rms_db DOUBLE,"
    "peak_db DOUBLE,"
    "emotion TEXT);"
)

KNN_DB_TABLES[energy_train_set_name]['insert'] = (
    "INSERT INTO energy_train_set("
    "standard_deviation_freq,"
    "crossing_rate,"
    "rms_db,"
    "peak_db,"
    "emotion)"
    "VALUES (%lf, %lf, %lf, %lf, %s);"
)


def create_training_set(db, cursor):
    for name, ddl in KNN_DB_TABLES.items():
        try:
            cursor.execute("DROP TABLE IF EXISTS %s;" % name)
            db.commit()
            cursor.execute(ddl['create'])
            db.commit()
        except pymysql.Error as e:
            db.rollback()
            sys.exit('[ERROR] %d: %s\n' % (e.args[0], e.args[1]))


def is_training_set_exists(cursor):
    for name in KNN_DB_TABLES.keys():
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


def save_in_dbtable(db, cursor, vect, emotion, tbname):
    try:
        if tbname == pitch_train_set_name:
            cursor.execute(KNN_DB_TABLES[tbname]['insert'] % (vect[0], vect[1], vect[2], vect[3], vect[4], vect[5],
                                                              vect[6], vect[7], "'" + emotion + "'"))
        elif tbname == summary_pitch_train_set_name:
            cursor.execute(KNN_DB_TABLES[tbname]['insert'] % (vect[0], vect[1], vect[2], vect[3], vect[4], vect[5],
                                                              "'" + emotion + "'"))
        else:
            cursor.execute(KNN_DB_TABLES[tbname]['insert'] % (vect[0], vect[1], vect[2], vect[3],
                                                              "'" + emotion + "'"))

        db.commit()
    except pymysql.Error as e:
        sys.exit('[ERROR] % d: % s\n' % (e.args[0], e.args[1]))
