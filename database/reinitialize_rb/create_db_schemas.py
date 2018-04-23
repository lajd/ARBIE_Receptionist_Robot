from tinydb import TinyDB, Query
from numpy import arange
from copy import deepcopy
from os.path import dirname, join
import datetime
import time


def initialize_appointment_schedule_db(path_to_db='/Users/Jonathan/PycharmProjects/ARBIE/database/databases/appointment_scheduling.json'):
    """
    Initializes db schema for 1 year time period (Jan-Dec)
    12 months. 30 days for each month accept [sept, april, june, november]. Feb has 28
    :param path_to_db: Path to where appointment schedules should be formed
    :return: Nothing
    """
    template_appointment_record = {'Timestamp':None,'Year':None, 'Month':None, 'Day':None, 'Time':None, 'Client':None, 'Doctor':None, 'Comment':None, 'Remark':None, 'Booked':False}

    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    months30 = ['September', 'April', 'June', 'November']
    month28 = ['February']

    day_time_range = (8, 18)  # 8AM until 6 AM, hours since midnight
    time_slot_size = 0.5  # Hours

    month_ints = range(len(months))

    # Database to populate
    db = TinyDB(path_to_db)

    years = range(2018, 2020, 1)

    for year in years:

        # Populate the months
        for month in months:
            if month in months30:
                days = 30
            elif month in month28:
                days = 28
            else:
                days = 31

            encode_month = month_ints[months.index(month)]

            # Populate the days
            for day in range(1, days+1):
                # Populate the times
                for time_slot_start in arange(day_time_range[0], day_time_range[1] + time_slot_size, time_slot_size):
                    record = deepcopy(template_appointment_record)

                    if '.5' in str(time_slot_start):
                        time_minutes = 30
                    else:
                        time_minutes = 0

                    dt = datetime.datetime(year, encode_month + 1, day, int(time_slot_start), time_minutes)
                    timestamp = time.mktime(dt.timetuple())

                    record['Timestamp'], record['Year'], record['Month'], record['Day'], record['Time'] = \
                        timestamp, year, encode_month, day, time_slot_start
                    db.insert(record)
                    print 'added record'


if __name__ == '__main__':
    pth = join(dirname(dirname(__file__)), 'databases/appointment_scheduling.json')
    initialize_appointment_schedule_db(pth)


