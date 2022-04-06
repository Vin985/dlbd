import datetime


def format_date_short(dates):
    return [x.strftime("%d-%m") for x in dates]


def label_date_from_julian(dates):
    res = [
        (datetime.datetime(2018, 1, 1) + datetime.timedelta(x)).strftime("%d-%m")
        for x in dates
    ]
    return res
