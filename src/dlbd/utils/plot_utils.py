import datetime


def format_date_short(dates):
    if (dates[len(dates) - 1] - dates[0]).days > 1:
        return [x.strftime("%d-%m") for x in dates]
    else:
        return [x.strftime("%H:%M") for x in dates]


def label_date_from_julian(dates):
    res = [
        (datetime.datetime(2018, 1, 1) + datetime.timedelta(x)).strftime("%d-%m")
        for x in dates
    ]
    return res
