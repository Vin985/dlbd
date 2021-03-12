import datetime


def join_tuple(to_join, sep):
    to_join = list(filter(None, to_join))
    if len(to_join) > 1:
        return sep.join(to_join)
    return to_join[0]


def exclude_rows(data, options):
    for key, value in options.items():
        if key.startswith("exclude_"):
            column = key.split("_", 1)[1]
            data = data.loc[~data[column].isin(value)]
    return data


def get_rows(data, options, include=True):
    for column, value in options.items():
        if include:
            data = data.loc[data[column].isin(value)]
        else:
            data = data.loc[~data[column].isin(value)]
    return data


def label_x(dates):
    res = [(datetime.datetime(2018, 1, 1) + datetime.timedelta(x)).strftime("%d-%m") for x in dates]
    print(res)
    return res
