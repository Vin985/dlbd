import os
import functools
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import datetime

METHODS = ["standard", "subsampling"]


def predictions2pdf(predictions, recording):
    fig = plt.figure(figsize=(15, 5))

    # plot spectrogram
    sp1 = fig.add_subplot(211)
    librosa.display.specshow(recording.create_spectrogram().spec)
    # plot activity
    sp2 = fig.add_subplot(212)
    sp2.plot(predictions["time"], predictions["activity"],
             'g', label='biotic activity')
    sp2.set_xlim([0, max(predictions["time"])])
    # fig.xlabel('Time (s)')
    #sp2.ylabel('Activity level')
    # fig.legend()

    # Save plots
    save_dir = 'plots/pdf/'
    print(os.getcwd())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(save_dir + recording.name + "_events.pdf")


def detect_events_standard(predictions, recording_id=-1, detection_options=None):
    print("standard detection")
    min_activity = detection_options.get("min_activity", 0.85)
    min_duration = detection_options.get("min_duration", 0.1)
    end_threshold = detection_options.get("end_threshold", 0.6)
    event_id = 0
    ongoing = False
    events = []
    start = 0
    end = 0
    # def detect_songs_events(predictions):
    for pred_time, activity in predictions.itertuples(index=False):
        # Check if prediction is above a defined threshold
        if activity > min_activity:
            # If not in a song, create a new event
            if not ongoing:
                ongoing = True
                event_id += 1
                start = pred_time
        elif ongoing:
            # if above an end threshold, consider it as a single event
            if activity > end_threshold:
                continue
            # If below the threshold and in an active event, end it
            ongoing = False
            end = pred_time
            # log event if its duration is greater than minimum threshold
            if (end - start) > min_duration:
                events.append({"event_id": event_id, "recording_id": recording_id,
                               "start": start, "end": end})
    events = pd.DataFrame(events)
    return events


def resample_max(x, threshold=0.98, mean_thresh=0):
    if any(x >= threshold) and x.mean() > mean_thresh:
        return 2
    return 0


def has_tag(x):
    if any(x) > 0:
        return 1
    return 0


def isolate_events_subsampling(predictions, step):
    tmp = predictions.loc[predictions.event > 0]
    tmp.reset_index(inplace=True)

    step = datetime.timedelta(milliseconds=step)
    start = None
    events = []
    event_id = 1
    if len(tmp):
        for _, x in tmp.iterrows():
            if not start:
                prev_time = x.datetime
                start = prev_time
                continue
            diff = x.datetime - prev_time
            if diff > step:
                end = prev_time + step
                events.append({"event_id": event_id, "recording_id": x.recording_id,
                               "start": start.timestamp(), "end": end.timestamp()})
                event_id += 1
                start = x.datetime
            prev_time = x.datetime

        end = prev_time + step
        events.append({"event_id": event_id, "recording_id": x.recording_id,
                       "start": start.timestamp(), "end": end.timestamp()})

    events = pd.DataFrame(events)
    return events


# def detect_events_subsampling(predictions, recording_id=-1, detection_options=None):
#     preds = predictions.copy()
#     if not "tag" in preds.columns:
#         preds.loc[:, "tag"] = -1
#     preds.loc[:, "event"] = -1
#     preds.loc[:, "datetime"] = pd.to_datetime(preds.time * 10**9)
#     preds.set_index("datetime", inplace=True)

#     min_activity = detection_options.get("min_activity", 0.85)
#     step = detection_options.get("min_duration", 0.1) * 1000
#     isolate_events = detection_options.get("isolate_events", False)

#     resampled = preds.resample(str(step)+"ms")
#     resample_func = functools.partial(resample_max, threshold=min_activity)
#     res = resampled.agg({"activity": resample_func,
#                          "tag"has_tag,
#                          "tag_index": self.get_tag_index})
#     res.rename(columns={"activity": "event"}, inplace=True)
#     res["recording_id"] = recording_id

#     if isolate_events:
#         return isolate_events_subsampling(res, step)

#     return res


# METHODS_FUNCTIONS = {"standard": detect_events_standard,
#                      "subsampling": detect_events_subsampling}


# def detect_songs_events(predictions, recording_id=-1, detection_options=None):
#     predictions = predictions[["time", "activity"]]
#     detection_options = detection_options or {}
#     method = detection_options.get("method", METHODS[0])
#     method_func = METHODS_FUNCTIONS[method]
#     return method_func(predictions, recording_id, detection_options)
