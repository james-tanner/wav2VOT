import librosa
import librosa.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from textgrid import TextGrid, IntervalTier, Interval

def plot_spectrogram_with_intervals(audio, preds, save_path, sr = 16000, n_fft = 256, hop_length = 10, label_names = None, label_colors = None, alpha = 0):
    '''
    Plot a spectrogram of a stop token
    overlaid with boundaries corresponding
    to the start/end of annotations
    '''
    if label_names is None:
        label_names = {0: "Window", 1: "Closure", 2: "VOT", 3: "Lenition"}
    if label_colors is None:
        label_colors = {
            0: "grey",
            1: "skyblue",
            2: "orange",
            3: "yellow"
        }

    duration = librosa.get_duration(y = audio, sr = sr)

    ## make spectrogram
    S = librosa.feature.melspectrogram(
        y = audio,
        sr = sr,
        n_fft = n_fft,
        win_length = 80,
        hop_length = hop_length,
        n_mels = 80
    )
    S_dB = librosa.power_to_db(S, ref = np.max)

    frame_times = np.linspace(0, duration, num = len(preds))

    ## global font size
    mpl.rcParams["font.size"] = 20

    ## define plot
    plt.figure(figsize=(12, 7), dpi = 600)
    librosa.display.specshow(
        S_dB,
        sr = sr,
        hop_length = hop_length,
        x_axis = 'time',
        y_axis = 'mel',
        cmap = 'viridis'
    )

    ## remove bar
    cb = plt.colorbar(format = '%+2.0f dB')
    cb.remove()

    ## add axis labels
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    ## get y-axis upper limit
    y_max = plt.gca().get_ylim()[1]

    prev_label = preds[0]
    start_time = frame_times[0]

    for i in range(1, len(preds)):
        if preds[i] != prev_label:
            end_time = frame_times[i]
            color = label_colors[prev_label]

            ## add overlay
            plt.axvspan(start_time, end_time, alpha = alpha, color = color)

            ## add boundaries
            plt.axvline(
                x = end_time,
                color = 'white',
                linestyle = '--',
                linewidth = 5)

            # add labels
            mid_time = (start_time + end_time) / 2
            plt.text(
                mid_time,
                y_max,
                label_names.get(prev_label, str(prev_label)),
                ha = 'center',
                va = 'bottom',
                fontsize = 20,
                color = 'black')

            start_time = end_time
            prev_label = preds[i]

    ## add last interval and label
    end_time = frame_times[-1]
    color = label_colors[prev_label]
    plt.axvspan(start_time, end_time, alpha = alpha, color = color)

    mid_time = (start_time + end_time) / 2
    plt.text(
        mid_time,
        y_max,
        label_names.get(prev_label, str(prev_label)),
        ha = 'center',
        va = 'bottom',
        fontsize = 20,
        color = 'black')

    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()

def extract_intervals(frames, audio_duration_sec):
    '''
    Convert a list of framewise labels back to
    timestamps, given an audio file's duration
    and sample rate
    '''
    num_frames = len(frames)
    frame_duration_ms = (audio_duration_sec / num_frames) * 1000

    boundaries = []
    prev_label = frames[0]
    start_idx = 0

    for i in range(1, num_frames):
        if frames[i] != prev_label:
            start_time = start_idx * frame_duration_ms
            end_time = i * frame_duration_ms
            boundaries.append((prev_label, start_time, end_time))
            start_idx = i
            prev_label = frames[i]

    # add final segment
    end_time = num_frames * frame_duration_ms
    boundaries.append((prev_label, start_idx * frame_duration_ms, end_time))

    return boundaries

def make_textgrid(timestamps, dur, tgpath, label_names = None, tier_name = "prediction"):
    '''
    Create a Praat TextGrid object
    containing predicted annotation intervals
    '''
    ## initialise interval tier
    tier = IntervalTier(
        name = tier_name,
        minTime = 0,
        maxTime = dur
    )

    for timestamp in timestamps:
        tm = (timestamp['label'], timestamp['start'], timestamp['end'])
        interval = Interval(
            ## convert to ms
            minTime = tm[1],
            maxTime = tm[2],
            ## add label
            mark = tm[0]
        )

        ## if the interval is too large
        ## for the tier, extend tier length
        try:
            tier.addInterval(interval)
        except ValueError:
            tier.maxTime = interval.maxTime
            tier.addInterval(interval)

    tg = TextGrid()
    tg.append(tier)

    tg.write(tgpath)

    return None

def get_input_data(fpath, default = "data.csv"):
    '''
    Convenience function for
    getting default name of
    dataset (data.csv)
    '''
    if fpath.suffix == ".csv":
        return fpath.parent, fpath.name
    else:
        return fpath, default

def get_output_lengths(length, kernel_sizes, strides):
    ## scale length in layer-wise fashion
    for k, s in zip(kernel_sizes, strides):
        pad = k // 2
        out = (length + 2 * pad - (k - 1) - 1) // s + 1
    return np.ceil(out)

def convert_to_intervals(predictions, duration):
    '''
    Get the start/end timestamps based on the
    converted intervals
    '''
    intervals = extract_intervals(predictions, audio_duration_sec = duration)

    label_names = {0: "Window", 1: "Closure", 2: "VOT", 3: "Lenition"}
    interval_list = [
        {
            "label" : label_names[interval[0].item()],
            "start" : interval[1]/1000,
            "end" : interval[2]/1000
        }
        for interval in intervals]
    return interval_list

def make_interval_dict(intervals, filename):
    '''
    Given a set of intervals, create a dictionary
    containing the start/end time of each interval
    '''
    interval_dict = {}

    interval_dict['filename'] = filename
    for interval in intervals:
        if interval['label'] not in ['Window', ""]:
            if interval['label'] == "Lenition":
                interval_dict['stop_start'], interval_dict['stop_end'] = interval['start'], interval['end']
                interval_dict['lenit'] = 1

            elif interval['label'] == "VOT":
                interval_dict['stop_start'], interval_dict['stop_end'] = interval['start'], interval['end']
                interval_dict['lenit'] = 0

            elif interval['label'] == "Closure":
                interval_dict['closure_start'], interval_dict['closure_end'] = interval['start'], interval['end']

            else:
                raise ValueError(f"Label {interval['label']} is not a valid label")

    ## finally get label sequence
    interval_dict['sequence'] = " ".join([i['label'][0] for i in intervals if i['label'] != ""])

    return interval_dict

def correct_window_offsets(interval_dict, window, start):
    '''
    Reset the timestamps based on the offsets from
    the context window
    '''
    if 'closure_start' in interval_dict:
        interval_dict['closure_start'] = (interval_dict['closure_start'] - window) + start
    if 'closure_end' in interval_dict:
        interval_dict['closure_end'] = (interval_dict['closure_end'] - window) + start

    if 'stop_start' in interval_dict:
        interval_dict['stop_start'] = (interval_dict['stop_start'] - window) + start
    if 'stop_end' in interval_dict:
        interval_dict['stop_end'] = (interval_dict['stop_end'] - window) + start
    interval_dict['left_window'] = window

    return interval_dict
