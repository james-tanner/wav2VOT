import parselmouth
import pandas as pd
import random
import shutil
import soundfile as sf
import multiprocessing
import yaml
import textgrid
import re

from argparse import ArgumentParser
from itertools import repeat
from librosa import load, resample
from pathlib import Path
from tqdm import tqdm

from scripts.prepare_wavs import resample_wav

def get_textgrid(tgpath):
    tg = textgrid.TextGrid()
    tg.read(str(tgpath))
    return tg

def get_stop_regex(labels):
    stop_re = re.compile(labels)
    return stop_re

def extract_intervals(tg, stop_regex, tiername_regex):
    
    stop_intervals = []

    for t in tg.getNames():
        tiername_found = re.search(tiername_regex, t)
        if tiername_found:

            for tier in tg.getList(t):
                for interval in tier:
                    found_labels = re.search(stop_regex, interval.mark)
                    if found_labels:
                        stop_intervals.append(interval)

    return stop_intervals

def extract_stops(tgpath, labels, tiername, output_dir, lmin, lmax, rmin, rmax, sample_rate):
    tg_name = tgpath.stem
    tg = get_textgrid(tgpath)
    intervals = extract_intervals(tg, labels, tiername)
    
    stop_list = []
    for i, interval in enumerate(intervals):
        obs_name = f"{tg_name}_{str(i)}"

        ## pull winodow sizes from distributions
        lwindow = round(random.uniform(lmin, lmax), ndigits = 3)
        rwindow = round(random.uniform(rmin, rmax), ndigits = 3)

        ## get stop info from TextGrid
        stop_start = interval.minTime
        stop_end = interval.maxTime
        stop_dur = stop_end - stop_start

        ## add window to timestamps
        slice_start = stop_start - lwindow
        slice_end = stop_end + rwindow

        ## extract audio based on timestamps
        sound_path = tgpath.parents[0].joinpath(tg_name).with_suffix(".wav")
        audio = parselmouth.Sound(str(sound_path))
        stop_audio = audio.extract_part(from_time = slice_start, to_time = slice_end)
        stop_output_path = output_dir.joinpath("wavs").joinpath(obs_name).with_suffix(".wav")
        stop_audio.save(file_path = str(stop_output_path), format = "WAV")
        resample_wav(stop_output_path, sample_rate)

        obs_dict = {
            "filename" : obs_name,
            "stop_label" : interval.mark,
            "stop_start" : lwindow,
            "stop_end" : lwindow + stop_dur,
            "left_window" : lwindow,
            "right_window" : rwindow,
            "window_start" : slice_start,
            "window_end" : slice_end
        }
        stop_list.append(obs_dict)

    stop_df = pd.DataFrame.from_records(stop_list)
    return stop_df        


if __name__ == "__main__":
    parser = ArgumentParser()

    ## required args
    parser.add_argument("input_dir", type = Path, help = "Path to directory containing wav/textgrids")
    parser.add_argument("stop_labels", type = str, help = "Regex of stop labels to extract (This works best as a restricted set of one character eg. ^[ptkbdgPTKBDG]$)")
    parser.add_argument("output_dir", type = Path, help = "Path for writing stop dataset")

    ## tiername
    parser.add_argument("--tiername", type = str, nargs = "?", default = "phones", help = "Substring indicating tier containing stops (default: phones)")

    ## window settings
    parser.add_argument("--left_window_min", type = float, nargs = "?", default = 0.03, help = "Minimum size of left-edge window (in seconds, default = 0.03)")
    parser.add_argument("--left_window_max", type = float, nargs = "?", default = 0.03, help = "Maximum size of the left-edge window (in seconds, default = 0.03)")
    parser.add_argument("--right_window_min", type = float, nargs = "?", default = 0.03, help = "Minimum size of right-edge window (in seconds, default = 0.03)")
    parser.add_argument("--right_window_max", type = float, nargs = "?", default = 0.03, help = "Maximum size of the right-edge window (in seconds, default = 0.03)")

    ## misc
    parser.add_argument("--seed", type = int, nargs = "?", default = None, help = "Random seed for window size sampling (default: None)")
    parser.add_argument("--sample_rate", type = int, nargs = "?", default = 16000, help = "Sample rate to use for wav resampling (default: 16000)")
    parser.add_argument("--clean", "-c", action = "store_true", help = "Clean output directory before run (default = False)")
    parser.add_argument("--num_jobs", "-j", type = int, nargs = "?", default = 8, help = "Number of multiprocessing jobs (default = 8)")
    args = parser.parse_args()
    args = parser.parse_args()

    ## get list of textgrids
    textgrids = list(args.input_dir.rglob("*.TextGrid"))

    ## make the output directory if it doesn't already exist
    args.output_dir.mkdir(parents = True, exist_ok = True)

    if args.clean:
        print("Cleaning output directory...")
        shutil.rmtree(args.output_dir)

    args.output_dir.joinpath("wavs").mkdir(parents = True, exist_ok = True)

    stop_re = get_stop_regex(args.stop_labels)

    with multiprocessing.get_context("spawn").Pool(args.num_jobs) as pool:
        stop_dfs = list(pool.starmap(
            extract_stops,
            tqdm(
                zip(textgrids,
                    repeat(stop_re),
                    repeat(args.tiername),
                    repeat(args.output_dir),
                    repeat(args.left_window_min),
                    repeat(args.left_window_max),
                    repeat(args.right_window_min),
                    repeat(args.right_window_max),
                    repeat(args.sample_rate)
                ),
                total = len(textgrids))
        ))
    window_df = pd.concat(stop_dfs)

    ## write window df
    window_df_path = args.output_dir.joinpath("window_data").with_suffix(".csv")
    window_df.to_csv(str(window_df_path), index = False, header = True)

    data_df = window_df[['filename', 'stop_start', 'stop_end']]
    data_df_path = args.output_dir.joinpath("data").with_suffix(".csv")
    data_df.to_csv(str(data_df_path), index = False, header = True)

    print("Done!")