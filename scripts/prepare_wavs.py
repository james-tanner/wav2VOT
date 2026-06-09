import parselmouth
import pandas as pd
import random
import shutil
import soundfile as sf
import multiprocessing
import time
import yaml

from argparse import ArgumentParser
from itertools import repeat
from librosa import load, resample
from pathlib import Path
from tqdm import tqdm

def make_subdirs(indir):
    '''
    Makes the subdirectories for saving wavs and textgrids
    '''
    indir.joinpath("wavs").mkdir(parents = True, exist_ok = True)

    return None

def get_wav_from_corpus(corpus, wav_name):
    '''
    Create parselmouth object from given
    audio path
    '''
    wav_fpath = corpus.joinpath(wav_name)
    wav_fpath = str(wav_fpath)
    return parselmouth.Sound(wav_fpath)

def resample_wav(wav_fpath, sr):
    '''
    Return wav at new sample rate
    '''
    wav_in, orig_sr = load(str(wav_fpath))
    wav_in = resample(wav_in, orig_sr = orig_sr, target_sr = sr)
    sf.write(str(wav_fpath), wav_in, samplerate = sr)

    return None

def make_wav_clips(wav, wav_label, corpus_dir, out_dir, start, end, lmin, lmax, rmin, rmax, o_id, sample_rate):
    '''
    Main processing function:
    Given the name of an audio file, extract:
    - Sliced audio at given time points (+ window)
    - Time points of stop annotations
    '''
    ## get discourse and speaker names as strings
    talkid_name = wav[wav_label].unique()[0]
    try:
        wav_path = list(corpus_dir.rglob(f"*{talkid_name}*.[Ww][Aa][Vv]"))[0]
    except IndexError:
        print(f"Could not find {corpus_dir.joinpath(talkid_name)}")
        return None

    ## get the WAV from the corpus and read as a Parselmouth Praat object
    wav_file = get_wav_from_corpus(corpus_dir, wav_path)

    wav_dict = []
    ## go through the dataset and extract
    ## all of the stop sections
    for _, obs in wav.iterrows():

        ## select random R/L window size based on lower/upper limits
        lwindow = round(random.uniform(lmin, lmax), ndigits = 3)
        rwindow = round(random.uniform(rmin, rmax), ndigits = 3)

        ## slice audio based on windows
        window_start, window_end = obs[start] - lwindow, obs[end] + rwindow
        obs_sound = wav_file.extract_part(from_time = window_start, to_time = window_end)

        ## save wav file
        obs_sound_fpath = str(obs[o_id]) + ".wav"
        wavout_fpath = out_dir.joinpath("wavs").joinpath(obs_sound_fpath)
        obs_sound.save(file_path = str(wavout_fpath), format = "WAV")
        resample_wav(wavout_fpath, sample_rate)

        ## make a reference of the window start/end points
        obs_dict = {
            'obs' : str(obs[o_id]),
            'start' : float(obs[start]),
            'end' : float(obs[end]),
            'left_window' : lwindow,
            'right_window' : rwindow,
            'window_start' : window_start,
            'window_end' : window_end
        }
        wav_dict.append(obs_dict)

    ## combine window info
    wav_df = pd.DataFrame.from_records(wav_dict)

    return wav_df

if __name__ == "__main__":
    parser = ArgumentParser()

    ## paths
    parser.add_argument("input_csv", type = Path, help = "Path to the CSV containing stops")
    parser.add_argument("corpus_dir", type = Path, help = "Path to top-level of corpus")
    parser.add_argument("out_dir", type = Path, help = "Path to temporary directory (where temporary wavs + prediction TextGrids are stored)")

    ## labels
    parser.add_argument("--wav_label", "-w", type = str, nargs = "?", default = "wav_label", help = "Name of the wav filename column (default = wav_label)")
    parser.add_argument("--obs_label", "-o", type = str, nargs = "?", default = "obs_label", help = "Name of the unique observation ID column (default = obs_label)")
    parser.add_argument("--phone_start", "-s", type = str, nargs = "?", default = "phone_start", help = "Name of the phone start column (default = phone_start)")
    parser.add_argument("--phone_end", "-e", type = str, nargs = "?", default = "phone_end", help = "Name of the phone end column (default = phone_end)")
    parser.add_argument("--closure_start", type = str, nargs = "?", default = None, help = "Name of the optional closure begin column")
    parser.add_argument("--lenited_label", type = str, nargs = "?", default = None, help = "Name of the optional lenition label")

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

    ## start timer
    begin = time.time()

    ## make the output directory if it doesn't already exist
    args.out_dir.mkdir(parents = True, exist_ok = True)

    if args.clean:
        print("Cleaning output directory...")
        shutil.rmtree(args.out_dir)

    ## make a subdir inside the temp
    ## all the clipped WAVs will be stored here
    wav_out_dir = make_subdirs(args.out_dir)

    ## read dataset
    dataset = pd.read_csv(args.input_csv)

    ## make config from input arguments
    config_dict = {
        "input_path" : str(args.input_csv.absolute()),
        "corpus_path" : str(args.corpus_dir.absolute()),
        "sample_rate" : args.sample_rate,
        "left_window_min" : args.left_window_min,
        "left_window_max" : args.left_window_max,
        "right_window_min" : args.right_window_min,
        "right_window_max" : args.right_window_max
    }

    ## split dataframe by WAV file
    print("Extracting wav files...")
    print(f"Left window: [{args.left_window_min},{args.left_window_max}]")
    print(f"Right window: [{args.right_window_min},{args.right_window_max}]")

    ## check lenition/reduction labels are binary
    if args.lenited_label is not None:
        assert dataset[args.lenited_label].isin([0,1]).all(), f"Lenited stop labels should be [0,1]; found {dataset[args.lenited_label].unique()}"

    ## split dataframe by wav file name
    data_wavs = [d for _, d in dataset.groupby([args.wav_label])]

    if args.closure_start is not None:
        segment_start = args.closure_start
    else:
        segment_start = args.phone_start

    ## start parallel job
    with multiprocessing.get_context("spawn").Pool(args.num_jobs) as pool:
        window_dfs = list(pool.starmap(make_wav_clips,
            tqdm(
            zip(data_wavs,
                repeat(args.wav_label),
                repeat(args.corpus_dir),
                repeat(args.out_dir),
                repeat(segment_start),
                repeat(args.phone_end),
                repeat(args.left_window_min),
                repeat(args.left_window_max),
                repeat(args.right_window_min),
                repeat(args.right_window_max),
                repeat(args.obs_label),
                repeat(args.sample_rate)
            ),
            total = len(data_wavs))
        ))
    window_df = pd.concat(window_dfs)

    ## only write the lines for wavs that are saved
    out_wavs = [str(wavpath.stem) for wavpath in list(args.out_dir.joinpath("wavs").glob("*.wav"))]
    dataset = dataset[dataset[args.obs_label].isin(out_wavs)]

    ## check that these match before saving
    assert len(out_wavs) == len(dataset), f"Dataset size mismatch: {len(out_wavs)} wavs; {len(dataset)} rows"

    ## format the window dataframe
    window_df = window_df[window_df['obs'].isin(out_wavs)]
    assert len(dataset) == len(window_df), f"Dataset and window data mismatch: {len(dataset)} rows in dataset; {len(window_df)} rows in window reference data"

    ## reset intervals relative to 0
    ## get interval durations from original timestamps
    if args.closure_start is not None:
        dataset.loc[:,'vot_dur'] = dataset[args.phone_end] - dataset[args.phone_start]
        dataset.loc[:,'closure_dur'] = dataset[args.phone_start] - dataset[args.closure_start]

        ## set these relative to the clip length
        dataset.loc[:,'closure_start'] = window_df['left_window'].values
        dataset.loc[:,'vot_start'] = dataset['closure_start'] + dataset['closure_dur']
        dataset.loc[:,'vot_end'] = dataset['vot_start'] + dataset['vot_dur']

    else:
        dataset.loc[:,'vot_dur'] = dataset[args.phone_end] - dataset[args.phone_start]
        dataset.loc[:,'vot_start'] = window_df['left_window'].values
        dataset.loc[:,'vot_end'] = dataset['vot_start'] + dataset['vot_dur']

    ## make reference CSV of ID labels and the training labels (if supplied)
    if args.closure_start is not None:
        if args.lenited_label is not None:
            dataset = dataset[[args.obs_label, 'closure_start', 'vot_start', 'vot_end', args.lenited_label]]
            colnames = ['obs_id', 'closure_start', 'vot_start', 'vot_end', 'lenition']
        else:
            dataset = dataset[[args.obs_label, 'closure_start', 'vot_start', 'vot_end']]
            colnames = ['obs_id', 'closure_start', 'vot_start', 'vot_end']
    else:
        if args.lenited_label is not None:
            dataset = dataset[[args.obs_label, 'vot_start', 'vot_end', args.lenited_label]]
            colnames = ['obs_id', 'vot_start', 'vot_end', 'lenition']
        else:
            dataset = dataset[[args.obs_label, 'vot_start', 'vot_end']]
            colnames = ['obs_id', 'vot_start', 'vot_end']

    print(f"{len(dataset)} observations written to file")
    data_csv_fpath = args.out_dir.joinpath("data.csv")
    dataset.to_csv(str(data_csv_fpath), index = False, header = colnames)

    window_df_path = args.out_dir.joinpath("window_data.csv")
    window_df.to_csv(str(window_df_path), index = False, header = True)

    with open(args.out_dir.joinpath("wav_config.yaml"), "w") as config_out:
        yaml.dump(config_dict, config_out, sort_keys=False)

    ## end timer
    end = time.time()

    print("Done!")
    print(f"Script took {round((end-begin)/60, 2)} minutes")
