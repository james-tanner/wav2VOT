import csv
import torch
import librosa
import shutil
import soundfile
import numpy as np

from argparse import ArgumentParser
from pandas import DataFrame
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

from model.classifier import Wav2VOT
from model.utils import get_input_data, make_textgrid, plot_spectrogram_with_intervals, get_output_lengths, convert_to_intervals, make_interval_dict, correct_window_offsets
from model.data import Wav2VOTPredictionDataset

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def write_predictions(batch):
    for i in range(len(batch['names'])):

        ## extract information from dicts
        name = batch['names'][i]
        preds = batch['preds'][i]
        audio = batch['audios'][i]
        left_window = batch['left_windows'][i]
        start = batch['starts'][i]

        ## smooth predicted labels based on minimum duration
        preds = smooth_preds(preds, min_frame_length = args.min_frame_length)
        audio_duration = librosa.get_duration(y = audio, sr = 16000)
        intervals = convert_to_intervals(preds, audio_duration)
        interval_dict = make_interval_dict(intervals, name)
        interval_dict = correct_window_offsets(interval_dict, left_window, start)

        if args.save_tgs:
            pred_path = args.out_path.joinpath("preds")
            pred_path.mkdir(parents = True, exist_ok = True)

            wav_out = pred_path.joinpath(name).with_suffix(".wav")
            tg_path = pred_path.joinpath(name).with_suffix(".TextGrid")

            make_textgrid(
                intervals,
                audio_duration,
                tg_path
            )

            soundfile.write(
                str(wav_out),
                audio,
                16000
            )
    return interval_dict


def pad_inputs(batch):
    names = [item["audio_name"] for item in batch]
    audios = [item["audio"] for item in batch]
    left_windows = [item['left_window'] for item in batch]
    starts = [item['start'] for item in batch]
    input_values = [item["input_values"] for item in batch]
    input_lengths = [item["input_lengths"] for item in batch]
    padded_input_values = pad_sequence(input_values, batch_first=True, padding_value = -1)
    return {
        "input_names" : names,
        "input_audios" : audios,
        "left_windows" : left_windows,
        "starts" : starts,
        "input_lengths" : input_lengths,
        "input_values" : padded_input_values}


def read_csv(data_csv):
    outdata = []
    with open(data_csv, "r") as f:
        data = csv.reader(f)
        for row in data:
            outdata.append(row[0])
    return outdata


def smooth_preds(pred_labels, min_frame_length = 5, ctc_label = 4):
    ## convert to numpy array and make copy
    pred_labels = np.array(pred_labels)
    out = pred_labels.copy()

    ## merge CTC blank labels with previous
    for i in range(len(out)):
        if out[i] == ctc_label:
            if i > 0:
                out[i] = out[i-1]
            ## use preceding non-CTC label if CTC label is first
            else:
                next_non_ctc = np.where(out != ctc_label)[0]
                if len(next_non_ctc) > 0:
                    out[i] = out[next_non_ctc[0]]
                else:
                    raise ValueError(f"Pathological sequence: {out}")

    ## Find predicted boundaries
    change_points = np.where(np.diff(pred_labels) != 0)[0] + 1
    ## concat arrays based on boundaries
    segment_starts = np.r_[0, change_points]
    segment_ends   = np.r_[change_points, len(pred_labels)]

    for start, end in zip(segment_starts, segment_ends):
        seg_len = end - start
        if seg_len < min_frame_length:
            ## replace with previous label if possible, else with next
            if start > 0:
                out[start:end] = out[start - 1]
            else:
                out[start:end] = out[end]

    return out


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("data_path", type = Path)
    parser.add_argument("model_path", type = Path)
    parser.add_argument("out_path", type = Path)
    parser.add_argument("--num_jobs", type = int, nargs = "?", default = 8, help = "Number of parallel jobs for writing predictions to file (default = 8)")
    parser.add_argument("--min_frame_length", type = int, nargs = "?", default = 5, help = "Minimum length of interval in (approximate) ms (default: 5)")
    parser.add_argument("--device", nargs = "?", type = str, default = "cpu")
    parser.add_argument("--save_tgs", action = "store_true", help = "Write predicted TextGrids and audio files (default: False)")
    parser.add_argument("--save_plots", action = "store_true", help = "Write spectrogram plots with annotation overlays (default: False)")
    parser.add_argument("--clean", action = "store_true", help = "Clean model output directory (default: False)")
    args = parser.parse_args()

    ## set device
    device = torch.device(args.device)

    ## load model
    model = Wav2VOT.from_pretrained(args.model_path).to(device)

    ## get shape of FE for later conversions
    kernel_sizes = model.config.conv_kernel
    strides = model.config.conv_stride

    ## make the output directory if it doesn't already exist
    args.out_path.mkdir(parents = True, exist_ok = True)

    if args.clean:
        print("Cleaning output directory...")
        shutil.rmtree(args.out_path)

    ## load dataset
    data_fpath, data_csv = get_input_data(args.data_path, default = "window_data.csv")

    ## load data into batches (batches not currently supported)
    dataset = Wav2VOTPredictionDataset(data_fpath.joinpath(data_csv))
    prediction_loader = DataLoader(dataset, batch_size = 1, shuffle = False, collate_fn = pad_inputs)

    ## run inference on batches
    print(f"Running inference (device = {args.device})")
    predictions = []
    with torch.no_grad():
        for batch in tqdm(prediction_loader, total = len(prediction_loader)):
            inputs = batch['input_values'].to(device)
            logits = model(inputs)["logits"]
            preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

            ## add batch dimension if missing
            ## e.g. when batch_size=1
            if preds.ndim == 1:
                preds = np.expand_dims(preds, axis=0)

            ## remove padding from tokens using the
            ## input lengths (in FE dimensions)
            unpadded_preds = []
            output_lengths = [get_output_lengths(L, kernel_sizes, strides) for L in batch['input_lengths']]
            for i, L in enumerate(output_lengths):
                unpadded_pred = preds[i, :L]
                unpadded_preds.append(unpadded_pred)

            pred_dict = {
                'preds' : unpadded_preds,
                'names' : batch['input_names'],
                'audios' : batch['input_audios'],
                'lengths' : batch['input_lengths'],
                'left_windows' : batch['left_windows'],
                'starts' : batch['starts']
            }

            predictions.append(pred_dict)

    print(f"Writing predictions to file (num_jobs = {args.num_jobs})")
    with Pool(args.num_jobs) as pool:
        predicted_intervals = list(tqdm(pool.imap_unordered(write_predictions, predictions), total = len(predictions)))

    predicted_df_path = args.out_path.joinpath("predictions").with_suffix(".csv")
    print(f"Writing predictions to {str(predicted_df_path)}")
    predicted_df = DataFrame.from_records(predicted_intervals)
    predicted_df.to_csv(predicted_df_path, index = False)

    if args.save_plots:
        plots_path = args.out_path.joinpath("plots")
        plots_path.mkdir(parents = True, exist_ok = True)
        print(f"Writing plots to {str(plots_path)}")

        for batch in tqdm(predictions, total = len(predictions)):
            for i in range(len(batch['names'])):

                ## extract information from dicts
                name = batch['names'][i]
                preds = batch['preds'][i]
                audio = batch['audios'][i]

                ## smooth predicted labels based on minimum duration
                preds = smooth_preds(preds, min_frame_length = args.min_frame_length)
                audio_duration = librosa.get_duration(y = audio, sr = 16000)

                plot_path = plots_path.joinpath(name).with_suffix(".pdf")
                plot_spectrogram_with_intervals(
                    audio,
                    preds,
                    plot_path
                )

    print("Done!")
