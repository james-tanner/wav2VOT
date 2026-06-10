# wav2VOT
_wav2VOT_ is a software tool for the automatic estimation of voice onset time (VOT), closure duration, and stop lenition using the wav2vec2 architecture, as reported in Tanner _et al_ (accepted).

## Installation 

Usage of a package manager such as `conda` or `virtualenv` is highly recommended.
[Install Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

1. Create conda environment
```
create -n w2vot python=3.14
conda activate w2vot
```

2. Clone the `wav2VOT` repository
```
git clone git clone https://github.com/james-tanner/wav2VOT.git
cd wav2VOT
```

3. Install dependencies
```
pip install -r requirements.txt
```

## Usage
_wav2VOT_ has two primary uses: the _prediction_ of {VOT, closure duration, lenition} for a speech corpus, or the _finetraining_ of a wav2VOT model given some annotated data.

### Prediction
To predict stop information for a speech corpus, _wav2VOT_ assumes you have:
1. a directory containing the speech data; and
2. a CSV file containing columns for the filenames, a unique ID for each stop token, and the start/boundaries for the stop tokens in question (i.e. force-aligned boundaries), e.g.
```
stopID,stopFile,stopStart,stopEnd
stop942L,corpusFile001,7.372942,7.380806
stop541L,corpusFile001,7.688801,7.709115
stop444K,corpusFile002,10.996745,11.017715
```

If your data does not exist in this tabular format, and instead exists in e.g., wav-TextGrid pairs, I recommend using the [PolyglotDB](https://polyglotdb.readthedocs.io/) software package to extract the stop labels from your data into a tabular format.

To prepare your data for prediction, run `scripts/prepare_wavs.py`, which will create a metadata CSV and a new directory containing the audio slices for each stop, including the preceding and following context window. For prediction, the following command-line arguments are used:
- path to the stops CSV (`/path/to/your/stops.csv`)
- path to the directory containing the audio files for the corpus (`/path/to/corpus/wavs/`)
- a path for the formatted dataset to be written to (`/path/to/output`)
- `__wav_label`: the column name in the CSV corresponding to the sound file name (e.g., `stopFile`)
- `__obs_label`: the column name in the CSV corrsponding to the unique stop/token ID (e.g., `stopID`)
- `__phone_start`: the column name in the CSV corresponding to the start of the stop in the sound file (e.g. `stopStart`)
- `__phone_end`: the column name in the CSV corresponding to the end of the stop in the sound file (e.g., `stopEnd`)
- `__left_window_[min/max]`: sets the range of left (preceding) context window sizes (in sec, default `0.03`)
- `__right_window_[min/max]`: sets the range of right (following) context window sizes (in sec, default `0.03`)
This script also supports additional customisation for generating the dataset -- run the script with `-h/--help` for more information)

For the corpus data as above, this would called as the following (run from the base `wav2vot` directory:
```
python -m scripts.prepare_wavs /path/to/stops.csv /path/to/corpus/wavs/ /path/to/output --wav_label=stop_file --obs_label=stopID --phone_start=stopStart --phone_end=stopEnd
```

Once the corpus has been formatted, you can now run `scripts/vot_predict.py` which will run the prediction. Pretrained _wav2VOT_ models can be downloaded from [TODO]. `vot_predict.py` takes the following core command-line arguments:
- path to the data to be predicted (`/path/to/output`)
- path to a pretrained _wav2VOT_ model (`/path/to/model/weights.pth`)
- path to write predictions (`/path/to/preds`)
- `__device`: whether to use CPU or GPU for running prediction (defauly = `cpu`)
- `__save_tgs`: whether to create TextGrids for the stop predictions (default = `False`)
- `__save_plots`: whether to create spectrogram images for the stop predictions (default = `False`)
