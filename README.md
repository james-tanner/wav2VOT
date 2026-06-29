# wav2VOT
<p float="left">
<img width="3600" alt="specs_combined" src="https://github.com/user-attachments/assets/90ce4d67-089f-45df-ae44-e996f92c46df" />
</p>

_wav2VOT_ is a software tool for the automatic estimation of voice onset time (VOT), closure duration, and stop lenition using the wav2vec2 architecture, as reported in Tanner _et al_ (accepted).

Preprint available [here](https://eprints.gla.ac.uk/388172/2/388172.pdf).

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
git clone https://github.com/james-tanner/wav2VOT.git
cd wav2VOT
```

3. Install dependencies
```
pip install -r requirements.txt
```

## Usage
_wav2VOT_ has two primary uses: the _prediction_ of {VOT, closure duration, lenition} for a speech corpus, or the _finetraining_ of a wav2VOT model given some annotated data.

### Prediction
#### via CSV (recommended)
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

#### via TextGrids (experimental)
If your data instead exists in Praat TextGrid format without any corresponding CSV of stop timestamps, you can use `scripts/prepare_wavs_textgrid.py` to prepare your data for prediction. This script has the following command-line arguments:
- path to the corpus of wav-TextGrid pairs (`/path/to/corpus/`)
- regular-expression of labels to extract (e.g. `^[ptkbdgPTKBDG]$`)
- path for the formatted dataset to be written to (`/path/to/output`)
- `__tiername`: the name of the TextGrid tier to search for labels (default: `phones`)
- the same `window_[min/max]` as above

And would be called e.g.
```
python -m scripts.prepare_wavs_textgrid /path/to/corpus/ "^[ptkbdgPTKBDG]$" /path/to/output --tiername=phones
```

#### Running prediction
Once the corpus has been formatted, you can now run `scripts/vot_predict.py` which will run the prediction. Pretrained _wav2VOT_ models can be downloaded from [the OSF repository](https://doi.org/10.17605/OSF.IO/JQU84). `vot_predict.py` takes the following core command-line arguments:
- path to the data to be predicted (`/path/to/output`)
- path to a pretrained _wav2VOT_ model (`/path/to/model/`)
- path to write predictions (`/path/to/preds`)
- `__device`: whether to use CPU or GPU for running prediction (default = `cpu`)
- `__save_tgs`: whether to create TextGrids for the stop predictions (default = `False`)
- `__save_plots`: whether to create spectrogram images for the stop predictions (default = `False`)

For example:
```
python -m scripts.vot_predict /path/to/output /path/to/model/ /path/to/preds --device="cpu"
```

will generate a `predictions.csv` file containing the {VOT, closure duration, lenition} predictions for each stop (without saving TextGrids or spectrogram plots).

### Training/Finetuning
This section explains how to train or finetune a _wav2VOT_ model with a set of annotated {VOT, closure duration, lenition}. As with prediction, _wav2VOT_ assumes that there exists a directory containing the speech audio files, as well as a CSV metadata. For finetuning, the annotations should exist in the metadata as start/end columns (and binary 0/1 lenition labels if provided, where `0` indicates _non-lenition_ (i.e. the presence of a burst):
```
stopID,stopFile,stopStart,stopEnd,closure_start,lenition
"S07M0833_00611138345L","S07M0833",0.033,0.0687299999999441,0.102740000000024,0
"S07M0833_00615850788L","S07M0833",,0.0360579999999727,0.0551160000000591,1
"S07M0833_00616379657L","S07M0833",0.04,0.0570149999999012,0.0792659999999978,0
"S07M0833_00616777889L","S07M0833",,0.0273429999999826,0.0446849999999677,1
"S07M0833_00617599551L","S07M0833",0.035,0.0454720000000498,0.0775400000000309,0
"S07M0833_00618039568L","S07M0833",0.023,0.0960839999999944,0.122251999999979,0
"S07M0833_00619502161L","S07M0833",0.028,0.0666029999999664,0.10673399999994,0
```

Once processed, the data can be used to finetune a new _wav2VOT_ model using `scripts/vot_train.py`. This script takes the following command-line arguments:
- path to the data to be predicted (`/path/to/output`)
- path to write model checkpoints (`/path/to/checkpoints`)
- `__base_model`: path to a pretrained _wav2VOT_ model checkpoint (default = `None`. If not provided, a _wav2VOT_ model with random weights will be initialised). Provide path to a pretrained model if you wish to _finetune_ it to your data.
- `__test_size`: amount of data to be used as test set (as a fraction, default = `0.2`)
- `__batch_size`: size of batches used in training and testing (default = `64`, reduce for a smaller memory footprint)
- `__num_epochs`: number of training epochs (default = `10`)
- `__eval_steps`: number of steps between testing loop (default = `200`)
- `__save_steps`: number of stops between saving model checkpoints (default = `200`)
- `__use_cpu`: whether to use CPU or GPU for model training (default = `False`)

Further training options can be found by using the `--help/-h` flag.

## Citation
Please cite this software as follows:
> Tanner. J., Sonderegger, M., Stuart-Smith, J., Kendall, T., Mielke, J. (accepted). 'wav2VOT: automatic estimation of voice onset time, closure duration, and burst realisation with wav2vec2`. Proceedings of Interspeech 2026. [Link](https://eprints.gla.ac.uk/388172/2/388172.pdf)
