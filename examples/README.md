# Example Tutorial

This page walks through a simple example of using _wav2VOT_ with a small corpus drawn from the _LJ Speech_ dataset ([link](https://keithito.com/LJ-Speech-Dataset/)). This tutorial goes through the process of extracting stop metadata from a corpus (wav + TextGrid pairs), pre-processing them for prediction, running the wav2VOT prediction, and examining the predicted output. 

## Stop extraction
### Option 1: via CSV
To find the stops that we wish to estimate, we can use [_PolyglotDB_](https://polyglotdb.readthedocs.io/) to programmatically extract the metadata for the stops of interest. Here we can utilise PolyglotDB's functionality to extract only a subset of stops (e.g. intervocalic). A script to extract these is included here (`stops.py`). PolyglotDB has its own separate installation and usage, and it is recommended to read the PolyglotDB documentation. For convenience, the output of this script of the LJ data is also included here (`LJ_stops.csv`):

```
speaker,discourse,phone_id,word,syllable_label,phone_label,phone_begin,phone_end
LJ,LJ025-0076,33e505ca-91a1-11f1-b5de-bc2411924b5d,devoid,D.IH0.V,D,6.08,6.17
LJ,LJ037-0171,33c9155e-91a1-11f1-b5de-bc2411924b5d,enabled,B.AH0.L.D,B,3.23,3.31
LJ,LJ025-0076,33e50688-91a1-11f1-b5de-bc2411924b5d,devoid,OY1.D,D,6.55,6.61
LJ,LJ025-0076,33e50188-91a1-11f1-b5de-bc2411924b5d,parasitically,T.IH0.K,T,3.82,3.91
LJ,LJ025-0076,33e509bc-91a1-11f1-b5de-bc2411924b5d,cavity,K.AE1.V,K,7.63,7.75
LJ,LJ037-0171,33c9189c-91a1-11f1-b5de-bc2411924b5d,conclude,K.AH0.N.K,K,4.18,4.3
LJ,LJ037-0171,33c9166c-91a1-11f1-b5de-bc2411924b5d,commission,K.AH0,K,3.53,3.65
LJ,LJ025-0076,33e50a7a-91a1-11f1-b5de-bc2411924b5d,cavity,T.IY0,T,7.98,8.07
```

Here each row corresponds to an intervocalic stop present in the LJ data. With this, it is possible to use wav2VOT's `prepare_wavs.py` to process the stops for prediction. This processing involves slicing the stop audio (with a determined left-right context window) and organising the metadata into a wav2VOT-friendly format. This script requires the paths to the CSV and audio file as arguments, as well as named arguments which specify the relevant CSV column names.

```
## activate the wav2VOT conda environment
conda activate w2v

## move to the wav2VOT directory
cd wav2VOT

## run the processing script on LJ data
## with the corresponding arguments
python -m scripts.prepare_wavs examples/LJ_stops.csv examples/LJ examples/LJ-processed --wav_label=discourse --obs_label=phone_id --phone_start=phone_begin --phone_end=phone_end
```

If run successfully, you should receive an output similar to:
```
Extracting wav files...
Left window: [0.03,0.03]
Right window: [0.03,0.03]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 3133.59it/s]
8 observations written to file
Done!
Script took 0.06 minutes
```

This will create a directory called `LJ-processed` in the `examples` directory. We can view the contents of this directory with `ls examples/LJ-processed`:
```
ls examples/LJ-processed
data.csv  wav_config.yaml  wavs  window_data.csv
```

Here `window_data.csv` is the metadata file used by wav2VOT to locate the sliced audio files on your machine, and contains the information about the position of the stops in the original audio files. We can view the contents of this file with `cat examples/LJ-processed/window_data.csv`:

```
cat examples/LJ-processed/window_data.csv

obs,start,end,left_window,right_window,window_start,window_end
33e505ca-91a1-11f1-b5de-bc2411924b5d,6.08,6.17,0.03,0.03,6.05,6.2
33e50688-91a1-11f1-b5de-bc2411924b5d,6.55,6.61,0.03,0.03,6.52,6.640000000000001
33e50188-91a1-11f1-b5de-bc2411924b5d,3.82,3.91,0.03,0.03,3.79,3.94
33e509bc-91a1-11f1-b5de-bc2411924b5d,7.63,7.75,0.03,0.03,7.6,7.78
33e50a7a-91a1-11f1-b5de-bc2411924b5d,7.98,8.07,0.03,0.03,7.95,8.1
33c9155e-91a1-11f1-b5de-bc2411924b5d,3.23,3.31,0.03,0.03,3.2,3.34
33c9189c-91a1-11f1-b5de-bc2411924b5d,4.18,4.3,0.03,0.03,4.1499999999999995,4.33
33c9166c-91a1-11f1-b5de-bc2411924b5d,3.53,3.65,0.03,0.03,3.5,3.6799999999999997
```

### Option 2: via TextGrids
The alternative to using a CSV to process the data (e.g. if your data is not PolyglotDB-compatible) is to instead extract directly from the TextGrids themselves. This approach has less functionality than the CSV method, and instead looks for the appropriate labels in a TextGrid based on the user-provided tier and label names. As this LJ data is MFA-aligned, we can use the default `phone` name as the tier name, and we can use the CMUDict labels for the stops, which are wrapped in a regular expression statement

```
## activate the wav2VOT conda environment
conda activate w2v

## move to the wav2VOT directory
cd wav2VOT

## process from TextGrids
python -m scripts.prepare_wavs_textgrid examples/LS "^[PTKBDG]$" examples/LJ-processed-tgs

## successful output
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 5797.24it/s]
Done!
```

Here we receive a greater number of stop tokens compared with the CSV method, as we are not able to provide additional structural/linguistic filters (such as surrounding context):
```
## show number of rows in CSV
wc -l examples/LJ-processed-tgs/window_data.csv
32
```

## Running prediction
Now that the stops have been processed, it is now possible to run the wav2VOT prediction script. wav2VOT requires a pre-trained model (`csj-base-model`) which can be downloaded from [OSF](https://osf.io/jqu84/). Download this directory and place it in your `examples` directory. As we want to visually explore the output of the prediction, we will use the `--save_tgs` flag, which will also save the predictions (and wav files) as a separate directory in our predictions output.

```
python -m scripts.vot_predict examples/LJ-processed examples/csj-base-model examples/LJ-predictions --save_tgs

## successful output
Running inference (device = cpu)
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:17<00:00,  2.24s/it]
Writing predictions to file (num_jobs = 8)
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 117734.85it/s]
Writing predictions to examples/LJ-predictions/predictions.csv
Done!
```
Once this has run successfully, the `examples/LJ-predictions` directory will be created. The CSV `predictions.csv` shows the predictions for all the stops, including the timestamps (relative to the original corpus timestampes) for the VOT and any observed closure. The `lenit` column is a binary label as to whether the stop was predicted as 'lenited' (without an observed burst) or not. The `sequence` column shows what overall sequence of events was observed for the stop (e.g. `W C V W` = 'window, closure, VOT, window').
```
cat examples/LJ-predictions/predictions.csv

filename,closure_start,closure_end,stop_start,stop_end,lenit,sequence,left_window
33e505ca-91a1-11f1-b5de-bc2411924b5d,6.0775625,6.153104166666667,6.153104166666667,6.1694375,0,W C V W,0.03
33e50688-91a1-11f1-b5de-bc2411924b5d,6.558974358974359,6.5835897435897435,6.5835897435897435,6.596923076923077,0,W C V W,0.03
33e50188-91a1-11f1-b5de-bc2411924b5d,3.8287916666666666,3.8665624999999997,3.8665624999999997,3.9114791666666666,0,W C V W,0.03
33e509bc-91a1-11f1-b5de-bc2411924b5d,7.631525423728814,7.682372881355932,7.682372881355932,7.753559322033898,0,W C V W,0.03
33e50a7a-91a1-11f1-b5de-bc2411924b5d,,,7.9836875,8.062291666666667,1,W L W,0.03
33c9155e-91a1-11f1-b5de-bc2411924b5d,3.236788321167883,3.2848175182481754,3.2848175182481754,3.297080291970803,0,W C V W,0.03
33c9189c-91a1-11f1-b5de-bc2411924b5d,4.1805190677966095,4.238505296610169,4.238505296610169,4.301578036723163,0,W C V W,0.03
33c9166c-91a1-11f1-b5de-bc2411924b5d,3.532542372881356,3.5966101694915253,3.5966101694915253,3.65864406779661,0,W C V W,0.03
```

As we used the `--save_tgs` flag, a separate `preds` directory was created, which contains the audio and predicted stop intervals, which can then be examined in Praat.

```
ls examples/LJ-predictions/preds

33c9155e-91a1-11f1-b5de-bc2411924b5d.TextGrid  33e505ca-91a1-11f1-b5de-bc2411924b5d.TextGrid
33c9155e-91a1-11f1-b5de-bc2411924b5d.wav       33e505ca-91a1-11f1-b5de-bc2411924b5d.wav
33c9166c-91a1-11f1-b5de-bc2411924b5d.TextGrid  33e50688-91a1-11f1-b5de-bc2411924b5d.TextGrid
33c9166c-91a1-11f1-b5de-bc2411924b5d.wav       33e50688-91a1-11f1-b5de-bc2411924b5d.wav
33c9189c-91a1-11f1-b5de-bc2411924b5d.TextGrid  33e509bc-91a1-11f1-b5de-bc2411924b5d.TextGrid
33c9189c-91a1-11f1-b5de-bc2411924b5d.wav       33e509bc-91a1-11f1-b5de-bc2411924b5d.wav
33e50188-91a1-11f1-b5de-bc2411924b5d.TextGrid  33e50a7a-91a1-11f1-b5de-bc2411924b5d.TextGrid
33e50188-91a1-11f1-b5de-bc2411924b5d.wav       33e50a7a-91a1-11f1-b5de-bc2411924b5d.wav
```
