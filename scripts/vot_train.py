import os
import pandas as pd

from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree
from sklearn.model_selection import train_test_split

from model.collator import Wav2VOTDataCollator
from model.classifier import Wav2VOT
from model.data import Wav2VOTDataset
from model.eval import compute_eval_metrics
from model.utils import get_input_data
from model.trainer import Wav2VOTTrainer
from transformers import Wav2Vec2Config, TrainingArguments

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_dir", type = Path, help = "Path to training data")
    parser.add_argument("output_path", type = Path, help = "Path to write model checkpoints")
    parser.add_argument("--base_model", type = Path, nargs = "?", default = None, help = "Path to base model (default: None)")
    parser.add_argument("--test_size", "-t", type = float, nargs = "?", default = 0.2, help = "Percentage of data used for test (default: 0.2)")
    parser.add_argument("--num_jobs", "-j", type = int, nargs = "?", default = 16, help = "Number of data loading jobs (default: 64)")
    parser.add_argument("--batch_size", "-b", type = int, nargs = "?", default = 64, help = "Training/eval batch size (default: 64)")
    parser.add_argument("--num_epochs", "-n", type = int, nargs = "?", default = 10, help = "Number of epochs to use for training (default: 10)")
    parser.add_argument("--learning_rate", "-l", type = float, nargs = "?", default = 1e-4, help = "Initial learning rate (default: 1e-4)")
    parser.add_argument("--eval_steps", "-e", type = int, nargs = "?", default = 200, help = "Number of steps between eval (default: 200)")
    parser.add_argument("--save_steps", "-s", type = int, nargs = "?", default = 200, help = "Number of steps between save (default: 200)")
    parser.add_argument("--metrics_path", "-m", type = str, nargs = "?", help = "Path to a local copy of HuggingFace Evaluate library if using offline (default: None)")
    parser.add_argument("--use_comet", action = "store_true", help = "Use comet.ml for tracking training metrics (default: False)")
    parser.add_argument("--use_cpu", action = "store_true", help = "Use CPU for training (default: False)")
    parser.add_argument("--clean", action = "store_true", help = "Clean model output directory (default: False)")
    args = parser.parse_args()

    if args.clean:
        print(f"Cleaning model output directory ({args.output_path})")
        try:
            rmtree(args.output_path)
        except FileNotFoundError:
            pass

    ## set the device value
    if args.use_cpu:
        device = "cpu"
    else:
        device = "cuda"
    print(f"Using CPU for training: {args.use_cpu}")

    if args.use_comet:
        import comet_ml
        os.environ['COMET_LOG_ASSETS'] = 'True'
        comet_ml.login(project_name='vot-align')
        training_report = "comet_ml"
    else:
        training_report = "none"

    data_fpath, data_csv = get_input_data(args.input_dir)
    dataset = pd.read_csv(data_fpath.joinpath(data_csv))

    ## split into training and test datasets
    train_df, eval_df = train_test_split(dataset, test_size = args.test_size, random_state = 42)

    ## write train/test to csv
    train_fpath = data_fpath.joinpath("train").with_suffix(".csv")
    eval_fpath = data_fpath.joinpath("eval").with_suffix(".csv")
    train_df.to_csv(train_fpath, index = False)
    eval_df.to_csv(eval_fpath, index = False)

    ## reload from csv
    train_dataset = Wav2VOTDataset(train_fpath, sampling_rate = 16000)
    eval_dataset = Wav2VOTDataset(eval_fpath, sampling_rate = 16000)

    ## load from model checkpoint if supplied
    if args.base_model is not None:
        print(f"Loading model from {args.base_model}")
        model = Wav2VOT.from_pretrained(args.base_model).to(device)
    else:
        print("No model supplied -- initialising new model")
        ## set up config for wav2vec
        ## especially convolution/FE size
        conf = Wav2Vec2Config(
            conv_dim = (512, 512, 512, 512),
            conv_stride = (2, 2, 2, 2),
            conv_kernel = (5, 5, 5, 5),
            ## 4 labels + CTC blank token
            num_labels = 5,
            mask_time_prob = 0.0,
            mask_feature_prob = 0.0,
            pad_token_id = -100
        )

        ## initialise model with config
        model = Wav2VOT(
            conf
        )

    ## load data collator
    data_collator = Wav2VOTDataCollator(
        label_padding_value = -100
    )

    print(f"Training dataset size:\t{len(train_dataset)}")
    print(f"Eval dataset size:\t{len(eval_dataset)}")

    training_args = TrainingArguments(
        output_dir = args.output_path,
        group_by_length = False,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        eval_strategy = "steps",
        num_train_epochs = args.num_epochs,
        gradient_checkpointing = True,
        gradient_accumulation_steps = 4,
        lr_scheduler_type = "linear",
        save_steps = args.save_steps,
        eval_steps = args.eval_steps,
        logging_steps = 1,
        learning_rate = 1e-5,
        warmup_ratio = 0.1,
        save_strategy = "best",
        metric_for_best_model = "eval_framewise_f1",
        length_column_name = None,
        save_total_limit = 1,
        use_cpu = args.use_cpu,
        report_to = training_report
    )

    trainer = Wav2VOTTrainer(
        model = model,
        data_collator = data_collator,
        args = training_args,
        compute_metrics = compute_eval_metrics,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset
    )

    print("Beginning training")
    trainer.train()
