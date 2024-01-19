import argparse
import configargparse


def process_args():
    # parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser()

    # Path to load default configs
    parser.add_argument('--config', is_config_file=True, help='Config file path')

    # Datasets
    parser.add_argument(
        "--train_data_file", default=None, type=str,
        help="The input training data file (a text file).")
    parser.add_argument(
        "--eval_data_file", default=None, type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    # Data loader
    parser.add_argument("--col_data", action="store_true", help="Using the specific dataset object in data.py")
    parser.add_argument("--split_sent", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the training dataset")
    parser.add_argument(
        "--block_size", default=-1, type=int,
        help="Optional input sequence length after tokenization."
             "The training dataset will be truncated in block of this size for training."
             "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )

    # Logging and Saving
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--ckpt_steps", type=int, default=1000, help="Checkpoint every X updates steps.")
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument(
        "--overwrite_output_dir", action="store_true",
        help="Overwrite the content of the output directory")

    # Model types
    parser.add_argument(
        "--model_type", type=str, help="The model architecture to be trained or fine-tuned.",)
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir")
    parser.add_argument(
        "--model_name_or_path", default=None, type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",)
    parser.add_argument(
        "--config_name", default=None, type=str, required=True,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",)
    parser.add_argument(
        "--tokenizer_name", default=None, type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",)
    parser.add_argument(
        "--cache_dir", default=None, type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",)
    parser.add_argument(
        "--overwrite_cache", action="store_true",
        help="Overwrite the cached training and evaluation sets")

    # MLM tasks
    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument(
        "--mlm_ratio", type=float, default=1., help="The ratio of mlm loss in the total loss.")

    # Model growth params
    parser.add_argument(
        "--source_model_path", nargs='*', default=None, type=str,
        help="Path to load the source ckpt.")
    parser.add_argument(
        "--grow_scheme", default='none', type=str,
        choices=['none', 'ligo'],
        help="Method to grow the model: [none, ligo]")
    parser.add_argument("--tune_depth", action='store_true', default=False)
    parser.add_argument("--tune_width", action='store_true', default=False)
    parser.add_argument(
        "--fuse_init_scheme", nargs='*', default=['rand'], type=str,
        choices=['rand', 'rand_softmax', 'stackbert', 'stackbert_noisy', 'sel', 'sel_noisy']
        help="Initialization of LiGO operator."
    )
    parser.add_argument(
        "--fuse_init_noise", nargs='*', type=float, default=[0.03],
        help="Noise scale to randomly initialize LiGO operator."
    )
    parser.add_argument("--fuse_tie_param", action='store_true', default=True, help="Turn on parameter tying for LiGO.")
    parser.add_argument("--no_fuse_tie_param", action='store_false', dest='fuse_tie_param', default=False,
        help="Turn off parameter tying for LiGO.")
    parser.set_defaults(fuse_tie_param=True)
    parser.add_argument("--tune_small_model", action='store_true', default=False,
        help="Extra feature: Enabling tuning of small model parameters when training LiGO.")
    parser.add_argument("--tune_residual", action='store_true', default=False,
        help="Extra feature: Adding bias terms in LiGO operator.")
    parser.add_argument("--tune_residual_noise", type=float, default=0.01,
        help="Extra feature: Noise scale to initialize bias terms in LiGO operator.")
    parser.add_argument("--learning_rate_res", default=None, type=float, help="Extra feature: The initial learning rate for learning bias in LiGO.")
    parser.add_argument("--weight_decay_res", default=None, type=float, help="Extra feature: Weight decay for learning bias in LiGO.")

    parser.add_argument(
        "--pretrained_ligo_path", default=None, type=str,
        help="Path to load the checkpoint of LiGO parameter.")

    # Batch Size and Training Steps
    parser.add_argument("--seed", type=int, default=95, help="random seed for initialization")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--max_steps", default=100000, type=int,
        help="Total number of training steps to perform. Override num_train_epochs.",)

    # Layer drop
    parser.add_argument("--layer_drop", action='store_true', default=False, help="Turn on layer drop.")
    parser.add_argument("--layer_drop_rate", type=float, default=False, help="The drop propability to drop a layer.")
    parser.add_argument("--layer_drop_lin_decay", action='store_true', default=False,
        help="Turn on linear decay of survival prob. The --layer_drop_rate will specify the rate for the last layer")

    # Token drop
    parser.add_argument("--token_drop", action='store_true', default=False, help="Turn on token drop.")
    parser.add_argument("--token_drop_rate", type=float, default=False, help="The drop propability to drop a layer.")
    parser.add_argument("--token_drop_start", type=int, default=2, help="The layer index to separate tokens")
    parser.add_argument("--token_drop_end", type=int, default=-1, help="The layer index to merge tokens")

    # Optimizer
    parser.add_argument("--lamb", action="store_true", help='Use the LAMB optimizer in apex')
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_ratio", default=0., type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # Scheduler
    parser.add_argument("--scheduler_type", default='linear', type=str, help="Type of lr scheduler.", choices=['linear', 'cosine', 'poly'])
    parser.add_argument("--scheduler_cosine_cycles", default=0.5, type=float, help="Number of cycles for cosine lr scheduler.")
    parser.add_argument("--scheduler_poly_power", default=1.0, type=float, help="Power of polynomial lr scheduler.")

    # Distributed Training
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--nr", type=int, default=0)

    # Half Precision
    parser.add_argument(
        "--fp16", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument(
        "--fp16_opt_level", type=str, default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",)

    return parser
