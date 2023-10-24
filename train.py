# coding=utf-8
import torch
import argparse

import logging
logging.getLogger('transformers.generation_utils').disabled = True


def parse_config():
    parser = argparse.ArgumentParser()
    # data configuration
    parser.add_argument("--model_name", type=str, default='gpt2')
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--min_len", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--eos_token", type=str)
    parser.add_argument("--pad_token", type=str)
    # mini-batch training configuration
    parser.add_argument("--number_of_gpu", type=int, help="Number of available GPUs.")
    parser.add_argument("--batch_size_per_gpu", type=int, help='batch size for each gpu.')
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation step.")
    parser.add_argument("--effective_batch_size", type=int,
        help="effective_bsz = batch_size_per_gpu x number_of_gpu x gradient_accumulation_steps")
    # pre-training configuration
    parser.add_argument("--total_steps", type=int,
        help="total effective training steps")
    parser.add_argument("--print_every", type=int,
        help="how many update steps to print one intermediate result")
    parser.add_argument("--save_every", type=int,
        help="how many update steps to save one model")
    # learning configuration
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_path_prefix", type=str, help="directory to save the model parameters.")

    # add machine translation parser
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument("--source_lang", type=str, default=None, help="Source language id for translation.")
    parser.add_argument("--target_lang", type=str, default=None, help="Target language id for translation.")

    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=True,
        help=(
            "Whether to pad all samples to model maximum sentence "
            "length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More"
            "efficient on GPU but very bad for TPU."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )

    # distributed training
    parser.add_argument('--local_rank', type=int, default=0, help = 'node rank for distributed training')

    # add contrastive parameters
    parser.add_argument("--contrastive-lambda", type=float,
                        default=0.0,
                        help="The contrastive loss weight")
    parser.add_argument("--temperature", type=float,
                        default=1.0, )

    return parser.parse_args()


def load_previous_best_model(path):
    import os
    filenames = os.listdir(path)
    for file in filenames:
        if file.startswith('training_step'):
            return path + '/' + file
    raise Exception('No best model found!')


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print('Using single GPU training.')
    else:
        pass
    args = parse_config()
    model_name = args.model_name

    # set the distribution environ
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device('cuda')

    print('Loading data...')
    from dataclass import Data
    data = Data(args, model_name)
    print('Data loaded.')

    from trainer import model_training
    print('############################################################')
    print('Start Training...')
    from constraint_cl import Cons_CL
    print('Initializaing model...')
    model = Cons_CL(model_name, data)
    if cuda_available:
        model = model.to(device)
        if multi_gpu_training:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        else:
            pass
    else:
        pass
    print('Model loaded')
    total_steps, print_every, save_every = args.total_steps, args.print_every, args.save_every
    ckpt_save_path = args.save_path_prefix
    model = model_training(args, data, model, total_steps, print_every, save_every, ckpt_save_path, cuda_available, device)
    print('Training stage completed!')
    print('############################################################')