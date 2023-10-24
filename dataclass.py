import random
import torch
import numpy as np
import progressbar
from torch.nn.utils import rnn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# add
from datasets import load_dataset
from transformers import (
    MBartTokenizer,
    M2M100Tokenizer,
    default_data_collator,
    DataCollatorForSeq2Seq,
    PhrasalConstraint
)


class Data:
    def __init__(self, args, model_name):
        '''
            model_name: mbart
            dataset_name: load the data from datasets library
            train_file: training data path
            validation_file: validation data path
        '''
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        ### load / download the dataset
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        else:
            data_files = {}
            valid_data_files = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            if args.validation_file is not None:
                valid_data_files["train"] = args.validation_file
            extension = "csv"
            raw_datasets_1 = load_dataset(extension, data_files=data_files, delimiter='\t')
            # raw_valid_datasets = load_dataset(extension, data_files=valid_data_files, delimiter='\t')

        # test_split_ratio = 0
        length = len(raw_datasets_1)
        if length < 1000:
            test_split_ratio = 0.1
        elif length < 10000:
            test_split_ratio = 0.01
        elif length < 50000:
            test_split_ratio = 0.001
        else:
            test_split_ratio = 0.0001

        # split into train/test
        raw_datasets = raw_datasets_1['train'].train_test_split(test_size=test_split_ratio)

        # process the dataset
        prefix = args.source_prefix if args.source_prefix is not None else ""
        column_names = raw_datasets["train"].column_names
        # column_valid_names = raw_valid_datasets["train"].column_names
        if isinstance(self.tokenizer, (M2M100Tokenizer, MBartTokenizer)):
            if args.target_lang is not None:
                self.tokenizer.tgt_lang = args.target_lang

        padding = "max_length" if args.pad_to_max_length else False

        # define a function to process the data
        def preprocess_function(examples):
            # process the word
            src_sentences = [str(ex) for ex in examples['src_text']]
            constraints = [str(ex) for ex in examples['constraints']]

            model_inputs = self.tokenizer(src_sentences, max_length=args.max_source_length, padding=padding, truncation=True)

            constraints_input_ids = []
            for forced_token_str in constraints:
                forced_token_list = eval(forced_token_str)
                force_words_ids = []
                for small_token_list in forced_token_list:
                    tmp_token_list = []
                    for token in small_token_list:
                        tmp_token_ids = self.tokenizer(token, add_special_tokens=False).input_ids
                        tmp_token_list.append(tmp_token_ids)
                    force_words_ids.append(tmp_token_list)
                constraints_input_ids.append(force_words_ids)

            final_clean_list = []
            for first_list in constraints_input_ids:
                tmp_first_list = []
                for second_list in first_list:
                    second_list.sort(key=lambda j: len(j), reverse=True)
                    tmp_second_list = []
                    for third_list in second_list:
                        if third_list in tmp_second_list:
                            continue
                        elif True in [set(third_list).issubset(set(tmp)) for tmp in tmp_second_list]:
                            continue
                        else:
                            tmp_second_list.append(third_list)
                    tmp_first_list.append(tmp_second_list)
                final_clean_list.append(tmp_first_list)

            model_inputs["constraints"] = final_clean_list
            return model_inputs

        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["test"]

        # create data loader
        label_pad_token_id = -100 if args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        if args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                label_pad_token_id=label_pad_token_id,
            )

        # add support for distributed parallel data training
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, collate_fn=data_collator, batch_size=args.batch_size_per_gpu, sampler=train_sampler
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size_per_gpu)

        self.raw_dtasets = raw_datasets
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.train_num = len(train_dataset)
        self.eval_num = len(eval_dataset)
        self.target_lang = args.target_lang

        print('train number:{}, dev number:{}'.format(self.train_num, self.eval_num))
