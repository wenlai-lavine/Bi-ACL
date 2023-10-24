# **Bi-ACL**

code implements for paper **[Mitigating Data Imbalance and Representation Degeneration in Multilingual Machine Translation
](https://arxiv.org/abs/2305.12786)** published in Findings of EMNLP 2023.

------

**Requirements**

1. Transformers (>=4.18.0)
2. Pytorch (>=1.9.0)

------

**Pipeline**

+ Data
   + Monolingual data: From [CCAligned](https://opus.nlpl.eu/CCAligned.php) and [news-crawl](https://data.statmt.org/news-crawl/). please find more detailed information in the paper.
   + Bilingual Dictionary: Extract directly from [wiktextract](https://github.com/tatuylonen/wiktextract/) tool. Note that, for pairs not involving English, please pivot them through English.
   + Data Preprocessing: Use the language detection tool to filter the sentences with mixed language first. Please see more details in the paper.

+ Training
   + For the baseline systems, please see the implementation on their original paper.
   + Training
     ```
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py \
        --model_name facebook/m2m100_418M \
        --train_file $INPUT_PATH \
        --source_lang ta \
        --target_lang en \
        --number_of_gpu 8 \
        --batch_size_per_gpu 8 \
        --gradient_accumulation_steps 1 \
        --effective_batch_size 8 \
        --total_steps 10000 \
        --print_every 10 \
        --save_every 10 \
        --learning_rate 2e-5 \
        --save_path_prefix $OUT_PATH
     ```
    

****
If you find our paper useful, please kindly cite our paper. Thanks!
```bibtex
@article{lai2023mitigating,
  title={Mitigating Data Imbalance and Representation Degeneration in Multilingual Machine Translation},
  author={Lai, Wen and Chronopoulou, Alexandra and Fraser, Alexander},
  journal={arXiv preprint arXiv:2305.12786},
  year={2023}
}
```
   
### Contact
If you have any questions about our paper, please feel convenient to let me know through email: [lavine@cis.lmu.de](mailto:lavine@cis.lmu.de) 

   

