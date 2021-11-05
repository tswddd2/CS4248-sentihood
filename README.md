# ABSA as a Sentence Pair Classification Task

Codes and corpora for paper "Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence" (NAACL 2019)

## Requirement

* pytorch: 1.0.0
* python: 3.7.1
* tensorflow: 1.13.1 (only needed for converting BERT-tensorflow-model to pytorch-model)
* numpy: 1.15.4
* nltk
* sklearn

## Step 1: prepare datasets

Already included in data/sentihood

## Step 2: prepare BERT-pytorch-model

Download [BERT-Base (Google's pre-trained models)](https://github.com/google-research/bert) and then convert a tensorflow checkpoint to a pytorch model.

For Bert-Small, download from [here](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8.zip) and unzip in folder bert-small

Then run following:

```
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path bert-small/bert_model.ckpt \
--bert_config_file bert-small/bert_config.json \
--pytorch_dump_path bert-small/pytorch_model.bin
```

## Step 3: train

For example, **BERT-pair-NLI_M** task on **SentiHood** dataset:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_classifier_TABSA.py \
--task_name sentihood_NLI_M \
--data_dir data/sentihood/bert-pair/ \
--vocab_file bert-small/vocab.txt \
--bert_config_file bert-small/bert_config.json \
--init_checkpoint bert-small/pytorch_model.bin \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 24 \
--learning_rate 2e-5 \
--num_train_epochs 6.0 \
--output_dir results/sentihood/NLI_M \
--seed 42
```

Note:

* For SentiHood, `--task_name` must be chosen in `sentihood_NLI_M`, `sentihood_QA_M`, `sentihood_NLI_B`, `sentihood_QA_B` and `sentihood_single`. And for `sentihood_single` task, 8 different tasks (use datasets generated in step 1, see directory `data/sentihood/bert-single`) should be trained separately and then evaluated together.
* For SemEval-2014, `--task_name` must be chosen in `semeval_NLI_M`, `semeval_QA_M`, `semeval_NLI_B`, `semeval_QA_B` and `semeval_single`. And for `semeval_single` task, 5 different tasks (use datasets generated in step 1, see directory : `data/semeval2014/bert-single`) should be trained separately and then evaluated together.

## Step 4: evaluation

Evaluate the results on test set (calculate Acc, F1, etc.).

For example, **BERT-pair-NLI_M** task on **SentiHood** dataset:

```
python evaluation.py --task_name sentihood_NLI_M --pred_data_dir results/sentihood/NLI_M/test_ep_4.txt
```

Note:

* As mentioned in step 3, for `sentihood_single` task, 8 different tasks should be trained separately and then evaluated together. `--pred_data_dir` should be a directory that contains **8 files** named as follows: `loc1_general.txt`, `loc1_price.txt`, `loc1_safety.txt`, `loc1_transit.txt`, `loc2_general.txt`, `loc2_price.txt`, `loc2_safety.txt` and `loc2_transit.txt`
* As mentioned in step 3, for `semeval_single` task, 5 different tasks should be trained separately and then evaluated together. `--pred_data_dir` should be a directory that contains **5 files** named as follows: `price.txt`, `anecdotes.txt`, `food.txt`, `ambience.txt` and `service.txt`
* For the rest 8 tasks, `--pred_data_dir` should be a file just like that in the example.


## Citation

```
@inproceedings{sun-etal-2019-utilizing,
    title = "Utilizing {BERT} for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence",
    author = "Sun, Chi  and
      Huang, Luyao  and
      Qiu, Xipeng",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1035",
    pages = "380--385"
}
```
