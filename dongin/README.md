# Additional pretraining with multilingual bert

In this section, additional pretrain on multilingual bert to achieve better performance of sentiment classification.

To do not cheat fine-tuning dataset, get another domain(or task)'s data.

### pretrain on kr3
```python
python pretrain_bert.py --model_path bert-base-multilingual-cased --data_path ../kr3.csv --num_epoch 10
```

### pretrain on nsmc
```python
python pretrain_bert.py --model_path bert-base-multilingual-cased --data_path ../data/nsmc/ratings.txt --num_epoch 10
```
### finetune on kr3
```python
python finetune_sentiment.py --model_path bert-base-multilingual-cased --data_path ../kr3.csv --num_epoch 10
```

### finetune on nsmc
```python
python finetune_sentiment.py --model_path bert-base-multilingual-cased --data_path ../data/nsmc/ratings.txt --num_epoch 10
```

#### other custom model OR train options
```python
python finetune_sentiment.py --model_path {CUSTOM_MODEL_DIR or HUGGINGFACE_PATH} --data_path {DATA_PATH} --lr {LEARING_RATE} --tokenizer_path {TOKENIZER FOLLOWS MODEL} --output_path {CHECKPOINT_SAVING_PATH} --batch_size {BATCH_SIZE} --num_epoch {EPOCHS} --es_patience {EARLYSTOPPING_PATIENCE} 
```

### Effect of additional pretraining

| fine-tune <br> \ </br>additional pre-train| KR3 ||  NSMC ||
 :--- | ---: | ---: | ---: | ---: |
|| F1 (macro) | Acc. | F1 (macro) | Acc. |
 no pre-training (original bert) | 0.8709 | 0.9348 | 0.8616 | 0.8617 |
 KR3 | - | - | <u>0.8748</u> | <u>0.8749</u> |
 NSMC + Naver Shopping + Steam | <u>0.9325</u> | <u>0.9653</u> | - | - |


- Additional pretrained model achieves better performance on fine-tuning task.
	- It was just a mini-project to perform additional pretrain and finetune sentiment classification task simply.
- Actually, there is a leap in which the original multilingual-bert was trained with less Korean data.
- We have future works that the more fine experiments (such as hyper-parameter tuning, distillation, etc.) and using Korean language model.