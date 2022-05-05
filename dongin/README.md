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