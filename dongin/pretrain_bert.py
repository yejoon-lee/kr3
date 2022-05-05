import pandas as pd
from transformers import BertTokenizer, BertForPreTraining
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from tqdm import tqdm
import numpy as np
import torch
import time, datetime
from nltk.tokenize import sent_tokenize
import random
import argparse
from pathlib import Path

def except_single_sent(sent_list):
	train_list = []
	for each_sent in sent_list:
		if len(each_sent)>1:
			train_list.append(each_sent)
	return train_list

def make_tokenizing(train_list):
	documents = [[]]
	for each_review in tqdm(train_list):
		for each_line in each_review:
			tokens = tokenizer.tokenize(each_line)
			tokens = tokenizer.convert_tokens_to_ids(tokens)
			documents[-1].append(tokens)
		documents.append([])
	documents = documents[0:-1]
	return documents


def create_examples_from_document(document, doc_index, block_size, tokenizer, short_seq_probability, nsp_probability):
    max_num_tokens = block_size - tokenizer.num_special_tokens_to_add(pair=True)
    target_seq_length = max_num_tokens
    if random.random() < short_seq_probability:
        target_seq_length = random.randint(2, max_num_tokens)

    current_chunk = []  # a buffer stored current working segments
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                
				if len(current_chunk) == 1 or random.random() < nsp_probability:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = random.randint(0, len(documents) - 1)
                        if random_document_index != doc_index:
                            break
                    # 여기서 랜덤하게 선택합니다 :-)
                    random_document = documents[random_document_index]
                    random_start = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
                    """Truncates a pair of sequences to a maximum sequence length."""
                    while True:
                        total_length = len(tokens_a) + len(tokens_b)
                        if total_length <= max_num_tokens:
                            break
                        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
                        assert len(trunc_tokens) >= 1
                        # We want to sometimes truncate from the front and sometimes from the
                        # back to add more randomness and avoid biases.
                        if random.random() < 0.5:
                            del trunc_tokens[0]
                        else:
                            trunc_tokens.pop()

                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                # add special tokens
                input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                # add token type ids, 0 for sentence a, 1 for sentence b
                token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
                
                example = {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                    "next_sentence_label": torch.tensor(1 if is_random_next else 0, dtype=torch.long),
                }

                examples.append(example)

            current_chunk = []
            current_length = 0

        i += 1


def main(args):
	tokenizer = BertTokenizer.from_pretrained(args.model_path)
	data = pd.read_csv(args.data_path)
	data['sent'] = data.Review.apply(lambda x: sent_tokenize(x))
	
	train_list = except_single_sent(data['sent'])
	documents = make_tokenizing(train_list)


	examples = []
	for doc_index, document in enumerate(documents):
		create_examples_from_document(document=document, doc_index=doc_index, block_size=args.block_size, tokenizer=tokenizer, short_seq_probability=args.short_seq_prob, nsp_probability=args.nsp_prob)

	

	model = BertForPreTraining.from_pretrained(args.model_path)

	training_args = TrainingArguments(output_dir=args.output_path, overwrite_output_dir=True, max_steps=295000, per_device_train_batch_size=args.batch_size, save_steps=args.save_steps, save_total_limit=args.save_total_limit, logging_steps=args.logging_steps)
	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
	
	trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=examples)
	
	trainer.train()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', default='bert-base-multilingual-cased')
	parser.add_argument('--data_path', default='../kr3.csv', type=Path)
	parser.add_argument('--output_path', default='./pretrain_ckp', type=Path)
	parser.add_argument('--batch_size', default=8, type=int)
	parser.add_argument('--train_steps', type=int)
	parser.add_argument('--num_epoch', type=int)

	parser.add_argument('--short_seq_prob', default=0.1, type=float)
	parser.add_argument('--nsp_prob', default=0.5, type=float)
	parser.add_argument('--block_size', default=256, type=int)
	parser.add_argument('--save_steps', default=1000, type=int)
	parser.add_argument('--save_total_limit', default=2, type=int)
	parser.add_argument('--logging_steps', default=5000, type=int)


	args = parser.parse_args()
	main(args)