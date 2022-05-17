import pandas as pd
import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

import argparse
import time, datetime
import os

def make_dataloader(dataset, tokenizer, MAX_LEN, is_train=False):
    input_ids, attention_masks, token_type_ids = [], [], []
    for each_review in tqdm(dataset['Review']):
        encoded_dict = tokenizer.encode_plus(text = each_review, add_special_tokens=True, max_length=MAX_LEN, padding='max_length', return_attention_mask=True, truncation=True)

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])
    
    tensordataset = TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(token_type_ids), torch.LongTensor(dataset['Rating'].values).unsqueeze(dim=1))
    if is_train:
        dataloader = DataLoader(tensordataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count() // 2 - 1)
    else:
        dataloader = DataLoader(tensordataset)
    return dataloader

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint_testing.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print("")
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def fine_tune(model, optimizer, train_loader, valid_loader, test_loader, early_stopping, args):
    train_loss_list, valid_loss_list, valid_accuracy_list, valid_f1_list = [], [], [], []
    optimizer.zero_grad()
    for epoch in range(args.num_epoch):
        t0 = time.time()
        train_pred_list = []
        train_loss, valid_loss, valid_accuracy = 0.0, 0.0, 0.0
        epoch_loss = 0.0
        model.train()
        optimizer.zero_grad()
        for batch in train_loader:
            b_input_ids, b_attention_masks, b_token_type_ids, b_labels = tuple(t.to(args.device) for t in batch)
            out = model(b_input_ids, b_attention_masks, b_token_type_ids, labels=b_labels)
            loss, logits = out.loss, out.logits
            epoch_loss += loss.item()        

            loss.backward()
            optimizer.step()
            #pred = torch.argmax(F.softmax(logits, dim=0), dim=1).unsqueeze(dim=1).cpu()
            #pred_list.append(pred)
            #epoch_accuracy += (pred==b_labels).cpu().numpy().mean()
            
            optimizer.zero_grad()
            
        train_loss = float(epoch_loss / len(train_loader))
        #train_accuracy = float(epoch_accuracy / len(train_dataloader))
        
        train_loss_list.append(train_loss)
        #train_accuracy_list.append(train_accuracy)
        
        valid_pred_list, valid_real_list = [], []
        valid_accuracy, valid_f1, epoch_accuracy, epoch_loss = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            model.eval()
            for batch in valid_loader:
                b_input_ids, b_attention_masks, b_token_type_ids, b_labels = tuple(t.to(args.device) for t in batch)
                out = model(b_input_ids, b_attention_masks, b_token_type_ids, labels=b_labels)
                loss, logits = out.loss, out.logits
                epoch_loss += loss.item()

                pred = torch.argmax(logits).cpu()
                valid_pred_list.append(pred)
                valid_real_list.append(b_labels.squeeze(dim=0).squeeze(dim=0).cpu())
                #epoch_accuracy += (pred==b_labels).cpu().numpy()
                
            
            valid_loss = float(epoch_loss / len(valid_loader))
            #valid_accuracy = float(epoch_accuracy / len(valid_dataloader))
            valid_accuracy = accuracy_score(valid_real_list, valid_pred_list)
            valid_f1 = f1_score(valid_real_list, valid_pred_list)
            
            valid_loss_list.append(valid_loss)
            valid_accuracy_list.append(valid_accuracy)
            valid_f1_list.append(valid_f1)

            print(f"EPOCH: {epoch}  ||  Elapsed: {format_time(time.time()-t0)}.")
            print(f"   Train_loss: {train_loss:.4f}  ||  Valid_acc: {valid_accuracy:.4f} | Valid_f1: {valid_f1:.4f} | Valid_loss: {valid_loss:.4f}")
            
            early_stopping(valid_loss, model)
            print("")
            if early_stopping.early_stop:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(early_stopping.path))
    test_pred_list, test_real_list = [], []
    test_accuracy, test_f1 = 0.0, 0.0
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            b_input_ids, b_attention_masks, b_token_type_ids, b_labels = tuple(t.to(args.device) for t in batch)
            out = model(b_input_ids, b_attention_masks, b_token_type_ids, labels=b_labels)
            loss, logits = out.loss, out.logits

            pred = torch.argmax(logits).cpu()
            test_pred_list.append(pred)
            test_real_list.append(b_labels.squeeze(dim=0).squeeze(dim=0).cpu())
            
        test_accuracy = accuracy_score(test_real_list, test_pred_list)
        test_f1 = f1_score(test_real_list, test_pred_list)

    print(f"Test_acc: {test_accuracy:.4f} | Test_f1: {test_f1:.4f}")
    print(classification_report(valid_real_list, valid_pred_list))

def main(args):
    #data = pd.read_parquet("../dataset_grammar.parquet")
    data = pd.read_csv(args.data_path)
    data = data.drop(index=data.index[data.Rating==2]).reset_index(drop=True)

    args['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_path == 'monologg/kobert':
        from KoBERT_Transformers.kobert_transformers.tokenization_kobert import KoBertTokenizer
        tokenizer = KoBertTokenizer.from_pretrained(args.model_path)
    else:
        args['tokenizer_path'] = args.tokenizer_path if args.tokenizer_path else args.model_path

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    train_data, valid_data = train_test_split(data, shuffle=True, stratify=data.Rating, random_state=217, test_size=0.2)
    valid_data, test_data = train_test_split(valid_data, stratify=valid_data.Rating, random_state=217, test_size=0.5)
    train_dataloader = make_dataloader(train_data, tokenizer, args.seq_length, is_train=True)
    valid_dataloader = make_dataloader(valid_data, tokenizer, args.seq_length)
    test_dataloader = make_dataloader(test_data, tokenizer, args.seq_length)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-6)
    print('Trainable Model Param # : ', sum(p.numel() for p in model.parameters() if p.requires_grad)) # in BERT, 177,854,978 / in KoElectra, 112,922,882

    early_stopping = EarlyStopping(patience = args.es_patience, verbose = True, path=f'./finetune_ckp/{args.model_path}_best_ckp.pt')

    fine_tune(model, optimizer, train_dataloader, valid_dataloader, test_dataloader, early_stopping, args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../kr3.csv', type=Path)
    parser.add_argument('--model_path', default='bert-base-multilingual-cased', help='huggingface dir or user custom dir')
    parser.add_argument('--lr', default=0.00001, type=int)
    parser.add_argument('--tokenizer_path', default=None, help="if you use user custom model, you should follow custom model's tokenizer path.")
    parser.add_argument('--output_path', default='./finetune_ckp', type=Path)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_epoch', default=50, type=int, help='Implicit epoch. Model use earlystopping which follows valid loss value.')
    parser.add_argument('--es_patience', default=10, type=int)
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--seq_length', default=256, type=int)

    args = parser.parse_args()
    main(args)