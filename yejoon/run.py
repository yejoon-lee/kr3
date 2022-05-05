from datasets import DatasetDict
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import time
import os
from typing import Optional
import loralib as lora
from utils import ConfusionMatrix  # local library

class RunFreeze:
    '''Wandb run with some of the BERT layers freezed. BERT model is from hugging face.
    Also the base class for other runs: RunLora and RunAdapter.
    Params:
        config: wandb config; the hyperparams and additional configs
        project: project name
        name: name of run
        notes: notes to add for this run
        tags: tags to add for this run

    Config: (config dictionary should have the following as its items)
        num_freeze:int, number of BERT layers to be freezed (0 for full model, maximum 12 layers can be freezed)
        lr:float, learning rate
        batch_size:int, batch size
        weight_decay:float, weight decay
        hgf_checkpoint:str, name of hugging face model
        num_epochs:int, total number of epochs
        n_log:int, training loss on batch is logged every 'n_log' batches. trainig loss on epoch is logged seperately
    '''
    def __init__(self, 
                config : dict,
                project : Optional[str] = None,
                name : Optional[str] = None,
                notes : Optional[str] = None,
                tags : Optional[list] = None):
        with wandb.init(config=config, project=project, name=name, notes=notes, tags=tags):
            self.pipeline()

    def pipeline(self):
        self.check_env()
        self.make()
        self.freeze()
        self.train()
        self.save_model()

    def check_env(self, use_adapter=False):
        '''Verify if transformer package is right. 'adapter-transformer' for RunAdapter, 
        vanilla 'transformer' otherwise. See RunAdapter class for precise context.'''
        if 'AdapterConfig' in dir(transformers) == use_adapter:
            raise EnvironmentError('Check whether transformer package is adapter-transformer or not.')

    def make(self):
        '''Basic setup for training'''
        # load tokenizer and model from hugging face
        tokenizer = BertTokenizer.from_pretrained(wandb.config.hgf_checkpoint)
        self.model = BertForSequenceClassification.from_pretrained(wandb.config.hgf_checkpoint, num_labels=2)

        # load tokenized dataset from local
        tokenized_dataset = DatasetDict().load_from_disk('tokenized')

        # pytorch dataloder with dynamic padding (implemented using hugging face API)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self.train_dloader = DataLoader(tokenized_dataset['train'], batch_size=wandb.config.batch_size, shuffle=True, collate_fn=data_collator)
        self.val_dloader = DataLoader(tokenized_dataset['test'], batch_size=wandb.config.batch_size, shuffle=False, collate_fn=data_collator)

    def freeze(self):
        '''Freeze BERT layers in the model'''
        # freeze BERT layers
        for i in range(wandb.config.num_freeze):
            for p in self.model.bert.encoder.layer[i].parameters():
                p.requires_grad = False

        # always freeze embedding if ANY BERT layer is freezed
        if wandb.config.num_freeze > 0:
            for p in self.model.bert.embeddings.parameters():
                p.requires_grad = False

    def train(self):
        '''Training and evaluation'''
        # put model on device
        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)
        # train configs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay) 
        # log model size
        wandb.run.summary['num_trainable_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # set initial values
        print("Start training")
        batch_idx = 0
        running_loss_batch = 0.0
        best_accuracy = 0.0
        best_f1 = 0.0

        for epoch in range(wandb.config.num_epochs):
            print(f'Epoch {epoch+1}')
            # to measure elapsed time
            t0 = time.time()

            # train
            running_loss_epoch = 0.0
            self.model.train()
            for batch in tqdm(self.train_dloader):
                batch_idx += 1

                # forward
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss_batch += loss.item()
                running_loss_epoch += loss.item()

                # log training loss on batches
                if batch_idx % wandb.config.n_log == 0:
                    wandb.log({'batch':batch_idx, 'loss':running_loss_batch / wandb.config.n_log})
                    running_loss_batch = 0.0

            # elapsed time
            elapsed_time = time.time() - t0

            # eval
            print("Start evaluation")
            self.model.eval()
            with torch.no_grad():
                val_running_loss = 0.0
                confusion_matrix = torch.zeros(2, 2, dtype=torch.int32)
                for batch in tqdm(self.val_dloader):
                    # forward
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    loss = outputs.loss

                    # loss
                    val_running_loss += loss.item()

                    # confusion matrix
                    for real, pred in zip(batch['labels'].cpu(), outputs.logits.argmax(axis=1).cpu()):
                        confusion_matrix[real][pred] += 1

            # log history
            reports = ConfusionMatrix(confusion_matrix)
            wandb.log({'epoch':epoch+1,
                        'train_loss':running_loss_epoch / len(self.train_dloader),
                        'val_loss':val_running_loss / len(self.val_dloader),
                        'accuracy':reports.accuracy,
                        'precision':reports.precision,
                        'recall':reports.recall,
                        'f1':reports.f1,
                        'elapsed_time':elapsed_time})

            # log summary
            if reports.accuracy > best_accuracy:
                best_accuracy = reports.accuracy
                wandb.run.summary['best_accuracy'] = best_accuracy
            if reports.f1 > best_f1:
                best_f1 = reports.f1
                wandb.run.summary['best_f1'] = best_f1

    def save_model(self):
        '''Save the model as wandb Artifact'''
        # create artifact
        artifact = wandb.Artifact("finetuned_BERT", type='model',
                                  description="Finetuned BERT with some layers freezed",
                                  metadata=dict(wandb.config))

        # save the model into the artifact
        with artifact.new_file(f'{wandb.run.name}.pth', 'wb') as f:
            torch.save(self.model.state_dict(), f)

        # log artifact and add a tag
        wandb.run.log_artifact(artifact)
        wandb.run.tags = wandb.run.tags + ('artifacted',)


class RunLora(RunFreeze):
    '''Wandb run with BERT model which LoRA is applied on. BERT model is from hugging face.
    https://github.com/microsoft/lora
    Params:
        config: wandb config; the hyperparams and additional configs
        project: project name
        name: name of run
        notes: notes to add for this run
        tags: tags to add for this run

    Config: (config dictionary should have the following as its items)
        lora_r:int, r for LoRA
        lora_module_names:iterable, submodules in self attention which LoRA would be applied. Subset of ['query', 'key', 'value']
        lr:float, learning rate
        batch_size:int, batch size
        weight_decay:float, weight decay
        hgf_checkpoint:str, name of hugging face model
        num_epochs:int, total number of epochs
        n_log:int, training loss on batch is logged every 'n_log' batches. trainig loss on epoch is logged seperately
    '''
    def pipeline(self):
        self.check_env()
        self.make()
        self.lora()
        self.train()
        self.save_model()

    def freeze(self):
        raise NotImplementedError # prevent misuse of the class

    def lora(self):
        '''Apply LoRA to the model, i.e. replace some of the layers in the model to LoRA layer.'''
        for layer in self.model.bert.encoder.layer:
            self_attn = layer.attention.self
            for name in wandb.config.lora_module_names:
                # make lora_module with same in and out dim with pretrained module
                pt_module = self_attn.get_submodule(name)
                lora_module = lora.Linear(pt_module.in_features, pt_module.out_features, r=wandb.config.lora_r)

                # copy pretrained params to lora_module
                lora_module.weight = pt_module.weight
                lora_module.bias = pt_module.bias

                # replace
                self_attn._modules[name] = lora_module

                # check if lora is successfully applied
                assert hasattr(self_attn.get_submodule(name), 'lora_A'), 'LoRA not applied successfully'

    def train(self):
        lora.mark_only_lora_as_trainable(self.model) # only lora params are trained
        super().train()

    def save_model(self):
        '''Save the model(only LoRA params) as wandb Artifact using methods from loralib.'''
        # create artifact
        artifact = wandb.Artifact("LoRA_BERT", type='model',
                                  description="Tuned LoRA params. Not containing pretrained params. Later load with strict=False",
                                  metadata=dict(wandb.config))

        # save the LoRA params into the artifact
        with artifact.new_file(f'{wandb.run.name}.pth', 'wb') as f:
            torch.save(lora.lora_state_dict(self.model), f)

        # log artifact and add a tag
        wandb.run.log_artifact(artifact)
        wandb.run.tags = wandb.run.tags + ('artifacted',)


class RunAdapter(RunFreeze):
    '''Wandb run with BERT model and adapter. BERT model is from hugging face.
    https://adapterhub.ml/
    Params:
        config: wandb config; the hyperparams and additional configs
        project: project name
        name: name of run
        notes: notes to add for this run
        tags: tags to add for this run

    Config: (config dictionary should have the following as its items)
        adapter_dim:int, bottleneck dimension of adapter
        lr:float, learning rate
        batch_size:int, batch size
        weight_decay:float, weight decay
        hgf_checkpoint:str, name of hugging face model
        num_epochs:int, total number of epochs
        n_log:int, training loss on batch is logged every 'n_log' batches. trainig loss on epoch is logged seperately
    '''
    def pipeline(self):
        self.check_env(use_adapter=True)
        self.make()
        self.adapter()
        self.train()
        self.save_model()

    def freeze(self):
        raise NotImplementedError  # prevent misuse of the class

    def adapter(self):
        '''Add adapters to the model using adapter-transformers. Use config from Houlsby et al., 2019.
        Set adapter dim by wandb config. Note that method HoulsbyConfig recieves reduction_factor as input.'''
        from transformers import HoulsbyConfig
        adapter_config = HoulsbyConfig(reduction_factor = 768 // wandb.config.adapter_dim) # 768 is the dimension in a transformer block of BERT-base.
        self.model.add_adapter('kr3', adapter_config)

    def train(self):
        self.model.train_adapter('kr3') # only adapters are trained
        self.model.set_active_adapters('kr3') # activate adapter
        super().train()

    def save_model(self):
        '''Save the model(only adapters) as wandb Artifact. Later load by model.load_adapters'''
        # create artifact
        artifact = wandb.Artifact("Adapter_BERT", type='model',
                                  description="Tuned adapters. Not containing pretrained params.",
                                  metadata=dict(wandb.config))

        # save the adapters into the artifact
        os.makedirs('adapters', exist_ok=True)
        self.model.save_all_adapters('adapters')
        artifact.add_dir('adapters')

        # log artifact and add a tag
        wandb.run.log_artifact(artifact)
        wandb.run.tags = wandb.run.tags + ('artifacted',)