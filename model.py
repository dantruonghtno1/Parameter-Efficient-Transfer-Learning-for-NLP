from pathlib import Path
from this import d 
from typing import List
import pytorch_lightning as pl
import torch
import torch.nn as nn 
from sklearn.metrics import accuracy_score
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

class AutoModelForSeqClassificationFinetuner(pl.LightningDataModule):
    def __init__(self, 
                 model_name: str, 
                 n_classes : int , 
                 max_length : int = 100, 
                 lr : float = 2e-5 , 
                 eps : float = 1e-8): 
        super(AutoModelForSeqClassificationFinetuner, self).__init__()
        self.max_length = max_length 
        self.lr = lr 
        self.eps = eps 
        
        config = AutoConfig.from_pretrained(model_name,
                                            num_labels = n_classes, 
                                            output_attentions = False, 
                                            output_hidden_states = False,
                                            torchscript = True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config = config)
    def forward(self, input_ids, attention_mask = None, labels = None):
        output = self.model(input_ids, attention_mask = attention_mask, labels = labels )
        return output 
    
    def tokenize(self, texts: List[str]):
        x_tokenized = self.tokenizer(texts, padding = True, 
                                     truncation = True, max_legth = self.max_length, 
                                     return_tensors = 'pt')
        input_ids = x_tokenized['input_ids'].to(self.device)
        attention_mask = x_tokenized['attention_mask'].to(self.device)
        return input_ids, attention_mask 
    
    def compute(self, batch):
        x = batch['x']
        y = batch['y']
        
        loss, logits = self(*self.tokenize(x), labels = y) 
        return loss, logits 
    
    def training_step(self, batch, batch_nb):
        loss, logits = self.compute(batch)
        return {
            'loss': loss
        }
    def validation_step(self, batch, batch_nb):
        loss, logits = self.compute(batch)
        y = batch['y']
        a, y_hat = torch.max(logits, dim = 1)
        val_acc = accuracy_score(y_hat.cpu(), y.cpu())
        return {
            'val_loss': loss, 
            'vall_acc': torch.tensor(val_acc)
        }
        
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack(x['val_acc'] for x in outputs).mean()
        
        self.log('val_loss', avg_loss, on_epoch = True, prog_bar = True, sync_dict = True)
        self.log('val_acc', avg_val_acc, on_epoch = True, prog_bar = True, sync_dict = True)
        
    def test_step(self, batch, batch_nb):
        loss, logits = self.compute(batch)
        
        y = batch['y']
        a, y_hat = torch.max(logits, dim = 1)
        test_acc = accuracy_score(y_hat.cpu(), y.cpu())
        return {
            'test_acc': torch.tensor(test_acc)
        }
        
    def test_epoch_end(self, outputs):
        acg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log('avg_test_acc', acg_test_acc, on_epoch = True, prog_bar = True, sync_dist = True)
        
    def confure_optimizers(self):
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], 
            lr = self.lr, 
            eps = self.eps 
        )
    def save_inference_artifact(self, output_dir : str):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        # save tokenizer 
        self.tokenizer.save_vocabulary(output_dir)
        #save torchscript model 
        self.cpu().eval()
        dummy_input_ids, _ = self.tokenizer(["simply dummy text to be use for tracing"])
        self.to_torchscript(file_path = str(output_dir/"model.pt"), 
                            method = "trace" , 
                            example_inputs = (dummy_input_ids,))
        
    
        