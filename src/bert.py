import logging
from mimetypes import init
from turtle import forward, hideturtle, up
import torch
import torch.nn as nn
from transformers import BertModel 
from transformers.models.bert.modeling_bert import ACT2FN, BertSelfOutput
from typing import NamedTuple, Union, Callable

logging.basicConfig(level=logging.INFO)

class AdapterConfig(NamedTuple):
    hidden_size : int 
    adapter_size : int 
    adapter_act : Union[str, Callable]
    adapter_initializer_range: float

def freeze_all_parameters(model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    return model 

class BertAdaper(nn.Module):
    def __init__(self, config: AdapterConfig):
        super(BertAdaper).__init__()
        self.down_project = torch.nn.Linear(config.hidden_size, config.adapter_size)
        nn.init.normal_(self.down_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.down_project.bias)
        
        if isinstance(config.adapter_act, str):
            self.activation = ACT2FN[config.adapter_act]
        else:
            self.activation = config.adapter_act
            
        self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
        nn.init.normal_(self.up_project.weight, std=config.adapter_initializer_range)
        nn.init.zeros_(self.up_project.bias)
    def forward(self, hidden_states: torch.Tensor):
        down_projected = self.down_project(hidden_states)
        activated = self.activation(down_projected)
        up_projected = self.up_project(activated)
        
        return hidden_states + up_projected

class BertAdapterSelfOutput(nn.Module):
    def __init__(self, config: AdapterConfig, self_output: BertSelfOutput):
        super(self, BertAdapterSelfOutput).__init__()
        self.self_out = self_output
        self.adapter = BertAdaper(config=config)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.self_out.dense(hidden_states)
        hidden_states = self.self_out.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.self_out.LayerNorm(hidden_states)
        
        return hidden_states
    
def adapter_bert_self_output(config: AdapterConfig):
    return lambda self_output : BertAdapterSelfOutput(config,self_output)

def add_bert_adapters(bert_model: BertModel, config: AdapterConfig) -> BertModel:
    for layer in bert_model.encoder.layer:
        layer.attention.output = adapter_bert_self_output(config)(layer.attention.output)
        layer.output = adapter_bert_self_output(config)(layer.output)
    return bert_model

def unfree_bert_adapters(bert_model: nn.Module) -> nn.Module:
    for name, sub_module in bert_model.modules():
        if isinstance(sub_module, (BertAdaper,nn.LayerNorm)):
            for param_name, param in sub_module.named_parameters():
                param.requires_grad = True
    return bert_model