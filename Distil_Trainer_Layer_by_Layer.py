from transformers import training_args
from transformers import Trainer
from torch import nn
import torch
from typing import Optional


class LayerDistillationLoss(nn.Module):
    def __init__(self, W_dist: float=1.0, W_cos: float=0.0, temperature: float=1.0):
        super(LayerDistillationLoss, self).__init__()
        self.W_dist = W_dist
        self.W_cos = W_cos
        self.temperature = temperature
        self.dist_loss = nn.KLDivLoss(reduction="batchmean")
        self.cos_loss = nn.CosineEmbeddingLoss(reduction="mean")
        self.task_loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, student_outputs, teacher_outputs):
        student_softmax = nn.functional.log_softmax(student_outputs / self.temperature, dim=-1)
        teacher_softmax = nn.functional.softmax(teacher_outputs / self.temperature, dim=-1)
        distillation_loss = self.W_dist * self.dist_loss(student_softmax, teacher_softmax) * (self.temperature ** 2)
        cosine_embed_loss = self.W_cos * self.cos_loss(student_outputs, teacher_outputs)
        total_loss = distillation_loss + cosine_embed_loss
        return total_loss
                
class LayerDistilTrainer(Trainer):
    def __init__(self, teacher_model: nn.Module, target_layer: int, W_dist=1.0, W_cos=1.0, temperature=2.0, *args, **kwargs):
        super(LayerDistilTrainer, self).__init__()
        self.W_dist = W_dist
        self.W_cos = W_cos
        self.W_task = 0
        self.temperature = temperature
        self.teacher_model = teacher_model
        self.target_layer = target_layer
        assert type(target_layer)==int, "target layer has to be an integer"
        assert W_cos >=0.0, "what is this nonsense"
        assert W_dist >=0.0, "what is this nonsense"

    def compute_loss(self, student_model, inputs):
        teacher_model = self.teacher_model
        student_model.train()
        assert self.target_layer < student_model.config.num_hidden_layers, "the target layer exceeds the number of layers the student has!"

        for params in student_model.parameters:
            params.requires_grad=False

        for params in student_model.distilbert.transformer.layer[self.target_layer]:
            params.requires_grad=True
        
        teacher_layer = (teacher_model.config.num_hidden_layers * self.target_layer) // student_model.config.num_hidden_layers

        student_outputs = student_model(**inputs).hidden_states[self.target_layer]
        teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs).hidden_states[teacher_layer]
        loss_fct = LayerDistillationLoss(student_model, teacher_model, self.target_layer, inputs, self.W_dist, self.W_cos, self.temperature)
        loss = loss_fct(student_outputs, teacher_outputs)
        student_model.train()
        return loss