from transformers import Trainer
from torch import nn
import torch
from typing import Optional


class DistillationLoss(nn.Module):
    def __init__(self, W_dist: float=1.0, W_cos: float=0.0, W_task: float=0.0, temperature: float=1.0):
        super(DistillationLoss, self).__init__()
        self.W_dist = W_dist
        self.W_cos = W_cos
        self.W_task = W_task
        self.temperature = temperature
        self.dist_loss = nn.KLDivLoss(reduction="batchmean")
        self.cos_loss = nn.CosineEmbeddingLoss(reduction="mean")
        self.task_loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, student_logits, teacher_logits, outputs: Optional[torch.tensor]=torch.tensor([1],dtype=torch.float32), labels: Optional[torch.tensor]=torch.tensor([1],dtype=torch.float32)):
        student_softmax = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_softmax = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
        distillation_loss = self.W_dist * self.dist_loss(student_softmax, teacher_softmax) * (self.temperature ** 2)
        cosine_embed_loss = self.W_cos * self.cos_loss(student_logits, teacher_logits)
        task_specific_loss = self.W_task * self.task_loss(outputs, labels)
        total_loss = distillation_loss + cosine_embed_loss + task_specific_loss
        return total_loss

class DistilTrainer(Trainer):
    def __init__(self, teacher_model, W_dist=1.0, W_cos=1.0, W_task=1.0, temperature=2.0, *args, **kwargs):
        super(DistilTrainer, self).__init__()
        self.W_dist = W_dist
        self.W_cos = W_cos
        self.W_task = W_task
        self.temperature = temperature
        self.teacher_model = teacher_model
        assert W_cos >=0.0, "what is this nonsense"
        assert W_dist >=0.0, "what is this nonsense"
        assert W_task >=0.0, "what is this nonsense"

    def compute_loss(self, student_model, inputs, return_outputs=False):
        student_model.train()
        labels = inputs.pop("labels")
        student_logits = student_model(**inputs).logits
        outputs = student_model(**inputs)
        with torch.no_grad():
            teacher_logits = self.teacher_model(**inputs).logits
        loss_fct = DistillationLoss(self.W_dist, self.W_cos, self.W_task, self.temperature)
        loss = loss_fct(student_logits, teacher_logits, outputs, labels)
        student_model.train()
        return loss