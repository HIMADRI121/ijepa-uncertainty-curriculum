'''import torch.nn as nn

class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Implement I-JEPA architecture
        self.encoder = nn.Sequential(
            # Your implementation here
        )
        
    def forward(self, x):
        return self.encoder(x)'''
import timm
import torch
import torch.nn as nn
import copy  # Required for deepcopy
import copy
import torch
import torch.nn as nn

class Teacher(nn.Module):
    def __init__(self, student):
        super().__init__()
        # Create a full copy of the student (not just a "backbone")
        self.model = copy.deepcopy(student)
        self.model.requires_grad_(False)  # Freeze teacher
        
    def update_ema(self, student, decay=0.999):
        # Update teacher weights via EMA
        with torch.no_grad():
            for t_param, s_param in zip(self.model.parameters(), student.parameters()):
                t_param.data.mul_(decay).add_(s_param.data, alpha=1-decay)
    
    def forward(self, x):
        return self.model(x)
