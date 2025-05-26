'''import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm


# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
num_epochs = 5
learning_rate = 1e-4
img_size = 224

# Initialize models
teacher = TeacherModel().to(device)
teacher.load_state_dict(torch.load('path/to/pretrained_weights.pth'))
teacher.eval()

student = StudentWithUncertainty().to(device)
optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

# Dataset and transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Use ImageNet or CIFAR-100 for testing
dataset = datasets.CIFAR100(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    student.train()
    total_loss = 0.0
    uncertainties = []
    
    for images, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        
        with torch.no_grad():
            teacher_features = teacher(images)
            
        student_mean, student_logvar = student(images)
        loss = heteroscedastic_loss(student_mean, student_logvar, teacher_features)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        uncertainties.extend(torch.exp(student_logvar).mean(dim=1).cpu().numpy())
    
    # Visualization after epoch
    print(f"\nEpoch {epoch+1} Results:")
    print(f"Average Loss: {total_loss/len(dataloader):.4f}")
    print(f"Mean Uncertainty: {np.mean(uncertainties):.4f}")
    
    # Visualize sample predictions
    with torch.no_grad():
        sample_images, _ = next(iter(dataloader))
        sample_images = sample_images.to(device)
        _, logvar = student(sample_images)
        plot_uncertainty(sample_images.cpu(), logvar.cpu())  # Implement visualization

# Save final model
torch.save(student.state_dict(), 'student_with_uncertainty.pth')'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.datasets.cifar import CIFAR100Subset  # Add `src.` prefix  # Replace with your dataset class
from src.models.student import Student  
from src.models.teacher import Teacher
from src.utils.uncertainty import compute_uncertainty
# Hyperparameters
batch_size = 32
epochs = 10
lr = 3e-4
mask_ratio = 0.5
image_size = 224  # ViT input size
patch_size = 16
num_patches = (image_size // patch_size) ** 2  # Calculate patches

# Initialize models
student = Student()
teacher = Teacher(student)  # Now works with full model copy
optimizer = torch.optim.Adam(student.parameters(), lr=lr)
from torchvision import transforms

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize dataset
train_dataset = CIFAR100Subset(
    root="./data",
    train=True,
    max_samples=1000,
    download=True,
    transform=train_transform  # Pass the transforms here
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



def get_uncertainty_masks(student, teacher, images, mask_ratio=0.5):
    with torch.no_grad():
        # Forward pass through student (MC-Dropout)
        student_outs = student(images)  # Shape: [num_samples * B, embed_dim]
        
        # Get dynamic dimensions
        num_samples = 5  # Must match Student.forward() passes
        batch_size = images.size(0)  # Use ACTUAL batch size (not hardcoded 32)
        embed_dim = student_outs.shape[-1]
        
        # Reshape for class tokens (not patches)
        student_outs = student_outs.view(num_samples, batch_size, 1, embed_dim)
        # Shape: [num_samples, B, 1, embed_dim]

        # Compute uncertainty (variance across samples)
        uncertainty = torch.var(student_outs, dim=0)  # [B, 1, embed_dim]
        uncertainty = uncertainty.mean(dim=-1)  # [B, 1]

        # Since there's only 1 "patch" (class token), mask it entirely
        k = int(mask_ratio * 1)  # mask_ratio of the single "patch"
        _, mask_indices = torch.topk(uncertainty, k=k, dim=-1)  # [B, k]

        return mask_indices

# Training loop
# Training loop
for epoch in range(epochs):
    for images, _ in train_dataloader:
        batch_size = images.size(0)  # Get actual batch size
        
        # Generate masks
        mask = get_uncertainty_masks(student, teacher, images, mask_ratio)
        
        # Forward pass (student returns [num_samples, B, 1, embed_dim])
        student_pred = student(images)  # [num_samples, B, 1, embed_dim]
        
        # Teacher target (class token)
        with torch.no_grad():
            teacher_target = teacher(images)  # [B, 1, embed_dim]
        
        # Compute loss for masked "patches" (class token)
        loss = 0
        for s_pred in student_pred:  # Iterate over MC-Dropout samples
            loss += F.mse_loss(s_pred[:, :, :], teacher_target[:, :, :])
        loss /= student_pred.size(0)  # Average over samples
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        teacher.update_ema(student)


# Inside training loop:
mask = get_uncertainty_masks(...)  # [B, k]

# Student prediction on masked patches
student_pred = student(images)  # [num_samples * B, num_patches, embed_dim]
student_pred = student_pred.view(num_samples, batch_size, num_patches, embed_dim)

# Teacher target (non-masked)
with torch.no_grad():
    teacher_target = teacher(images)  # [B, num_patches, embed_dim]

# Compute loss for masked patches
loss = 0
for s_pred in student_pred:  # Iterate over MC-Dropout samples
    # Gather masked patches for each sample
    s_pred_masked = s_pred[torch.arange(batch_size).unsqueeze(1), mask]  # [B, k, embed_dim]
    t_target_masked = teacher_target[torch.arange(batch_size).unsqueeze(1), mask]  # [B, k, embed_dim]
    loss += F.mse_loss(s_pred_masked, t_target_masked)
loss /= num_samples  # Average over MC-Dropout samples





