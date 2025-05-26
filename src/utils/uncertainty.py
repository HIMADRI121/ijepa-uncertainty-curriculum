'''import torch
import torch.nn.functional as F

def cosine_uncertainty(student_embeds, teacher_embeds):
    similarity = F.cosine_similarity(student_embeds, teacher_embeds, dim=-1)
    uncertainty = 1 - similarity  # Higher = more uncertain
    return uncertainty
'''
import torch
import torch.nn as nn
import copy  
# utils/uncertainty.py
def compute_uncertainty(embeddings):
    # embeddings: [num_samples, B, num_patches, embed_dim]
    variance = torch.var(embeddings, dim=0)  # Variance over samples
    return variance.mean(dim=-1)  # Average over embedding dimension