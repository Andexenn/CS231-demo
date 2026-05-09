import torch
import torch.nn as nn
import torch.nn.functional as F

class ABMIL(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, output_class=2):
        super(ABMIL, self).__init__()
        self.L = hidden_size
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(input_size, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, output_class)
        )

    def forward(self, x):
        x = x.squeeze(0) if len(x.shape) > 2 else x
        H = self.feature_extractor_part1(x)
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)
        logits = self.classifier(M)
        return logits, A, H

def load_model(checkpoint_path, device='cpu', input_size=1024, hidden_size=512, output_class=2):
    model = ABMIL(input_size=input_size, hidden_size=hidden_size, output_class=output_class)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model

def predict(model, bag_features, device='cpu'):
    """
    bag_features: Tensor of shape (num_patches, input_size)
    Returns: prediction (int), label (str), probabilities (numpy array), attention weights (numpy array)
    """
    class_map = {0: "LUAD", 1: "LUSC"}
    bag_features = bag_features.to(device)
    with torch.no_grad():
        logits, A, _ = model(bag_features)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, class_map[pred], probs.squeeze().cpu().numpy(), A.squeeze().cpu().numpy()
