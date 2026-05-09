import torch
import torch.nn as nn

class CEMIL(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, output_class=2):
        super(CEMIL, self).__init__()
        self.prm = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.context_mod = nn.Parameter(torch.ones(1, hidden_size))
        self.attention_V = nn.Linear(hidden_size, 128)
        self.attention_U = nn.Linear(hidden_size, 128)
        self.attention_weights = nn.Linear(128, 1)
        self.classifier = nn.Linear(hidden_size, output_class)

    def forward(self, x):
        h = self.prm(x)
        h_prime = h * self.context_mod
        A_V = torch.tanh(self.attention_V(h_prime))
        A_U = torch.sigmoid(self.attention_U(h_prime))
        A = self.attention_weights(A_V * A_U)
        A = torch.softmax(A, dim=0)
        z = torch.sum(A * h, dim=0, keepdim=True)
        logits = self.classifier(z)
        return logits, A, h

def load_models(instructor_ckpt, learner_ckpt, device='cpu', input_size=1024, hidden_size=512, output_class=2):
    instructor = CEMIL(input_size=input_size, hidden_size=hidden_size, output_class=output_class)
    learner = CEMIL(input_size=input_size, hidden_size=hidden_size, output_class=output_class)
    
    def _load(model, path):
        ckpt = torch.load(path, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
            
    _load(instructor, instructor_ckpt)
    _load(learner, learner_ckpt)
    
    instructor.to(device).eval()
    learner.to(device).eval()
    
    return instructor, learner

def predict(instructor, learner, bag_features, k_ratio=0.6, device='cpu'):
    """
    bag_features: Tensor of shape (num_patches, input_size)
    Returns: prediction (int), label (str), probabilities, top indices selected, instructor attention, learner attention
    """
    class_map = {0: "LUAD", 1: "LUSC"}
    bag_features = bag_features.squeeze(0).to(device)
    with torch.no_grad():
        # Get attention from instructor
        _, A_I, _ = instructor(bag_features)
        num_patches = bag_features.shape[0]
        k_patches = max(1, int(num_patches * k_ratio))
        _, top_indices = torch.topk(A_I.squeeze(), k_patches)
        top_bag = bag_features[top_indices]
        
        # Learner predicts using the top-k patches
        logits, A_L, _ = learner(top_bag)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        
    return pred, class_map[pred], probs.squeeze().cpu().numpy(), top_indices.cpu().numpy(), A_I.squeeze().cpu().numpy(), A_L.squeeze().cpu().numpy()
