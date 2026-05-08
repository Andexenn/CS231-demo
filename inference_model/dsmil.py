import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IClassifier(nn.Module):
    def __init__(self, feature_size, output_class):
        super(IClassifier, self).__init__()
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        c = self.fc(x)
        return x, c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, nonlinear=True):
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Identity()
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):
        V = self.v(feats)
        Q = self.q(feats).view(feats.shape[0], -1)
        _, m_indices = torch.sort(c, 0, descending=True)
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        q_max = self.q(m_feats)
        A = torch.mm(Q, q_max.transpose(0, 1))
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=feats.device)), 0)
        B = torch.mm(A.transpose(0, 1), V)
        B = B.view(1, B.shape[0], B.shape[1])
        C = self.fcc(B)
        C = C.view(1, -1)
        return C, A, B

class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier, in_feats=1024, reduced_feats=512):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.feature_extractor = nn.Sequential(nn.Linear(in_feats, reduced_feats), nn.ReLU())

    def forward(self, x):
        x = self.feature_extractor(x)
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        return classes, prediction_bag, A, B

def load_model(checkpoint_path, device='cpu', in_feats=1024, reduced_feats=512, output_class=2):
    model = MILNet(
        IClassifier(reduced_feats, output_class),
        BClassifier(reduced_feats, output_class),
        in_feats=in_feats,
        reduced_feats=reduced_feats
    )
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
        
    model.to(device).eval()
    return model

def predict(model, bag_features, device='cpu'):
    """
    bag_features: Tensor of shape (num_patches, in_feats)
    Returns: prediction (int), probabilities, attention weights
    """
    bag_features = bag_features.squeeze(0).to(device)
    with torch.no_grad():
        ins_scores, bag_prediction, A, B = model(bag_features)
        max_prediction, _ = torch.max(ins_scores, 0)
        
        # Final prediction is average of both streams (max-pooling & attention)
        score = 0.5 * torch.sigmoid(bag_prediction) + 0.5 * torch.sigmoid(max_prediction.view(1, -1))
        probs = score.squeeze().cpu().numpy()
        pred = np.argmax(probs)
        
    return int(pred), probs, A.squeeze().cpu().numpy()
