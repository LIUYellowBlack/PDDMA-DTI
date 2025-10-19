import os
import sys
from datetime import datetime

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.model_selection import StratifiedKFold
import argparse
import pickle as pkl
import torch
import torch.nn.functional as F
import numpy as np
from xgboost import XGBClassifier

from utils import *
from physics_diffusion import (
    globalPhysicsInformedDiffusion,
    globalphysics_informed_aver,
    localPhysicsDiffusion,
    local_physics_aver,
)
from model import DNN

parser = argparse.ArgumentParser(description="RUN PIGE-DTI TRAINING")
parser.add_argument('--device', default='cuda:0', type=str, help='Device for Training')
parser.add_argument('--dataset', default='human', type=str, help='Dataset to use')

parser.add_argument('--diffusion_coef', type=float, default=3, help='Diffusion coefficient')
parser.add_argument('--physics_steps', type=int, default=60, help='Physics diffusion steps')
parser.add_argument('--local_diffusion', default='True', help='Enable local physics diffusion (flag)')
parser.add_argument('--local_diffusion_coef', type=float, default=0.5, help='local diffusion coefficient')
parser.add_argument('--local_steps', type=int, default=5, help='local diffusion steps')
parser.add_argument('--local_alpha', type=float, default=0.8, help='local preservation coefficient')

inputs = parser.parse_args()
log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("log", exist_ok=True)
log_path = os.path.join("log", f"{inputs.dataset}_{log_time}.txt")

# Custom Logger class (prints to both stdout and log file)
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_path)

pd.set_option('display.max_rows', 10)

# Dataset node count
if inputs.dataset == 'bindingdb':
    drug_num = 14643
    protein_num = 2623
elif inputs.dataset == 'DrugBank1.4':
    drug_num = 549
    protein_num = 424
else:
    num = pkl.load(open(f'../data/{inputs.dataset}/num.pkl', 'rb'))
    drug_num = num['drug_num']
    protein_num = num['prot_num']

print(f"Start Running on Dataset: {inputs.dataset}")

# Load data
AllNode = pd.read_csv(f'../data/{inputs.dataset}/Allnode_DrPr.csv', names=[0, 1], skiprows=1)
Alledge = pd.read_csv(f'../data/{inputs.dataset}/DrPrNum_DrPr.csv', header=None)
prot_edge = pd.read_csv(f'../data/{inputs.dataset}/prot_edge.csv')
drug_edge = pd.read_csv(f'../data/{inputs.dataset}/drug_edge.csv')
drug_prot_edge = pd.read_csv(f'../data/{inputs.dataset}/drug_prot_edge.csv')
features = pd.read_csv(f'../data/{inputs.dataset}/AllNodeAttribute_DrPr.csv', header=None)
features = features.iloc[:, 1:]

labels = pd.DataFrame(np.random.rand(len(AllNode), 1))
labels[:drug_num] = 0
labels[drug_num:] = 1
labels = labels[0]

adj, features, labels, idx_train, idx_val, idx_test = load_data(drug_prot_edge, features, labels)

class item:
    def __init__(self):
        self.k1 = inputs.physics_steps
        self.k2 = 10
        self.epsilon1 = 0.03
        self.epsilon2 = 0.05
        self.hidden = 64
        self.dropout = 0.8
        self.runs = 1
        self.diffusion_coef = inputs.diffusion_coef
        self.physics_method = 'implicit'
        # local diffusion parameters
        self.local_diffusion = inputs.local_diffusion
        self.local_diffusion_coef = inputs.local_diffusion_coef
        self.local_steps = inputs.local_steps
        self.local_alpha = inputs.local_alpha

args = item()

# Initialize physics diffusion
physics_diffuser = globalPhysicsInformedDiffusion(
    diffusion_coef=args.diffusion_coef,
    time_steps=args.k1,
    method=args.physics_method
)

if args.local_diffusion:
    local_diffuser = localPhysicsDiffusion(
        diffusion_coef=args.local_diffusion_coef,
        time_steps=args.local_steps,
        method=args.physics_method,
        alpha=args.local_alpha
    )
    print("local physics diffusion enabled!")
    print(f"Local diffusion parameters: coef={args.local_diffusion_coef}, steps={args.local_steps}, alpha={args.local_alpha}")

# Feature normalization (consistent with original script)
features = F.normalize(features, p=1)

# Physics diffusion process
print("Starting DTI physics diffusion process...")
hops, feature_list = physics_diffuser.adaptive_diffusion(
    features, adj, epsilon=args.epsilon1
)
print("DTI physics diffusion completed.")

input_feature = globalphysics_informed_aver(hops, adj, feature_list, alpha=0.15)
print(f"Input feature shape: {input_feature.shape}")

# Device and model initialization - consistent with first script: no DNN training (only one forward pass to get embeddings)
device = torch.device(inputs.device if torch.cuda.is_available() else "cpu")
n_class = 64  # Consistent embedding dimension with first script
input_feature = input_feature.to(device)

print("Projecting DTI embeddings to 64 dimensions through DNN...")

model = DNN(features.shape[1], args.hidden, n_class, args.dropout).to(device)
model.eval()

with torch.no_grad():
    output, embeddings = model(input_feature)
if args.local_diffusion:
    probs = torch.softmax(output, dim=1)
    hops_local, local_list = local_diffuser.adaptive_label_diffusion(
        probs.cpu(), adj, None, epsilon=0.01
    )
    diffused_probs = local_physics_aver(hops_local, adj, local_list, args.local_alpha)
    prob_embeddings = torch.log(diffused_probs + 1e-8)
    if prob_embeddings.shape[0] == embeddings.shape[0]:
        pe = prob_embeddings.to(device)
        embeddings = torch.cat([embeddings, pe], dim=1)
        print("Probability information from local diffusion concatenated into embeddings.")
    else:
        print("Warning: Local diffusion output row count does not match embeddings, skipping concatenation.")

# Save embeddings as CSV (by node order)
Emdebding_GCN = pd.DataFrame(embeddings.detach().cpu().numpy())
embedding_filename = f'../data/{inputs.dataset}/Physics_GCN_Node_Embeddings.csv'
if args.local_diffusion:
    embedding_filename = f'../data/{inputs.dataset}/Physics_localDiffusion_Node_Embeddings.csv'

os.makedirs(os.path.dirname(embedding_filename), exist_ok=True)
Emdebding_GCN.to_csv(embedding_filename, header=None, index=False)
print(f"Physics diffusion DTI embeddings saved: {embedding_filename}")

# ---------- GBDT Classification Part (consistent with original script) ----------
Positive = Alledge
AllNegative = pd.read_csv(f'../data/{inputs.dataset}/AllNegative_DrPr.csv', header=None)
n_num = len(AllNegative)
print(f"positive edge num: {len(Positive)}, negative edge num: {len(AllNegative)}")
Negative = AllNegative.sample(n=n_num, random_state=520)
Positive[2] = Positive.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
Negative[2] = Negative.apply(lambda x: 0 if x[0] < 0 else 0, axis=1)
result = pd.concat([Positive, Negative]).reset_index(drop=True)

# Use generated embeddings to build GBDT feature pairs
X = pd.concat([Emdebding_GCN.loc[result[0].values.tolist()].reset_index(drop=True),
               Emdebding_GCN.loc[result[1].values.tolist()].reset_index(drop=True)], axis=1)
Y = result[2]

# ---------- Physics Interaction Sum Enhancement ----------
beta = 0.01  # Nonlinear strength, default 0.1
embeddings_np = embeddings.detach().cpu().numpy()  # Convert to numpy

# Fix: Normalize embeddings to avoid overflow and invalid values
embeddings_tensor = torch.from_numpy(embeddings_np)
embeddings_np = F.normalize(embeddings_tensor, p=2, dim=1).numpy()

# Fix: Use len(result) instead of len(Alledge), and calculate based on result
interaction_sum = np.zeros(len(result))
for idx in range(len(result)):
    d_idx, p_idx = result.iloc[idx, 0], result.iloc[idx, 1]
    inter = np.sum(embeddings_np[d_idx] * embeddings_np[p_idx])  # Inner product sum, simulating physics interaction
    interaction_sum[idx] = inter + beta * (inter ** 2)  # Add quadratic term, nonlinear
print("Physics interaction sum enhancement completed")

# Add to X
X['phys_inter'] = interaction_sum

NmedEdge = 499
DmedEdge = 7
SmedEdge = 0.85

k_fold = 10
print("%d fold CV" % k_fold)
i = 1

# For recording metrics per fold
aucs = []
auprs = []
f1s = []
sensitivities = []
specificities = []
precisions = []
accuracies = []
tprs = []
mean_fpr = np.linspace(0, 1, 1000)

best_auc = 0
best_clf = None
skf = StratifiedKFold(n_splits=k_fold, random_state=0, shuffle=True)

for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    clf = XGBClassifier(n_estimators=NmedEdge,
                        max_depth=DmedEdge,
                        subsample=SmedEdge,
                        learning_rate=0.1)

    clf.fit(np.array(X_train), np.array(Y_train))

    # Probability and prediction
    y_score = clf.predict_proba(np.array(X_test))[:, 1]
    y_pred = (y_score >= 0.5).astype(int)  # Threshold adjustable

    # AUC / AUPR
    fpr, tpr, thresholds = roc_curve(Y_test, y_score)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aupr = average_precision_score(Y_test, y_score)

    # Confusion matrix count (ensure numpy array)
    Yt = np.array(Y_test).astype(int)
    TP = int(((Yt == 1) & (y_pred == 1)).sum())
    TN = int(((Yt == 0) & (y_pred == 0)).sum())
    FP = int(((Yt == 0) & (y_pred == 1)).sum())
    FN = int(((Yt == 1) & (y_pred == 0)).sum())

    # Metric calculation (with division by zero protection)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

    # Record
    aucs.append(roc_auc)
    auprs.append(aupr)
    f1s.append(f1)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    precisions.append(precision)
    accuracies.append(accuracy)

    # Save current best model (by AUC)
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_clf = clf
        torch.save(clf, './n_best_GBDT.pt')

    print('Fold %d --> AUC=%0.4f, AUPR=%0.4f, F1=%0.4f, Sens=%.4f, Spec=%.4f, Prec=%.4f, Acc=%.4f'
          % (i, roc_auc, aupr, f1, sensitivity, specificity, precision, accuracy))
    i += 1

# Summary and print mean and standard deviation
def mean_std(arr):
    return np.mean(arr), np.std(arr)

mean_auc, std_auc = mean_std(aucs)
mean_aupr, std_aupr = mean_std(auprs)
mean_f1, std_f1 = mean_std(f1s)
mean_sens, std_sens = mean_std(sensitivities)
mean_spec, std_spec = mean_std(specificities)
mean_prec, std_prec = mean_std(precisions)
mean_acc, std_acc = mean_std(accuracies)

print("\n===== CV Summary =====")
print(f"AUC    : {mean_auc:.4f} ± {std_auc:.4f}")
print(f"AUPR   : {mean_aupr:.4f} ± {std_aupr:.4f}")
print(f"F1     : {mean_f1:.4f} ± {std_f1:.4f}")
print(f"Sens   : {mean_sens:.4f} ± {std_sens:.4f}")
print(f"Spec   : {mean_spec:.4f} ± {std_spec:.4f}")
print(f"Prec   : {mean_prec:.4f} ± {std_prec:.4f}")
print(f"Acc    : {mean_acc:.4f} ± {std_acc:.4f}")
print("======================\n")