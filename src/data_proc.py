import random
import numpy as np
import pandas as pd
import torch
from Bio.Align import substitution_matrices
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import pickle as pkl
from Bio import pairwise2
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
            'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
            'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()]
    )

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")


    atom_features_list = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        atom_features_list.append(feature / sum(feature))  # 归一化

    x = torch.tensor(atom_features_list, dtype=torch.float)


    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edges.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])  # 无向图

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


    data = Data(x=x, edge_index=edge_index)
    return data

class GCNLayer(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_feats * 2, out_feats)

    def forward(self, x, edge_index):

        agg = torch.zeros_like(x)
        if edge_index.numel() == 0:
            print("ZERO")
            return torch.zeros([20, 128]).to(device)
        agg[edge_index[0]] = x[edge_index[1]]
        out = self.linear(torch.cat((x, agg), dim=-1))
        return out

def extract_features(smiles):
    graph_data = smiles_to_graph(smiles)
    gcn_layer = GCNLayer(in_feats=78, out_feats=128).to(device)  # 输入维度改为78
    node_embeddings = gcn_layer(graph_data.x.to(device), graph_data.edge_index.to(device))
    # 使用平均池化作为读出函数
    molecule_embedding = node_embeddings.mean(dim=0)
    return molecule_embedding


aa_index = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    'X': 20, 'U': 21
}


def aa_sequence_to_graph(sequence):
    # 提取氨基酸特征
    node_features = torch.tensor([aa_index[aa] for aa in sequence], dtype=torch.float).unsqueeze(1)

    # 构建邻接矩阵
    edges = []
    for i in range(len(sequence) - 1):
        edges.append((i, i + 1))
        edges.append((i + 1, i))  # 无向图

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


    x = node_features
    data = Data(x=x, edge_index=edge_index)
    return data

class ProteinGCN(torch.nn.Module):
    def __init__(self):
        super(ProteinGCN, self).__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)  # 20为氨基酸种类数

    def forward(self, data):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def extract_aa_features(sequence):
    graph_data = aa_sequence_to_graph(sequence)
    model = ProteinGCN().to(device)
    node_embeddings = model(graph_data.to(device))

    protein_embedding = node_embeddings.mean(dim=0)
    return protein_embedding


def calculate_drug_similarity(smiles_list):
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=4096) for mol in molecules]

    maccs_fps = []
    for mol in molecules:
        if mol is None:
            maccs_fps.append(None)
            continue
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        maccs_fps.append(fp)

    def fused_tanimoto(fp1_m, fp1_maccs, fp2_m, fp2_maccs):
        sim_morgan = AllChem.DataStructs.TanimotoSimilarity(fp1_m, fp2_m)
        if fp1_maccs is None or fp2_maccs is None:
            sim_maccs = 0.0
        else:
            sim_maccs = AllChem.DataStructs.TanimotoSimilarity(fp1_maccs, fp2_maccs)
        return 0.7 * sim_morgan + 0.3 * sim_maccs

    similarity_matrix = []
    for i in range(len(molecules)):
        row = []
        for j in range(len(molecules)):
            if i == j:
                row.append(1.0)
            else:
                sim = fused_tanimoto(morgan_fps[i], maccs_fps[i], morgan_fps[j], maccs_fps[j])
                row.append(sim)
        similarity_matrix.append(row)
    return np.array(similarity_matrix)

def calculate_protein_similarity(sequence_list):
    # 加载BLOSUM62矩阵
    blosum62 = substitution_matrices.load("BLOSUM62")

    similarity_matrix = []
    for i in range(len(sequence_list)):
        row = []
        for j in range(len(sequence_list)):
            if i == j:
                row.append(1.0)
            else:
                # 使用BLOSUM62的全局对齐 (gap_open=-10, gap_extend=-0.5 为标准)
                alignments = pairwise2.align.globalds(
                    sequence_list[i], sequence_list[j],
                    blosum62, -10, -0.5
                )
                if alignments:
                    score = alignments[0][2]
                    # 标准化: 分数 / (min(len1, len2) * max_blosum)
                    len_min = min(len(sequence_list[i]), len(sequence_list[j]))
                    max_blosum = 11  # BLOSUM62最大值 (identity for W/W)
                    normalized_sim = score / (len_min * max_blosum)
                    row.append(max(0, min(1, normalized_sim)))  # 夹到[0,1]
                else:
                    row.append(0.0)
        similarity_matrix.append(row)
        print(f"Processed sequence {i + 1}/{len(sequence_list)}")  # 进度提示
    return np.array(similarity_matrix)


dataset = "DrugBank"
#DrugBank
device = "cuda:0"
full = pd.read_csv(f"../data/{dataset}/full.csv")
smiles_list = []
protein_sequences = []

for i, row in full.iterrows():
    if row['SMILES'] not in smiles_list:
        smiles_list.append(row['SMILES'])
    if row['Protein'] not in protein_sequences:
        protein_sequences.append(row['Protein'])

prot2index = {}
drug2index = {}
for i, d in enumerate(smiles_list):
    if d not in drug2index:
        drug2index[d] = len(drug2index)

for i, p in enumerate(protein_sequences):
    if p not in prot2index:
        prot2index[p] = len(drug2index) + len(prot2index)

pkl.dump(drug2index, open(f"../data/{dataset}/drug2index.pkl", "wb"))
pkl.dump(prot2index, open(f"../data/{dataset}/prot2index.pkl", "wb"))

positive_pair_d = []
positive_pair_p = []
positive_pair = []
for i, row in full.iterrows():
    if int(row['Y']) == 1:
        positive_pair_d.append(drug2index[row['SMILES']])
        positive_pair_p.append(prot2index[row['Protein']])
        positive_pair.append((drug2index[row['SMILES']], prot2index[row['Protein']]))

negative_pair_d = []
negative_pair_p = []
ind_d = list(drug2index.values())
ind_p = list(prot2index.values())
while len(negative_pair_d) < len(positive_pair):
    i_d = random.choice(ind_d)
    i_p = random.choice(ind_p)
    if (i_d, i_p) not in positive_pair:
        negative_pair_d.append(i_d)
        negative_pair_p.append(i_p)

Allnode = [i for i in range(len(drug2index) + len(prot2index))]
Allnode_df = pd.DataFrame({'node': Allnode})

print(f'pos pair: {len(positive_pair)}, neg pair: {len(negative_pair_d)}')

DrPrNum_Drpr = pd.DataFrame({'pair_d': positive_pair_d, 'pair_p': positive_pair_p})
DrPrNum_Drpr.to_csv(f"../data/{dataset}/DrPrNum_DrPr.csv", index=False, header=False)
AllNegative_DrPr = pd.DataFrame({'pair_d': negative_pair_d, 'pair_p': negative_pair_p})
AllNegative_DrPr.to_csv(f"../data/{dataset}/AllNegative_DrPr.csv", index=False, header=False)
Allnode_df.to_csv(f"../data/{dataset}/Allnode_DrPr.csv", index=False, header=False)

num = {"drug_num": len(drug2index), "prot_num": len(prot2index)}
pkl.dump(num, open(f"../data/{dataset}/num.pkl", "wb"))

drug_embeddings = []
for i, d in enumerate(smiles_list):
    drug_embeddings.append(extract_features(d).cpu().detach().numpy())

protein_embeddings = []
for i, p in enumerate(protein_sequences):
    protein_embeddings.append(extract_aa_features(p).cpu().detach().numpy())

prot_similarity_matrix = cosine_similarity(protein_embeddings)
drug_similarity_matrix = calculate_drug_similarity(smiles_list)

pkl.dump(drug_similarity_matrix, open(f"../data/{dataset}/drug_similarity_matrix.pkl", "wb"))
pkl.dump(prot_similarity_matrix, open(f"../data/{dataset}/prot_similarity_matrix.pkl", "wb"))
AllNodeAttribute_DrPr = pd.concat([pd.DataFrame(drug_embeddings), pd.DataFrame(protein_embeddings)], axis=0)
AllNodeAttribute_DrPr.to_csv(f"../data/{dataset}/AllNodeAttribute_DrPr.csv", index=False, header=False)

prot_edge1 = []
prot_edge2 = []
drug_edge1 = []
drug_edge2 = []

for i, row in pd.read_csv(f'../data/{dataset}/DrPrNum_DrPr.csv', header=None).iterrows():
    prot_edge1.append(int(row[0]))
    prot_edge2.append(int(row[1]))
    drug_edge1.append(int(row[0]))
    drug_edge2.append(int(row[1]))

print(f"{dataset} positive edges: {len(prot_edge1)}")

pp = 0
p_e1 = []
p_e2 = []
for i in range(len(prot_similarity_matrix)):
    for j in range(i + 1, len(prot_similarity_matrix)):
        if prot_similarity_matrix[i][j] > 0.985:
            p_e1.append(i)
            p_e2.append(j)
            pp += 1

random_array = np.random.randint(0, len(p_e1) - 1, size=min(500, len(p_e1) - 1))
for ind in random_array:
    prot_edge1.append(p_e1[ind])
    prot_edge2.append(p_e2[ind])

print("Prot-Prot edges: ", pp)
df = pd.DataFrame({'0': prot_edge1, '1': prot_edge2})
df.to_csv(f'../data/{dataset}/prot_edge.csv', index=False, header=False)
print("P edges: ", len(prot_edge1))

dd = 0
d_e1 = []
d_e2 = []
for i in range(len(drug_similarity_matrix)):
    for j in range(i + 1, len(drug_similarity_matrix)):
        if drug_similarity_matrix[i][j] > 0.988:
            d_e1.append(i)
            d_e2.append(j)
            dd += 1

random_array = np.random.randint(0, len(d_e1) - 1, size=min(500, len(d_e1) - 1))
for ind in random_array:
    prot_edge1.append(d_e1[ind])
    prot_edge2.append(d_e2[ind])
    drug_edge1.append(d_e1[ind])
    drug_edge2.append(d_e2[ind])

print("Drug-Drug edges: ", dd)
print("Total edges: ", len(prot_edge1))
print("D edges: ", len(drug_edge1))

df = pd.DataFrame({'0': drug_edge1, '1': drug_edge2})
df.to_csv(f'../data/{dataset}/drug_edge.csv', index=False, header=False)

df = pd.DataFrame({'0': prot_edge1, '1': prot_edge2})
df.to_csv(f'../data/{dataset}/drug_prot_edge.csv', index=False, header=False)

print("Done")

