import numpy as np
import torch
import torch.nn.functional as F_nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


def compute_embedding_complexity(model, forget_loader, retain_loader, device='cuda', layer_name=None, chunk_size=512):
    """Diagnostic for how "hard" the forget set is to separate from retain
    data: inter_class_proximity (how close forget samples are to their
    nearest retain neighbor) minus intra_class_cohesion (how tightly forget
    samples cluster around their own centroid). Higher complexity means the
    forget class overlaps more with retain classes in feature space."""
    model.eval()

    forget_embeddings = []
    retain_embeddings = []

    def _extract(x):
        if layer_name is None:
            if hasattr(model, 'module'):
                return model.module.get_embedding(x)
            return model.get_embedding(x)
        # Convert "9" → int for AllCNN; keep string for ResNet
        layer = int(layer_name) if layer_name.isdigit() else layer_name
        if hasattr(model, 'module'):
            _, feat_dict = model.module.forward_with_features(x, capture_layers=[layer])
        else:
            _, feat_dict = model.forward_with_features(x, capture_layers=[layer])
        feat = feat_dict[layer]
        if feat.dim() > 2:
            feat = feat.mean(dim=(2, 3))  # Global average pooling for conv layers
        return feat

    with torch.no_grad():
        for x, _ in forget_loader:
            forget_embeddings.append(_extract(x.to(device)))
        for x, _ in retain_loader:
            retain_embeddings.append(_extract(x.to(device)))

    forget_embeddings = torch.cat(forget_embeddings, dim=0)  # [N_f, d]
    retain_embeddings = torch.cat(retain_embeddings, dim=0)  # [N_r, d]

    forget_normed = F_nn.normalize(forget_embeddings, dim=1, eps=1e-8)  # [N_f, d]
    retain_normed = F_nn.normalize(retain_embeddings, dim=1, eps=1e-8)  # [N_r, d]

    # Forget centroid (normalised for cosine similarity)
    mu_f = F_nn.normalize(forget_embeddings.mean(dim=0, keepdim=True), dim=1, eps=1e-8)  # [1, d]

    # Inter-class proximity: E[max_{x_r} cos_sim(x_f, x_r)]
    # Chunked over retain to avoid OOM on large retain sets
    max_sims = torch.full((len(forget_normed),), -float('inf'), device=device)
    for start in range(0, len(retain_normed), chunk_size):
        chunk = retain_normed[start:start + chunk_size]       # [chunk, d]
        sims = torch.mm(forget_normed, chunk.t())              # [N_f, chunk]
        chunk_max, _ = sims.max(dim=1)                         # [N_f]
        max_sims = torch.maximum(max_sims, chunk_max)

    inter_class_proximity = max_sims.mean().item()

    # Intra-class cohesion: E[cos_sim(x_f, mu_f)]
    intra_sims = torch.mm(forget_normed, mu_f.t()).squeeze(1)  # [N_f]
    intra_class_cohesion = intra_sims.mean().item()

    complexity = inter_class_proximity - intra_class_cohesion

    return {
        'complexity': complexity,
        'inter_class_proximity': inter_class_proximity,
        'intra_class_cohesion': intra_class_cohesion,
        'forget_centroid_norm_raw': forget_embeddings.mean(dim=0).norm().item(),
        'n_forget': len(forget_embeddings),
        'n_retain': len(retain_embeddings),
        'max_sims_per_sample': max_sims.cpu().numpy(),
    }


def get_embeddings_predictions_and_forget_indications(model, forget_loader, remain_loader, device):
    model.eval()
    embeddings = []
    predictions = []
    is_forget_sample = []
    true_labels = []

    with torch.no_grad():
        for data, target in tqdm(forget_loader, desc="Processing Forget Samples"):
            data = data.to(device)
            if isinstance(model, torch.nn.DataParallel):
                emb = model.module.get_embedding(data)
            else:
                emb = model.get_embedding(data)            
            output = model(data)
            embeddings.append(emb.cpu().numpy())
            predictions.append(output.argmax(dim=1).cpu().numpy())
            true_labels.append(target.cpu().numpy())
            is_forget_sample.append(np.ones(len(data), dtype=bool))

        for data, target in tqdm(remain_loader, desc="Processing Remain Samples"):
            data = data.to(device)
            if isinstance(model, torch.nn.DataParallel):
                emb = model.module.get_embedding(data)
            else:
                emb = model.get_embedding(data)
            output = model(data)
            embeddings.append(emb.cpu().numpy())
            predictions.append(output.argmax(dim=1).cpu().numpy())
            true_labels.append(target.cpu().numpy())
            is_forget_sample.append(np.zeros(len(data), dtype=bool))

    embeddings = np.concatenate(embeddings, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    is_forget_sample = np.concatenate(is_forget_sample, axis=0)

    return embeddings, predictions, is_forget_sample, true_labels



def plot_tsne(embeddings, predictions, is_forget_sample, true_labels, title, ax, add_legend=False):
    tsne = TSNE(n_components=2, random_state=0)
    pts = tsne.fit_transform(embeddings)

    mis = predictions != true_labels
    rem = ~is_forget_sample
    fog =  is_forget_sample

    class_colors = {0:'red', 1:'green', 2:'blue'}  
    true_color_map = np.array([class_colors[t] for t in true_labels])
    pred_color_map = np.array([class_colors[p] for p in predictions])
    edge_colors = np.where(mis, pred_color_map, 'none')
    line_widths = np.where(mis, 3, 0)

    def draw_boundary():
        clf = KNeighborsClassifier(n_neighbors=5).fit(pts, predictions)
        x_min, x_max = pts[:,0].min(), pts[:,0].max()
        y_min, y_max = pts[:,1].min(), pts[:,1].max()
        padding = 2.0
        xx, yy = np.meshgrid(np.linspace(x_min-padding, x_max+padding, 400),
                             np.linspace(y_min-padding, y_max+padding, 400))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        cmap = ListedColormap([class_colors[c] for c in sorted(class_colors)])
        ax.contourf(xx, yy, Z, alpha=0.25, cmap=cmap, 
                    levels=np.arange(len(class_colors)+1)-0.5)

        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

    draw_boundary()

    ax.scatter(
        pts[rem,0], pts[rem,1],
        facecolors=true_color_map[rem],
        edgecolors=edge_colors[rem],
        linewidths=line_widths[rem],
        marker='o', s=200, alpha=0.8
    )

    ax.scatter(
        pts[fog,0], pts[fog,1],
        facecolors=true_color_map[fog],
        edgecolors=edge_colors[fog],
        linewidths=line_widths[fog],
        marker='*', s=500, alpha=0.95
    )

    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

    if add_legend:
        handles = [
            plt.Line2D([0],[0], marker='o', color='w',
                       markerfacecolor=class_colors[c], markersize=10)
            for c in sorted(class_colors)
        ]
        labels = [f'Class {c}' for c in sorted(class_colors)]
        star = plt.Line2D([0],[0], marker='*', color='k',
                          markerfacecolor='white', markersize=12,
                          linestyle='None', label='Forget Set')
        mis_edge = plt.Line2D([0],[0], marker='o', markerfacecolor='white',
                              markeredgecolor='grey', markersize=10,
                              linestyle='None', label='Misclassified Edge')
        ax.legend(handles + [star, mis_edge],
                  labels + ['Forget Set'],
                  loc='best', fontsize=11, frameon=True)



