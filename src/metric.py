import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
import os

def feature_vis(
    feature_map: torch.Tensor,
    cache_dir: str,
    name: str
):
    # data = feature_map.unsqueeze(0).cpu().numpy()
    # pca = PCA(n_components=1)
    # data_2d = pca.fit_transform(data)

    # plt.scatter(data_2d[:, 0], data_2d[:, 1])
    # plt.title('PCA projection')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.show()

    # output_file = os.path.join(cache_dir, 'feature_pca_projection.png')
    # plt.savefig(output_file, dpi=300)

    # print(f"PCA results have saved in {output_file}")

    vector = feature_map.cpu().numpy()
    n = len(vector)
    side_length = int(np.ceil(np.sqrt(n)))
    padded_vector = np.pad(vector, (0, side_length**2 - n), mode='constant')
    square_matrix = padded_vector.reshape((side_length, side_length))

    plt.figure(figsize=(8, 8))
    plt.imshow(square_matrix, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Heatmap of Folded Feature Vector')
    plt.axis('off')

    heatmap_file = os.path.join(cache_dir, f'{name}_feature_vector_heatmap.png')
    plt.savefig(heatmap_file, dpi=300)
    plt.show()
    print(f"heatmap has saved in {heatmap_file}")

def test_density():
    pass