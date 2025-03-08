# write a function to get the ranking from biggest to smallest value in 2d tensor (its x and y)
# printed as:
# x:0. y:1, values: 1.7
# x:0, y:2, values: 1.5
# import torch
def get_ranking(tensor):
    # Flatten tensor to 1D and get sorted indices
    flat_indices = torch.argsort(tensor.flatten(), descending=True)
    
    # Convert flat indices back to 2D coordinates
    height = tensor.shape[0]
    width = tensor.shape[1]
    x_coords = flat_indices // width
    y_coords = flat_indices % width
    
    # Print coordinates and values in sorted order
    for x, y in zip(x_coords, y_coords):
        value = tensor[x, y].item()
        print(f"x:{x.item()}, y:{y.item()}, values: {value:.4f}")
    
# ts = torch.tensor([[-0.0000,  0.3008,  0.2002, -0.0000, -0.0000, -0.0000,  0.3008, -0.0000],
#         [ 0.3008,  0.2002,  0.2002,  0.2002,  0.2002,  0.1001,  0.4004,  0.1001],
#         [ 0.1001,  0.2002,  0.4004,  0.1001,  0.2002,  0.6992,  0.2002,  0.1001],
#         [ 0.3008, -0.0000,  0.1001,  0.2002, -0.2002,  0.2002,  0.2002,  0.2002],
#         [ 0.1001,  0.2002, -0.0000,  0.4004,  0.2002,  0.4004, -0.6016,  0.2002],
#         [-2.4062,  0.1001,  0.1001,  0.3008,  0.8008,  0.1001,  0.1001,  0.2002],
#         [ 0.2002, -0.1001, -0.0000,  0.6016, -0.0000, -0.0000,  0.1001, -0.1001],
#         [-0.0000,  0.2002,  0.6016,  0.6992,  0.3008, -0.1001,  0.2002,  0.6992],
#         [ 0.2002,  0.1001,  0.4004,  0.8984,  0.2002,  0.4004,  0.2002,  0.2002],
#         [ 0.1001,  0.4004,  0.1001,  0.4004,  0.1001,  0.3008,  0.3008,  0.2002],
#         [ 0.1001,  0.1001, -0.1001,  0.4004,  0.4004,  0.2002,  0.4004,  0.2002],
#         [ 0.4004,  0.2002,  0.6992, -0.2002, -0.2002,  0.2002,  0.2002,  0.3008],
#         [-0.2002,  0.4004,  0.6992, -0.3008, -0.1001,  0.2002,  0.5000,  1.0000],
#         [ 0.1001,  0.4004, -0.5000,  0.2002,  0.2002, -0.0000,  0.6016,  0.5000],
#         [ 0.1001,  0.5000, -0.1001,  0.6016, -0.4004, -0.6016,  0.6016,  0.1001],
#         [ 0.1001,  0.2002, -0.1001, -0.6016,  0.3008, -0.0000,  0.6992, -0.2002],
#         [ 0.3008,  0.2002,  0.6016,  0.2002, -0.0000,  0.2002, -0.1001, -0.1001],
#         [ 0.3008,  0.4004,  0.2002,  1.2031,  0.2002, -0.0000,  0.2002, -0.2002],
#         [ 0.1001, -0.0000, -0.1001, -0.2002, -0.0000,  0.4004,  0.4004, -0.2002],
#         [ 0.2002,  0.6016, -0.1001,  0.1001, -0.1001,  0.2002,  0.2002,  0.6016],
#         [-0.0000,  0.4004, -1.8984,  1.8984,  0.1001,  0.3008,  0.6016,  0.6016],
#         [-0.0000,  0.3008,  0.1001, -0.0000,  0.1001,  0.2002,  0.3008,  0.2002],
#         [ 0.3008,  0.2002,  0.3008,  0.2002,  1.5000, -1.7031,  0.1001, -0.0000],
#         [ 0.6992,  0.1001,  0.1001, -0.5000,  1.3984, -0.2002,  0.5000,  0.1001],
#         [ 0.2002,  0.2002, -0.4004,  1.2031, -0.1001, -0.1001,  0.1001, -0.0000],
#         [ 0.1001,  0.1001,  3.0000, -0.3008,  0.6016, -0.3008, -0.2002, -0.8008]],
#        device='cuda:0')
# get_ranking(ts)

# load this /mnt/weka/hw_workspace/qy_workspace/lightning/outputs/patching_results_attn_head.pt
import torch
import matplotlib.pyplot as plt
patching_results = torch.load('/mnt/weka/hw_workspace/qy_workspace/lightning/outputs/patching_results_attn_head.pt')
print(patching_results.shape)
# draw the patching_results (3d tensor [26, 25, 8] ) (complete)
plt.figure(figsize=(40, 80))
num_rows = 7
num_cols = 4
for i in range(patching_results.shape[0]):
    plt.subplot(num_rows, num_cols, i+1)
    plt.imshow(patching_results[i].cpu().numpy(), cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"attn_head_out Activation Patching By Pos {i}")
    plt.xlabel("Position")
    plt.ylabel("Head")
plt.tight_layout()
plt.savefig('patching_results_attn_head.png', bbox_inches='tight')
plt.close()

# print the ranking average
average_value = torch.mean(patching_results, dim=(1))
get_ranking(average_value)