import matplotlib.pyplot as plt
'''
def plot_uncertainty(images, logvar):
    uncertainties = torch.exp(logvar).mean(dim=1)
    
    plt.figure(figsize=(15, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title(f"Uncertainty: {uncertainties[i]:.4f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()'''
def plot_uncertainty(image, uncertainty):
    # Reshape uncertainty to patch grid
    heatmap = uncertainty.reshape(grid_size, grid_size)
    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap="viridis")
    plt.savefig("uncertainty.png")