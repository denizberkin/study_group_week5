import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def visualize_predictions(model, data_loader, device, num_samples=4):
    """Visualize model predictions with original images and ground truth masks."""
    model.eval()

    samples = next(iter(data_loader))
    images, true_masks = samples['image'].to(device), samples['mask'].to(device)

    preds = model(images)
    preds = torch.sigmoid(preds) > 0.5

    images = images.cpu()
    true_masks = true_masks.cpu()
    preds = preds.cpu()

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    
    for i in range(num_samples):
        image = images[i].squeeze().numpy()
        true_mask = true_masks[i].squeeze().numpy()
        pred_mask = preds[i].squeeze().numpy()

        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()