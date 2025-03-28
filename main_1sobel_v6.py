# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F  # Import torch.nn.functional

torch.autograd.set_detect_anomaly(True)

# Import modules
from modules_1sobel_v3.unet_ch6 import UNet
from modules_1sobel_v3.loss_function_g1 import MS_SSIM_L1_LOSS
from modules_1sobel_v3.sam import SAM
from modules_1sobel_v3.dataset_1sobel import TimeSeriesDataset

torch.cuda.set_per_process_memory_fraction(0.1, device=None)

# Hyperparameters settings
num_epochs = 30
batch_size = 4  #
learning_rate = 0.0012
lp_values = [0]
alpha = 0.0025
date_today = "s1_al " + f"{alpha}"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])



def calculate_metrics(output_image, target_image):
    """
    Calculates evaluation metrics between output and target images.  (evaluate.py version)
    """
    output_image = output_image.squeeze()  # Remove batch and channel dimensions
    target_image = target_image.squeeze()

    # Ensure tensors are on CPU and converted to NumPy arrays
    output_image_np = output_image.cpu().detach().numpy()
    target_image_np = target_image.cpu().detach().numpy()

    # 1. Mean Squared Error (MSE)
    mse = F.mse_loss(output_image, target_image).item()

    # 2. Mean Absolute Error (MAE)
    mae = F.l1_loss(output_image, target_image).item()

    # 3. Peak Signal-to-Noise Ratio (PSNR)
    #   - First, convert images to 0-255 range (assuming they were normalized to [-1, 1])
    output_image_255 = ((output_image_np + 1) / 2) * 255
    target_image_255 = ((target_image_np + 1) / 2) * 255

    #   - Clip values to be within 0-255
    output_image_255 = np.clip(output_image_255, 0, 255).astype(np.uint8)
    target_image_255 = np.clip(target_image_255, 0, 255).astype(np.uint8)

    #   - Calculate PSNR
    if mse == 0:
      psnr = float('inf')
    else:
      psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    # 4. Structural Similarity Index (SSIM) - using scikit-image
    ssim_score = ssim(output_image_255, target_image_255, data_range=255, multichannel=False)

    return {"MSE": mse, "MAE": mae, "PSNR": psnr, "SSIM": ssim_score}




for lp in lp_values:

    folders = [f'./pred_{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}',
               f'./target_{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}',
               f'./result_{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}',
               "./Satellite"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Folder created: {folder}")
        else:
            print(f"Folder already exists: {folder}")

    # Model, Loss, Optimizer
    model = UNet(n_channels=6).to(device)
    criterion = MS_SSIM_L1_LOSS(data_range=2.0, 
                                alpha=alpha,
                                cuda_dev=device.index if device.type == 'cuda' else 0)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=learning_rate, momentum=0.9)

    # Datasets and Dataloaders
    train_dataset = TimeSeriesDataset("fast_train", transform=transform, split='train', lp=lp)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TimeSeriesDataset("fast_test", transform=transform, split='test', lp=lp)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Scheduler (ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False)


    # Initialize lists to store metrics for each epoch
    train_losses = []
    train_mses = []
    train_maes = []
    train_psnrs = []
    train_ssims = []



    # Training Loop
    for epoch in range(num_epochs):
        print(f"statistics server, gpu:{device}, one sobel image is used")

        model.train()
        epoch_loss = 0
        epoch_mses = []  # Use lists, not numpy arrays initially
        epoch_maes = []
        epoch_psnrs = []
        epoch_ssims = []


        train_dataloader_tqdm = tqdm(train_dataloader, unit="batch", leave=False)
        for inputs, targets, filename in train_dataloader_tqdm:
            inputs, targets = inputs.to(device), targets.to(device)


            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                return loss, outputs  # Return outputs as well

            loss, outputs = closure()
            optimizer.step(closure)
            epoch_loss += loss.item()

            # Calculate metrics for the batch
            with torch.no_grad():
                for i in range(outputs.shape[0]):
                    metrics = calculate_metrics(outputs[i], targets[i])  # Use the new function
                    epoch_mses.append(metrics["MSE"])  # Append individual values
                    epoch_maes.append(metrics["MAE"])
                    epoch_psnrs.append(metrics["PSNR"])
                    epoch_ssims.append(metrics["SSIM"])


            train_dataloader_tqdm.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_dataloader)

        # Calculate average metrics for the epoch and convert to numpy arrays
        avg_mse = np.mean(epoch_mses)
        avg_mae = np.mean(epoch_maes)
        avg_psnr = np.mean(epoch_psnrs)
        avg_ssim = np.mean(epoch_ssims)


        # Append to lists
        train_losses.append(avg_epoch_loss)
        train_mses.append(avg_mse)
        train_maes.append(avg_mae)
        train_psnrs.append(avg_psnr)
        train_ssims.append(avg_ssim)


        # 현재 학습률 출력
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Learning Rate: {current_lr:.6f}, MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

        # Scheduler step
        scheduler.step(avg_epoch_loss)



    # Save Model
    torch.save(model.state_dict(), f'./Satellite/{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}.pt')
    #model = torch.jit.load(f'./Satellite/{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}.pt')


    # Save training errors to a file
    with open(f"train_error_{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}.txt", "w") as f:
        f.write("Epoch\tLoss\tMSE\tMAE\tPSNR\tSSIM\n")
        for i in range(num_epochs):
            f.write(f"{i+1}\t{train_losses[i]:.4f}\t{train_mses[i]:.4f}\t{train_maes[i]:.4f}\t{train_psnrs[i]:.4f}\t{train_ssims[i]:.4f}\n")
    print("train error is saved")


    # Inference Loop
    model.eval()
    with torch.no_grad():
        for j in range(len(test_dataset)):
            sample_input, sample_target, filename = test_dataset[j]
            sample_input = sample_input.unsqueeze(0).to(device)
            sample_target = sample_target.to(device)
            sample_output = model(sample_input)


            # 타임스탬프 추출 및 lp 포함한 파일명 생성
            if isinstance(filename, list) or isinstance(filename, tuple):
                filename = filename[0]
            date_str = os.path.splitext(filename)[0]
            output_filename = f"{date_str}_{lp}"


            # 출력 이미지 저장
            output_image = sample_output.squeeze().cpu()
            save_image(output_image, f'./pred_{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}/{output_filename}_predicted.png', normalize=False)

            # 타겟 이미지 저장
            target_image = sample_target.squeeze().cpu()
            save_image(target_image, f'./target_{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}/{output_filename}_target.png', normalize=False)


            # 그림 저장
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(output_image.squeeze().cpu(), cmap='gray')
            plt.title(f"Predicted Image (lp={lp})")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(target_image.squeeze().cpu(), cmap='gray')
            plt.title("Target Image")
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(f'./result_{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}/{output_filename}.png')
            plt.close()

print("Training and inference complete.")