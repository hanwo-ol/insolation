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
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
torch.cuda.set_per_process_memory_fraction(0.99, device=0)  # GPU 0의 메모리 80%만 사용
torch.cuda.set_per_process_memory_fraction(0.99, device=1)  # GPU 1의 메모리 80%만 사용
# Import modules
from modules_1sobel_DA.unet_ch6 import UNet
# loss function 분리
from modules_1sobel_DA.loss_function_g0 import MS_SSIM_L1_LOSS as MS_SSIM_L1_LOSS_g0
from modules_1sobel_DA.loss_function_g1 import MS_SSIM_L1_LOSS as MS_SSIM_L1_LOSS_g1
from modules_1sobel_DA.sam import SAM
from modules_1sobel_DA.dataset_1sobel import TimeSeriesDataset


# Hyperparameters settings
num_epochs = 10
batch_size = 4
learning_rate = 0.0012
lp_values = [0]  
alpha = 0.025  
date_today = "s1_al_" + f"{alpha}"

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def calculate_metrics(output_image, target_image):
    """Calculates evaluation metrics (MSE, MAE, PSNR, SSIM)."""
    output_image = output_image.squeeze()
    target_image = target_image.squeeze()
    output_image_np = output_image.cpu().detach().numpy()
    target_image_np = target_image.cpu().detach().numpy()
    mse = F.mse_loss(output_image, target_image).item()
    mae = F.l1_loss(output_image, target_image).item()
    output_image_255 = ((output_image_np + 1) / 2) * 255
    target_image_255 = ((target_image_np + 1) / 2) * 255
    output_image_255 = np.clip(output_image_255, 0, 255).astype(np.uint8)
    target_image_255 = np.clip(target_image_255, 0, 255).astype(np.uint8)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    ssim_score = ssim(output_image_255, target_image_255, data_range=255, multichannel=False)
    return {"MSE": mse, "MAE": mae, "PSNR": psnr, "SSIM": ssim_score}


for lp in lp_values:  # In this example, we're only using lp=0

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

    # --- Model, Loss, Optimizer ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet(n_channels=6).to(device)  

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1])  # Wrap with DataParallel *after* moving to device

    # model.to(device)  # <--- 이 부분을 제거합니다.

    # Loss functions (one for each GPU, if available)
    criterion_g0 = MS_SSIM_L1_LOSS_g0(data_range=2.0, alpha=alpha, cuda_dev=0).to("cuda:0") #.to("cuda:0")추가
    criterion_g1 = MS_SSIM_L1_LOSS_g1(data_range=2.0, alpha=alpha, cuda_dev=1).to("cuda:1") #.to("cuda:1")추가


    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=learning_rate, momentum=0.9)

    # --- Datasets and Dataloaders ---
    train_dataset = TimeSeriesDataset("fast_train", transform=transform, split='train', lp=lp)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4) # num_workers 추가
    test_dataset = TimeSeriesDataset("fast_test", transform=transform, split='test', lp=lp)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4) # num_workers 추가

    # --- Scheduler (ReduceLROnPlateau) ---
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False)

    # --- Initialize lists to store metrics for each epoch ---
    train_losses = []
    train_mses = []
    train_maes = []
    train_psnrs = []
    train_ssims = []


    # --- Training Loop ---
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        model.train()
        epoch_loss = 0
        epoch_mses = []
        epoch_maes = []
        epoch_psnrs = []
        epoch_ssims = []


        train_dataloader_tqdm = tqdm(train_dataloader, unit="batch", leave=False)
        for inputs, targets, filename in train_dataloader_tqdm:
            # --- Move data to the appropriate devices ---
            inputs = inputs.to("cuda:0")  # Always move inputs to cuda:0
            targets = targets.to("cuda:0") # Always move targets to cuda:0


            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)  # Model will distribute across GPUs

                # --- Calculate loss on the appropriate GPU ---
                # Determine which loss function to use based on the *output* device.
                if outputs.device == torch.device("cuda:0"):
                    loss = criterion_g0(outputs, targets)
                elif outputs.device == torch.device("cuda:1"):
                    loss = criterion_g1(outputs, targets)
                else: #CPU
                    loss = criterion_g0(outputs, targets)


                loss.backward()
                return loss, outputs

            loss, outputs = closure()
            optimizer.step(closure)
            epoch_loss += loss.item()

            # --- Calculate metrics (move outputs to CPU for metric calculation) ---
            with torch.no_grad():
                for i in range(outputs.shape[0]):
                    metrics = calculate_metrics(outputs[i], targets)
                    epoch_mses.append(metrics["MSE"])
                    epoch_maes.append(metrics["MAE"])
                    epoch_psnrs.append(metrics["PSNR"])
                    epoch_ssims.append(metrics["SSIM"])

            train_dataloader_tqdm.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_dataloader)

        avg_mse = np.mean(epoch_mses)
        avg_mae = np.mean(epoch_maes)
        avg_psnr = np.mean(epoch_psnrs)
        avg_ssim = np.mean(epoch_ssims)

        train_losses.append(avg_epoch_loss)
        train_mses.append(avg_mse)
        train_maes.append(avg_mae)
        train_psnrs.append(avg_psnr)
        train_ssims.append(avg_ssim)

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Learning Rate: {current_lr:.6f}, MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

        scheduler.step(avg_epoch_loss)

    # --- Save Model ---
    # Save the *entire* model (including DataParallel wrapper)
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), f'./Satellite/{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}.pt')
    else:
        torch.save(model.state_dict(), f'./Satellite/{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}.pt')



    # --- Save training errors to a file ---
    with open(f"train_error_{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}.txt", "w") as f:
        f.write("Epoch\tLoss\tMSE\tMAE\tPSNR\tSSIM\n")
        for i in range(num_epochs):
            f.write(f"{i+1}\t{train_losses[i]:.4f}\t{train_mses[i]:.4f}\t{train_maes[i]:.4f}\t{train_psnrs[i]:.4f}\t{train_ssims[i]:.4f}\n")
    print("train error is saved")

    # --- Inference Loop ---
    model.eval()
    with torch.no_grad():
        for j in range(len(test_dataset)):
            sample_input, sample_target, filename = test_dataset[j]
            sample_input = sample_input.unsqueeze(0).to("cuda:0")  # Move input to cuda:0
            sample_target = sample_target.to("cuda:0")
            sample_output = model(sample_input)
            if isinstance(sample_output, tuple): #DP
                sample_output = sample_output[0]

            if isinstance(filename, list) or isinstance(filename, tuple):
                filename = filename[0]
            date_str = os.path.splitext(filename)[0]
            output_filename = f"{date_str}_{lp}"

            output_image = sample_output.squeeze().cpu()
            save_image(output_image, f'./pred_{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}/{output_filename}_predicted.png', normalize=False)

            target_image = sample_target.squeeze().cpu()
            save_image(target_image, f'./target_{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}/{output_filename}_target.png', normalize=False)

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