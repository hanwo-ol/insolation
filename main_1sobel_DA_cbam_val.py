# main_1sobel_DA_cbam_val.py
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
import time # For timing epochs

# Enable anomaly detection for debugging NaN issues
# torch.autograd.set_detect_anomaly(True) # Uncomment if needed

# --- Module Imports ---
# Ensure these paths match your project structure
from modules_1sobel_DA.unet_ch6_da import UNet_DA_CBAM_Selective
from modules_1sobel_DA.loss_function_g0 import MS_SSIM_L1_LOSS
from modules_1sobel_DA.sam import SAM
from modules_1sobel_DA.dataset_1sobel import TimeSeriesDataset

# --- Hyperparameters ---
num_epochs = 10 # Number of training epochs
batch_size = 4  # Batch size for training
# NOTE: Validation batch size can be larger if memory allows, as no gradients are stored
validation_batch_size = 8 # Or keep it the same as batch_size
learning_rate = 0.0012 # Initial learning rate
lp_values = [0] # Limit lp values for faster testing/debugging if needed [0, 1, 2, 3]
date_today = "s1_t_da_sel_cbam_val" # Identifier for this run (added _val)
N_CHANNELS_MODEL = 6 # Number of input channels

# --- Dual Attention Configuration ---
DA_REDUCTION_FACTOR = 16
USE_CHECKPOINT = True

# --- Device Setup ---
if torch.cuda.is_available():
    try:
        # Simple selection of cuda:0 if available
        device_id = 0
        device = torch.device(f"cuda:{device_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
    except Exception as e:
        print(f"Error setting GPU, falling back to CPU: {e}")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
print(f"Selected device: {device}")

# --- Data Transformations ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --- Metrics Calculation Function ---
def calculate_metrics(output_image, target_image):
    """Calculates evaluation metrics (MSE, MAE, PSNR, SSIM)."""
    output_image = output_image.squeeze()
    target_image = target_image.squeeze()
    output_image_np = output_image.cpu().detach().numpy()
    target_image_np = target_image.cpu().detach().numpy()

    output_image_cpu = output_image.cpu().to(torch.float32)
    target_image_cpu = target_image.cpu().to(torch.float32)

    mse = F.mse_loss(output_image_cpu, target_image_cpu).item()
    mae = F.l1_loss(output_image_cpu, target_image_cpu).item()

    output_image_255 = np.clip(((output_image_np + 1) / 2) * 255, 0, 255).astype(np.uint8)
    target_image_255 = np.clip(((target_image_np + 1) / 2) * 255, 0, 255).astype(np.uint8)

    mse_255 = np.mean((output_image_255.astype(np.float32) - target_image_255.astype(np.float32)) ** 2)
    if mse_255 == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse_255))

    if output_image_255.ndim > 2: output_image_255 = output_image_255.squeeze()
    if target_image_255.ndim > 2: target_image_255 = target_image_255.squeeze()
    min_dim = min(output_image_255.shape[-2:]) # Use last two dimensions for H, W
    win_size = 7
    if min_dim < win_size:
        win_size = max(3, min_dim if min_dim % 2 != 0 else min_dim - 1)

    try:
        # Specify channel_axis=None for grayscale if needed by newer skimage versions
        ssim_score = ssim(output_image_255, target_image_255, data_range=255, win_size=win_size) # Removed channel_axis
    except ValueError as e:
         print(f"SSIM calculation error: {e}. Setting SSIM to 0.")
         ssim_score = 0.0
    except IndexError as e:
         print(f"SSIM calculation error (likely due to win_size > image size): {e}. Setting SSIM to 0.")
         ssim_score = 0.0


    return {"MSE": mse, "MAE": mae, "PSNR": psnr, "SSIM": ssim_score}


# --- Main Loop ---
for lp in lp_values:
    print(f"\n--- Starting Training and Inference for lp = {lp} ---")

    # --- Define Output Folders ---
    base_filename = f"{date_today}_lp{lp}_e{num_epochs}_b{batch_size}_lr{learning_rate}"
    pred_folder = f'./pred_{base_filename}'
    target_folder = f'./target_{base_filename}'
    result_folder = f'./result_{base_filename}'
    model_save_dir = "./Satellite"
    folders = [pred_folder, target_folder, result_folder, model_save_dir]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # --- Model, Loss, Optimizer Setup ---
    model = UNet_DA_CBAM_Selective(
        n_channels=N_CHANNELS_MODEL, n_classes=1,
        da_reduction=DA_REDUCTION_FACTOR, use_checkpoint=USE_CHECKPOINT
    ).to(device)
    criterion = MS_SSIM_L1_LOSS(data_range=2.0, cuda_dev=device.index if device.type == 'cuda' else -1)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=learning_rate, momentum=0.9)

    # --- Datasets and Dataloaders ---
    num_workers = min(os.cpu_count() // 2, 8) if os.cpu_count() else 4
    print(f"Using {num_workers} workers for DataLoaders.")
    try:
        train_dataset = TimeSeriesDataset("fast_train", transform=transform, split='train', lp=lp)
        # Pin memory true if using GPU
        pin_memory_flag = True if device.type == 'cuda' else False
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=True)
        val_dataset = TimeSeriesDataset("fast_test", transform=transform, split='test', lp=lp) # Use 'test' set as validation
        val_dataloader = DataLoader(val_dataset, batch_size=validation_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory_flag) # Use validation_batch_size
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}. Skipping lp={lp}.")
        continue
    except Exception as e:
        print(f"An unexpected error occurred during dataset loading: {e}. Skipping lp={lp}.")
        continue

    # --- Learning Rate Scheduler ---
    # Schedule based on validation loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.base_optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # --- Metrics Storage for all epochs ---
    history = {
        'train_loss': [], 'train_mse': [], 'train_mae': [], 'train_psnr': [], 'train_ssim': [],
        'val_loss': [], 'val_mse': [], 'val_mae': [], 'val_psnr': [], 'val_ssim': []
    }

    # --- Training Loop ---
    print(f"Starting training for {num_epochs} epochs...")
    best_val_loss = float('inf') # For saving the best model based on validation loss

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train() # Set model to training mode
        epoch_train_loss = 0
        epoch_train_mses, epoch_train_maes, epoch_train_psnrs, epoch_train_ssims = [], [], [], []

        train_dataloader_tqdm = tqdm(train_dataloader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs} Training")

        for inputs, targets, _ in train_dataloader_tqdm: # No need for filename in training loop if not used
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # SAM Closure
            def closure():
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if torch.isnan(loss):
                    print("\nWarning: NaN loss detected in closure!")
                loss.backward()
                return loss, outputs

            try:
                # SAM Steps
                loss, outputs = closure()
                if torch.isnan(loss): raise ValueError("NaN loss detected after first closure.")
                optimizer.first_step(zero_grad=True)
                loss_second, _ = closure()
                if torch.isnan(loss_second): raise ValueError("NaN loss detected after second closure.")
                optimizer.second_step(zero_grad=True)
            except ValueError as e:
                 print(f"\nError during SAM step: {e}. Skipping batch.")
                 continue

            # --- Training Metrics Calculation (for monitoring this batch) ---
            with torch.no_grad():
                for i in range(outputs.shape[0]):
                    try:
                        metrics = calculate_metrics(outputs[i], targets[i])
                        epoch_train_mses.append(metrics["MSE"])
                        epoch_train_maes.append(metrics["MAE"])
                        if not np.isinf(metrics["PSNR"]):
                            epoch_train_psnrs.append(metrics["PSNR"])
                        epoch_train_ssims.append(metrics["SSIM"])
                    except Exception as metric_e:
                        print(f"\nWarning: Error calculating train metrics: {metric_e}")
                        epoch_train_mses.append(np.nan)
                        epoch_train_maes.append(np.nan)
                        epoch_train_ssims.append(np.nan)


            batch_loss = loss.item()
            epoch_train_loss += batch_loss
            train_dataloader_tqdm.set_postfix(loss=f"{batch_loss:.4f}")

        # --- End of Training Epoch Calculation ---
        avg_train_loss = epoch_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_train_mse = np.nanmean(epoch_train_mses) if epoch_train_mses else 0
        avg_train_mae = np.nanmean(epoch_train_maes) if epoch_train_maes else 0
        avg_train_psnr = np.nanmean(epoch_train_psnrs) if epoch_train_psnrs else 0
        avg_train_ssim = np.nanmean(epoch_train_ssims) if epoch_train_ssims else 0

        history['train_loss'].append(avg_train_loss)
        history['train_mse'].append(avg_train_mse)
        history['train_mae'].append(avg_train_mae)
        history['train_psnr'].append(avg_train_psnr)
        history['train_ssim'].append(avg_train_ssim)

        # --- Validation Loop ---
        model.eval() # Set model to evaluation mode
        epoch_val_loss = 0
        epoch_val_mses, epoch_val_maes, epoch_val_psnrs, epoch_val_ssims = [], [], [], []
        val_dataloader_tqdm = tqdm(val_dataloader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs} Validation")

        with torch.no_grad(): # Disable gradient calculations for validation
            for inputs, targets, _ in val_dataloader_tqdm:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets) # Calculate validation loss

                if torch.isnan(loss):
                    print("\nWarning: NaN loss detected during validation!")
                    val_batch_loss = np.nan # Or some indicator value
                else:
                    val_batch_loss = loss.item()
                    epoch_val_loss += val_batch_loss

                # Calculate validation metrics
                for i in range(outputs.shape[0]):
                    try:
                        metrics = calculate_metrics(outputs[i], targets[i])
                        epoch_val_mses.append(metrics["MSE"])
                        epoch_val_maes.append(metrics["MAE"])
                        if not np.isinf(metrics["PSNR"]):
                            epoch_val_psnrs.append(metrics["PSNR"])
                        epoch_val_ssims.append(metrics["SSIM"])
                    except Exception as metric_e:
                         print(f"\nWarning: Error calculating validation metrics: {metric_e}")
                         epoch_val_mses.append(np.nan)
                         epoch_val_maes.append(np.nan)
                         epoch_val_ssims.append(np.nan)

                val_dataloader_tqdm.set_postfix(loss=f"{val_batch_loss:.4f}")

        # --- End of Validation Epoch Calculation ---
        avg_val_loss = epoch_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
        avg_val_mse = np.nanmean(epoch_val_mses) if epoch_val_mses else 0
        avg_val_mae = np.nanmean(epoch_val_maes) if epoch_val_maes else 0
        avg_val_psnr = np.nanmean(epoch_val_psnrs) if epoch_val_psnrs else 0
        avg_val_ssim = np.nanmean(epoch_val_ssims) if epoch_val_ssims else 0

        history['val_loss'].append(avg_val_loss)
        history['val_mse'].append(avg_val_mse)
        history['val_mae'].append(avg_val_mae)
        history['val_psnr'].append(avg_val_psnr)
        history['val_ssim'].append(avg_val_ssim)

        # --- Epoch Summary ---
        epoch_duration = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch + 1}/{num_epochs}] ({epoch_duration:.2f}s) Summary:")
        print(f"  Train -> Loss: {avg_train_loss:.4f}, MSE: {avg_train_mse:.4f}, MAE: {avg_train_mae:.4f}, PSNR: {avg_train_psnr:.4f}, SSIM: {avg_train_ssim:.4f}")
        print(f"  Valid -> Loss: {avg_val_loss:.4f}, MSE: {avg_val_mse:.4f}, MAE: {avg_val_mae:.4f}, PSNR: {avg_val_psnr:.4f}, SSIM: {avg_val_ssim:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_save_path = os.path.join(model_save_dir, f'{base_filename}_best.pt')
            try:
                torch.save(model.state_dict(), best_model_save_path)
                print(f"  ** Best model saved based on val_loss: {best_val_loss:.4f} at epoch {epoch+1} **")
            except Exception as e:
                print(f"Error saving best model: {e}")


    # --- End of Training ---

    # --- Save Final Model ---
    final_model_save_path = os.path.join(model_save_dir, f'{base_filename}_final.pt')
    try:
        torch.save(model.state_dict(), final_model_save_path)
        print(f"Final model state_dict saved to: {final_model_save_path}")
    except Exception as e:
        print(f"Error saving final model state_dict: {e}")

    # --- Save Training History ---
    history_save_path = f"train_history_{base_filename}.txt"
    try:
        with open(history_save_path, "w") as f:
            # Write header
            header = "Epoch\t" + "\t".join([f"train_{k}" for k in history if k.startswith('train')]) + "\t" + "\t".join([f"val_{k}" for k in history if k.startswith('val')]) + "\n"
            f.write(header)
            # Write data
            for i in range(num_epochs):
                f.write(f"{i+1}\t")
                f.write(f"{history['train_loss'][i]:.4f}\t{history['train_mse'][i]:.4f}\t{history['train_mae'][i]:.4f}\t{history['train_psnr'][i]:.4f}\t{history['train_ssim'][i]:.4f}\t")
                f.write(f"{history['val_loss'][i]:.4f}\t{history['val_mse'][i]:.4f}\t{history['val_mae'][i]:.4f}\t{history['val_psnr'][i]:.4f}\t{history['val_ssim'][i]:.4f}\n")
        print(f"Training history saved to: {history_save_path}")
    except Exception as e:
        print(f"Error saving training history: {e}")


    # --- Inference using the BEST saved model ---
    print(f"\n--- Starting Inference using BEST model for lp = {lp} ---")
    # Load the *best* model saved during training
    inference_model = UNet_DA_CBAM_Selective(
        n_channels=N_CHANNELS_MODEL, n_classes=1,
        da_reduction=DA_REDUCTION_FACTOR, use_checkpoint=False
    ).to(device)
    # Use the path to the best saved model
    best_model_load_path = os.path.join(model_save_dir, f'{base_filename}_best.pt')
    try:
        if os.path.exists(best_model_load_path):
             inference_model.load_state_dict(torch.load(best_model_load_path, map_location=device))
             inference_model.eval()
             print(f"Best model ({best_model_load_path}) loaded successfully for inference.")
        else:
             print(f"Error: Best model file not found at {best_model_load_path}. Skipping inference.")
             continue # Skip inference if best model wasn't saved

    except Exception as e:
        print(f"Error loading best model state_dict for inference: {e}. Skipping inference.")
        continue

    inference_dataloader_tqdm = tqdm(val_dataloader, unit="image", desc="Inference (using best model)") # Use val_dataloader for inference
    inference_metrics_list = [] # To store metrics from inference run

    with torch.no_grad():
        for j, (sample_input, sample_target, filename) in enumerate(inference_dataloader_tqdm):
            sample_input = sample_input.to(device, non_blocking=True)
            try:
                sample_output = inference_model(sample_input)
            except Exception as inf_e:
                print(f"\nError during inference forward pass: {inf_e}. Skipping image {filename}")
                continue

            # Process filename
            if isinstance(filename, list) or isinstance(filename, tuple): filename = filename[0]
            date_str = os.path.splitext(filename)[0]
            output_filename_base = f"{date_str}_{lp}"

            # Get images on CPU
            output_image = sample_output.squeeze().cpu()
            target_image = sample_target.squeeze().cpu()

            # Save images
            pred_save_path = os.path.join(pred_folder, f"{output_filename_base}_predicted.png")
            save_image(output_image, pred_save_path, normalize=False)
            target_save_path = os.path.join(target_folder, f"{output_filename_base}_target.png")
            save_image(target_image, target_save_path, normalize=False)

            # Calculate and store inference metrics
            inf_metrics = calculate_metrics(output_image, target_image)
            inference_metrics_list.append(inf_metrics)

            # Save plot
            try:
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(output_image.numpy(), cmap='gray')
                title_text = f"Predicted (lp={lp})\nMSE={inf_metrics['MSE']:.4f} MAE={inf_metrics['MAE']:.4f}\nPSNR={inf_metrics['PSNR']:.2f} SSIM={inf_metrics['SSIM']:.4f}"
                plt.title(title_text, fontsize=9)
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.imshow(target_image.numpy(), cmap='gray')
                plt.title("Target Image")
                plt.axis('off')
                diff = torch.abs(output_image - target_image)
                plt.subplot(1, 3, 3)
                plt.imshow(diff.numpy(), cmap='gray')
                plt.title("Difference (Absolute)")
                plt.axis('off')
                plt.tight_layout()
                result_save_path = os.path.join(result_folder, f"{output_filename_base}.png")
                plt.savefig(result_save_path)
                plt.close()
            except Exception as plot_e:
                 print(f"\nError generating/saving plot for {output_filename_base}: {plot_e}")
                 plt.close()

    # --- Calculate and Print Average Inference Metrics ---
    if inference_metrics_list:
        avg_inf_mse = np.nanmean([m['MSE'] for m in inference_metrics_list])
        avg_inf_mae = np.nanmean([m['MAE'] for m in inference_metrics_list])
        avg_inf_psnr = np.nanmean([m['PSNR'] for m in inference_metrics_list if not np.isinf(m['PSNR'])]) # Exclude inf
        avg_inf_ssim = np.nanmean([m['SSIM'] for m in inference_metrics_list])
        print("\n--- Average Inference Metrics (using best model on validation set) ---")
        print(f"  MSE: {avg_inf_mse:.4f}, MAE: {avg_inf_mae:.4f}, PSNR: {avg_inf_psnr:.4f}, SSIM: {avg_inf_ssim:.4f}")
    else:
        print("\nNo inference metrics calculated.")


print("\n--- Script finished for all lp values. ---")
