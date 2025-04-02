## Per-Epoch Metric Calculation Explanation

The script calculates average metrics for each training epoch by first calculating metrics for individual images within each batch and then averaging these values across the entire epoch.

**Notation:**

*   $E$: The current epoch number (from 1 to `num_epochs`).
*   $N_{\text{batches}}$: The total number of batches in the training dataloader for the current epoch.
*   $B$: The `batch_size`.
*   $N_{\text{images}} = N_{\text{batches}} \times B$: The total number of images processed in the epoch.
*   $b$: Index for the current batch ($1 \le b \le N_{\text{batches}}$).
*   $i$: Index for the current image within a batch ($1 \le i \le B$).
*   $O_{b,i}$: The $i$-th predicted output image tensor in batch $b$.
*   $T_{b,i}$: The $i$-th target (ground truth) image tensor in batch $b$.
*   $H, W$: Height and Width of the images.
*   $p$: Index for a pixel within an image ($1 \le p \le H \times W$).
*   $O_{b,i,p}, T_{b,i,p}$: Pixel values at index $p$ for the predicted and target images, respectively (assumed to be normalized, e.g., in [-1, 1]).
*   $O_{b,i}^{\text{(0-255)}}, T_{b,i}^{\text{(0-255)}}$: Image pixel values scaled and clipped to the [0, 255] range.
*   `criterion`: The MS-SSIM-L1 loss function instance.
*   `calculate_metrics`: The Python function defined to compute MSE, MAE, PSNR, SSIM for a single image pair.

--- 

### 1. Average Epoch Loss ($L_{\text{epoch}}$)

*   **Concept:** The average value of the loss function (`criterion`) computed over all batches in the epoch. This reflects the primary objective the model is trying to minimize during training.
*   **Within Batch:** For each batch $b$, the script calculates the loss *before* the final optimizer step (using the outputs from the first SAM closure):   
    $L_{\text{batch}, b} = \text{criterion}(O_{b}, T_{b})$   
    where $O_b$ and $T_b$ represent the batch of outputs and targets. The `.item()` method extracts the scalar loss value.
    
*   **Per Epoch Calculation:** The script sums the scalar loss values from each batch and divides by the total number of batches.
    $$L_{\text{epoch}} = \frac{1}{N_{\text{batches}}} \sum_{b=1}^{N_{\text{batches}}} L_{\text{batch}, b}$$   
    This value is stored in the `train_losses` list.



### 2. Average Epoch Mean Squared Error ( $MSE_{\text{epoch}}$ )

* **Concept:** The average squared difference between predicted and target pixel values, averaged over all images processed in the epoch. Penalizes larger errors more heavily.
* **Within Batch (per image):** For each image pair $(O_{b,i}, T_{b,i})$, the `` `calculate_metrics` `` function computes:
    $$MSE_{b,i} = \frac{1}{H \times W} \sum_{p=1}^{H \times W} (O_{b,i,p} - T_{b,i,p})^2$$

* **Per Epoch Calculation:** The script collects all $MSE_{b,i}$ values into the `` `epoch_mses` `` list and calculates the mean using `` `np.nanmean` `` (to handle potential NaNs if metric calculation failed).
    $$MSE_{\text{epoch}} = \text{Mean} ( \{ MSE_{b,i} \mid \forall {b, i} \}) \approx \dfrac{1}{N_{\text{valid\_images}}} \sum_{b=1}^{N_{\text{batches}}} \sum_{i=1}^{B} MSE_{b,i}$$

    where $N_{\text{valid\_images}}$ is the count of images for which MSE could be computed. This value is stored in the `` `train_mses` `` list.



### 3. Average Epoch Mean Absolute Error ($MAE_{\text{epoch}}$)

*   **Concept:** The average absolute difference between predicted and target pixel values, averaged over all images processed in the epoch. Measures the average magnitude of errors.
*   **Within Batch (per image):** For each image pair $(O_{b,i}, T_{b,i})$, the `calculate_metrics` function computes:
    $$
    MAE_{b,i} = \frac{1}{H \times W} \sum_{p=1}^{H \times W} |O_{b,i,p} - T_{b,i,p}|
    $$
    *(Note: The Python code uses `F.l1_loss(output_image_cpu, target_image_cpu).item()` which calculates this mean value directly.)*
*   **Per Epoch Calculation:** Similar to MSE, the script collects all $MAE_{b,i}$ values into `epoch_maes` and calculates the mean using `np.nanmean`.
    $$
    MAE_{\text{epoch}} = \text{Mean}( \{ MAE_{b,i} \mid \forall b, i \} ) \approx \frac{1}{N_{\text{valid\_images}}} \sum_{b=1}^{N_{\text{batches}}} \sum_{i=1}^{B} MAE_{b,i}
    $$
    This value is stored in the `train_maes` list.

---

### 4. Average Epoch Peak Signal-to-Noise Ratio ($PSNR_{\text{epoch}}$)

*   **Concept:** Measures the ratio between the maximum possible power of a signal (image) and the power of corrupting noise (error) that affects its fidelity. Higher values indicate better quality (less noise/error relative to the signal). It's usually measured in decibels (dB).
*   **Within Batch (per image):** The `calculate_metrics` function first calculates the MSE on the 0-255 scaled images ($MSE_{b,i}^{\text{(0-255)}}$). Then:
    $$
    PSNR_{b,i} =
    \begin{cases}
    \infty & \text{if } MSE_{b,i}^{\text{(0-255)}} = 0 \\
    20 \log_{10} \left( \frac{MAX_I}{\sqrt{MSE_{b,i}^{\text{(0-255)}}}} \right) & \text{if } MSE_{b,i}^{\text{(0-255)}} > 0
    \end{cases}
    $$
    where $MAX_I = 255$ (maximum pixel value for 8-bit images).
*   **Per Epoch Calculation:** The script collects *only finite* $PSNR_{b,i}$ values (where $MSE > 0$) into the `epoch_psnrs` list and calculates the mean using `np.nanmean`.
    $$
    PSNR_{\text{epoch}} = \text{Mean}( \{ PSNR_{b,i} \mid \forall b, i \text{ where } PSNR_{b,i} \neq \infty \} )
    $$
    This value is stored in the `train_psnrs` list.

---

### 5. Average Epoch Structural Similarity Index ($SSIM_{\text{epoch}}$)

*   **Concept:** Measures the similarity between two images based on perceived changes in structural information, luminance, and contrast. It's designed to align better with human visual perception of similarity than MSE or PSNR. Values range from -1 to 1, where 1 indicates identical images.
*   **Within Batch (per image):** The `calculate_metrics` function uses `skimage.metrics.structural_similarity` on the 0-255 scaled images:
    $$
    SSIM_{b,i} = \text{ssim}(O_{b,i}^{\text{(0-255)}}, T_{b,i}^{\text{(0-255)}})
    $$
    The `ssim` function internally compares local means ($\mu$), standard deviations ($\sigma$), and cross-covariance ($\sigma_{xy}$) using specific weighting factors and stability constants.
*   **Per Epoch Calculation:** Similar to MSE/MAE, the script collects all $SSIM_{b,i}$ values into `epoch_ssims` and calculates the mean using `np.nanmean`.
    $$
    SSIM_{\text{epoch}} = \text{Mean}( \{ SSIM_{b,i} \mid \forall b, i \} ) \approx \frac{1}{N_{\text{valid\_images}}} \sum_{b=1}^{N_{\text{batches}}} \sum_{i=1}^{B} SSIM_{b,i}
    $$
    This value is stored in the `train_ssims` list.

---

These epoch-averaged metrics provide a summary of the model's performance on the entire training dataset for that specific training epoch. They are then saved to the `train_metrics_...csv` file for later analysis.
