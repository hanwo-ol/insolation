## Image Evaluation Metrics Explained

These metrics quantify the difference or similarity between a predicted image (K) and a target (ground truth) image (I). The images have dimensions $m \times n$ (height $m$, width $n$).

### 1. Mean Squared Error (MSE)

*   **Concept:** Measures the average squared difference between corresponding pixel values in the two images.
*   **Formula:**
    $$
    \text{MSE}(I, K) = \frac{1}{m \times n} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} [I(i, j) - K(i, j)]^2
    $$
    
*   **Variables:**
    *   $m, n$: Height and width of the images.
    *   $I(i, j)$: Pixel value at coordinate $(i, j)$ in the target image $I$.
    *   $K(i, j)$: Pixel value at coordinate $(i, j)$ in the predicted image $K$.
*   **Interpretation:**
    *   Range: $[0, \infty)$.
    *   A lower MSE value indicates less error and better similarity between the images. An MSE of 0 means the images are identical.

### 2. Mean Absolute Error (MAE)

*   **Concept:** Measures the average absolute difference between corresponding pixel values. It treats all errors linearly, making it less sensitive to large outliers compared to MSE.
*   **Formula:**
    $$
    \text{MAE}(I, K) = \frac{1}{m \times n} \sum_{i=0}^{m-1} \sum_{j=0}^{n-1} |I(i, j) - K(i, j)|
    $$
*   **Variables:**
    *   $m, n$: Height and width of the images.
    *   $I(i, j)$: Pixel value at coordinate $(i, j)$ in the target image $I$.
    *   $K(i, j)$: Pixel value at coordinate $(i, j)$ in the predicted image $K$.
*   **Interpretation:**
    *   Range: $[0, \infty)$.
    *   A lower MAE value indicates less error and better similarity. An MAE of 0 means the images are identical. It represents the average pixel intensity difference.

### 3. Peak Signal-to-Noise Ratio (PSNR)

*   **Concept:** Measures the ratio between the maximum possible power of a signal (the maximum pixel value) and the power of the corrupting noise (the MSE). It's commonly used to assess the quality of reconstruction in lossy compression or image restoration.
*   **Formula:**
    $$
    \text{PSNR}(I, K) = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}(I, K)} \right)
    $$
    Alternatively, using the square root of MSE:
    $$
    \text{PSNR}(I, K) = 20 \cdot \log_{10} \left( \frac{\text{MAX}_I}{\sqrt{\text{MSE}(I, K)}} \right)
    $$
*   **Variables:**
    *   $\text{MAX}_I$: The maximum possible pixel value for the image type. For an 8-bit grayscale image, $\text{MAX}_I = 255$. For images normalized to $[0, 1]$, $\text{MAX}_I = 1.0$. For images normalized to $[-1, 1]$, convention varies, but often $\text{MAX}_I$ is considered the *range* (2.0) or the maximum value (1.0), however, **the `calculate_metrics` code provided uses $\text{MAX}_I=255$**, implying the MSE used in the formula *must* also be calculated on images scaled to the 0-255 range.
    *   $\text{MSE}(I, K)$: The Mean Squared Error between the target image $I$ and the predicted image $K$, calculated on the same scale as $\text{MAX}_I$.
*   **Interpretation:**
    *   Range: Typically positive values, measured in decibels (dB). Can be $\infty$ if MSE is 0.
    *   A **higher** PSNR value indicates better quality (less noise/error relative to the signal).

### 4. Structural Similarity Index (SSIM)

*   **Concept:** Measures the similarity between two images by considering changes in structural information, luminance, and contrast, aiming to align better with human visual perception than MSE or PSNR. It's calculated locally over image windows.
*   **Formula (for a single local window):**
    $$
    \text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
    $$
    The overall SSIM score for the image is typically the mean of the SSIM values calculated over all local windows.
*   **Variables (for a local window x from I, y from K):**
    *   $\mu_x, \mu_y$: Local mean of window $x$ and $y$.
    *   $\sigma_x^2, \sigma_y^2$: Local variance of window $x$ and $y$.
    *   $\sigma_{xy}$: Local covariance of windows $x$ and $y$.
    *   $C_1 = (k_1 L)^2$, $C_2 = (k_2 L)^2$: Constants to stabilize division with weak denominators.
    *   $L$: The dynamic range of the pixel values (e.g., 255 for 8-bit images). **The `calculate_metrics` code provided uses $L=255$**.
    *   $k_1, k_2$: Small constant values (defaults are typically $k_1=0.01$, $k_2=0.03$).
*   **Interpretation:**
    *   Range: Typically $[-1, 1]$, often $[0, 1]$ in image processing contexts.
    *   A value of 1 indicates perfect structural similarity. Values closer to 1 indicate higher similarity. A **higher** SSIM value is better.

---
