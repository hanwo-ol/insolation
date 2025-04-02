

각 훈련 에폭(epoch)이 끝날 때 출력 및 저장되는 "Avg Metrics"가 어떻게 계산되는지 적음

## 훈련 중 에폭별 평균 메트릭은   
해당 에폭 동안 처리된 **모든 개별 학습 이미지 쌍**에 대해 계산된 메트릭 값들의 **산술 평균**입니다. 
* 즉, 각 배치(batch) 내의 모든 이미지에 대해 개별적으로 메트릭(MSE, MAE, PSNR, SSIM)을 계산하고, 이 값들을 에폭 전체에 걸쳐 누적한 다음, 마지막에 평균을 내는 방식입니다.

**수식**

* 에폭 $E$ 동안 처리된 총 학습 이미지의 수를 $N_E$라고 하고, 
* 각 에폭에서 $j$번째 학습 이미지 쌍에 대한 예측 이미지와 타겟 이미지를 각각 $I_{pred, j}$, $I_{target, j}$라고 할 때,
* 특정 메트릭 $Metric$에 대한 $j$번째 이미지 쌍의 계산 값을 $Metric(I_{pred, j}, I_{target, j})$라고 정의합니다.

1.  **Average MSE (평균 제곱 오차):**
    $AvgMSE_E = \frac{1}{N_E} \sum_{j=1}^{N_E} MSE(I_{pred, j}, I_{target, j})$
    (코드에서는 `np.nanmean`을 사용하므로, 만약 $MSE(I_{pred, j}, I_{target, j})$ 계산 중 NaN이 발생했다면 해당 $j$는 합계와 $N_E$ 카운트에서 제외됩니다.)

2.  **Average MAE (평균 절대 오차):**
    $AvgMAE_E = \frac{1}{N_E} \sum_{j=1}^{N_E} MAE(I_{pred, j}, I_{target, j})$
    (NaN 값은 `np.nanmean`에 의해 제외)

3.  **Average PSNR (평균 최대 신호 대 잡음비):**
    PSNR 계산 시 MSE가 0이면 PSNR은 무한대($\infty$)가 됩니다. 처리과정에서는 이 무한대 값을 평균 계산에서 제외합니다.   
    따라서 유한한(finite) PSNR 값들의 집합을 $M_{E, PSNR}$이라고 하고, 이 집합의 원소 개수(유한한 PSNR 값의 개수)를 $|M_{E, PSNR}|$이라고 하면 다음과 같습니다:
    $AvgPSNR_E = \frac{1}{|M_{E, PSNR}|} \sum_{m \in M_{E, PSNR}} m$
    (여기서 $m = PSNR(I_{pred, j}, I_{target, j})$ 이고 $m \neq \infty$ 입니다. NaN 값 또한 `np.nanmean`에 의해 제외됩니다.)

5.  **Average SSIM (평균 구조적 유사성 지수):**
    $AvgSSIM_E = \frac{1}{N_E} \sum_{j=1}^{N_E} SSIM(I_{pred, j}, I_{target, j})$
    (NaN 값은 `np.nanmean`에 의해 제외)
