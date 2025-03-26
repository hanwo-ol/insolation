paper: **DA-TRANSUNET: INTEGRATING SPATIAL AND CHANNEL DUAL ATTENTION WITH TRANSFORMER U-NET FOR MEDICAL IMAGE SEGMENTATION**

#### 3.2 Dual Attention Block (DA-Block) 이해를 위한 게시물

# DA-Block (Dual Attention Block)의 목적

* DA-Block은 Image segmentation 모델(특히 U-Net based architecture)의 성능을 향상시키기 위해 설계된 모듈임.
* 기존의 트랜스포머는 전체적인(global) 관계를 파악하는 데 강점이 있긴 함, 그런데 이미지 내의 특정 '위치(position)' 정보나 '채널(channel)' 간의 관계와 같은 이미지 고유의 특징(image-specific features)을 세밀하게 포착하는 데는 한계가 있음.
* 반면, 컨볼루션 신경망(CNN)은 Local feature 추출에는 강하지만, 이미지 전체의 넓은 문맥을 파악하기는 어렵다는 단점이 있음.
* DA-Block은 이러한 한계를 극복하기 위해 **위치 어텐션(PAM)** 과 **채널 어텐션(CAM)** 두 가지 메커니즘을 결합하여 이미지의 공간적(spatial) 정보와 채널 간의 상호 의존성을 모두 효과적으로 학습할 수 있게 했음.
* 모듈의 목표는 더 풍부하고 정확한 특징 표현(feature representation)을 생성하여 분할 성능을 개선하는 것임.

# DA-Block의 구성 요소   

<img width="815" alt="image" src="https://github.com/user-attachments/assets/38d9d809-d039-4265-bb78-a5c7828fb75d" />   

(위는 그림 2)   
DA-Block은 크게 두 개의 병렬적인 브랜치(branch)로 구성되는데, 
각 브랜치는 입력 특징 맵을 받아 각각 PAM과 CAM을 통해 처리한 후, 그 결과를 합쳐 최종 출력을 생성하는 구조임. (그림 2를 보면 됨)

## PAM (Position Attention Module - 위치 어텐션 모듈)   
<img width="395" alt="image" src="https://github.com/user-attachments/assets/255f345f-4940-41ea-bdb1-a5893c82dcb1" />   

(위는 그림 3)   
* **목표:** 이미지 내의 서로 다른 위치(픽셀)들 간의 공간적인 관계를 학습하고자 함.
  * 즉, 특정 위치의 특징을 계산할 때 이미지 내의 모든 다른 위치들의 특징을 얼마나 중요하게 고려할지를 결정하는 것. 

* **작동 방식 (그림 3 참조):**
  * 입력 특징 맵 A ($C \times H \times W$)가 들어오면, 세 개의 다른 컨볼루션 레이어를 거쳐 특징 맵 B, C, D를 생성. (각 크기는 $C \times H \times W$임.)
  * B와 C를 각각 $C \times N$ ($N=H \times W$) 형태로 재구성(reshape). 각 이미지의 N은 픽셀의 수임.
  * C의 전치(transpose)와 B를 행렬 곱셈하고, 소프트맥스(softmax) 함수를 적용하여 **공간 어텐션 맵 S** ($N \times N$)를 계산. (아래 수식)

$$S_{ji}=\dfrac{exp(B_{i}\cdot C_{j})}{\sum_{i=1}^{N}exp(B_{i}\cdot C_{j})}$$

  * $S_{ji}$는 i번째 위치가 j번째 위치의 특징을 계산하는 데 얼마나 영향을 미치는지를 나타내는 가중치임.
  * D를 $C \times N$ 형태로 reshape.
  * D($C \times N$)와 S($N \times N$)의 전치(transpose)를 행렬 곱셈하고, 결과를 다시 $C \times H \times W$ 형태로 재구성 reshape.
  * 이 결과에 학습 가능한 가중치 파라미터 α를 곱한 후, 원본 입력 특징 맵 A와 더하여 최종 출력 E ($C \times H \times W$)를 얻음.(alpha는 아래 수식을 보면 됨.)   
$$E_{j}=\alpha\sum_{i=1}^{N}(s_{ji}D_{i})+A_{j}$$

* **PAM을 사용하는 이유?**
  * 각 위치의 특징을 계산할 때 전체 이미지의 공간적 문맥을 고려하게 됨, 멀리 떨어진 위치 간의 연관성도 파악할 수 있게 된다고 함.

## CAM (Channel Attention Module - 채널 어텐션 모듈)   
<img width="421" alt="image" src="https://github.com/user-attachments/assets/b59c4d6a-f14c-4f58-b317-0e85c8b5b45c" />    

(위는 그림 4임)   
* **목표:** 특징 맵의 여러 채널(channel)들 간의 상호 의존성을 학습하기 위한 모듈.
  * 즉, 어떤 채널 특징이 다른 채널 특징과 연관성이 높은지를 파악하고 이를 특징 계산에 반영하는 방식.

* **작동 방식 (그림 4 참조):**
  * 입력 특징 맵 A ($C \times H \times W$)를 $C \times N$ 형태로 reshape.
  * 재구성된 A와 A의 전치(transpose)를 행렬 곱셈하고, 소프트맥스 함수를 적용하여 **채널 어텐션 맵 X** ($C \times C$)를 생성. (아래 수식)

$$x_{ji}=\dfrac{exp(A_{i}\cdot A_{j})}{\sum_{i=1}^{C}exp(A_{i}\cdot A_{j})}$$ 

  * $x_{ji}$는 i번째 채널이 j번째 채널에 얼마나 영향을 미치는지를 나타내는 가중치가 됨.
  * X의 전치(transpose)와 재구성된 A를 행렬 곱셈하고, 결과를 다시 $C \times H \times W$ 형태로 reshape.
  * 위의 결과에 학습 가능한 가중치 **파라미터 β**를 곱한 후, 원본 입력 특징 맵 A와 더하여 최종 출력 E ($C \times H \times W$)를 아래 수식과 같이 얻음. (아래 수식)
$$E_{j}=\beta\sum_{i=1}^{N}(x_{ji}A_{i})+A_{j}$$

* **CAM을 사용하는 이유?**
  * 각 채널의 특징을 계산할 때 다른 모든 채널과의 관계를 고려하게 되어, 채널 간의 유의미한 정보를 강조하고 불필요한 정보는 억제하는 효과를 기대할 수 있음.

## DA-Block 아키텍처 통합 (아래 그림 2)

<img width="611" alt="image" src="https://github.com/user-attachments/assets/01c7b173-33ea-478e-9740-d7eb3356cfae" />   

* 입력 특징 맵은 두 개의 병렬 경로로 나눔(위에서 본 PAM, CAM임).
* 각 경로는 먼저 컨볼루션을 통해 채널 수를 줄여(1/16로 스케일링) 계산 효율성을 높임. (수식 5, 7)

$$\alpha^{1}=Conv(input)$$
$$\alpha^{2}=Conv(input)$$

* 한 경로는 PAM을 통과하고, 다른 경로는 CAM을 통과시킴.
$$\hat{\alpha^{1}}=Conv(PAM(\alpha^{1}))$$
$$\hat{\alpha^{2}}=Conv(CAM(\alpha^{2}))$$

* 각 경로의 PAM/CAM 출력은 다시 컨볼루션을 통과시킴.
* 두 경로의 출력을 합산(element-wise sum).
* 마지막으로 다시 컨볼루션을 통과시켜 채널 수를 원래대로 복원하고 최종 DA-Block 출력을 생성.    
$$output=Conv(\hat{\alpha^{1}}+\hat{\alpha^{2}})$$

* **결론:** 이 구조는 위치와 채널 정보를 동시에 고려하여 특징 표현을 풍부하게 만들고, 이를 통해 모델의 전체적인 분할 성능을 향상시키게 된다는 것.

DA-Block은 트랜스포머의 전역적 특징 추출 능력과 U-Net의 지역적 특징 추출 능력을 보완하며, 이미지 고유의 위치 및 채널 정보를 효과적으로 활용하여 의료 영상 분할과 같은 복잡한 작업에서 더 나은 성능을 달성하도록 도울 수 있게 됨.
