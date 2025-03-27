이 모듈은 입력 특징 맵(feature map)에 대해 **공간적(spatial)** 및 **채널(channel)** 차원 모두에서 중요한 특징을 강조하고 long-range dependency(장거리 의존성)를 포착하기 위해 설계되었습니다. 크게 두 가지 주요 구성 요소인 **PAM (Position Attention Module)**과 **CAM (Channel Attention Module)**으로 나뉩니다.

**1. `DualAttentionModule` 클래스**

* **초기화 (`__init__`)**:
    * `in_channels`: 입력 특징 맵의 채널 수를 받습니다.
    * `reduction_factor`: PAM과 CAM 내부 연산 시 채널 수를 줄이는 비율입니다. 계산 효율성을 높이기 위함입니다. `inter_channels` (중간 채널 수)는 `in_channels // reduction_factor`로 계산되며, 최소 1이 되도록 보장합니다.
    * `use_checkpoint`: 학습 중 메모리 사용량을 줄이기 위해 gradient checkpointing 사용 여부를 결정합니다.
    * **입력 Convolution (`conv_in_pa`, `conv_in_ca`)**: 입력 특징 맵(`x`)을 PAM과 CAM에 각각 넣기 전에 채널 수를 `in_channels`에서 `inter_channels`로 줄이는 3x3 Convolution, Normalization(GroupNorm 또는 BatchNorm), ReLU 활성화 함수로 구성된 블록입니다. 각 모듈은 독립된 입력 컨볼루션을 가집니다.
    * **Attention 모듈 인스턴스화**: `PAM_Module(inter_channels)`와 `CAM_Module(inter_channels)`을 생성합니다. 이 모듈들은 줄어든 `inter_channels` 차원에서 연산을 수행합니다.
    * **출력 Convolution (`conv_out_pa`, `conv_out_ca`)**: PAM과 CAM의 출력(`sa_feat`, `sc_feat`)을 각각 받아 처리하는 3x3 Convolution, Normalization, ReLU 블록입니다. 이 단계에서도 채널 수는 `inter_channels`로 유지됩니다.
    * **최종 출력 Convolution (`conv_out`)**: `conv_out_pa`와 `conv_out_ca`의 출력을 합친 후(element-wise sum), Dropout을 적용하고 1x1 Convolution을 사용하여 채널 수를 다시 원래의 `in_channels`로 복원합니다. 즉, 이 모듈은 최종적으로 입력과 동일한 채널 수를 출력합니다.

* **순전파 (`forward`)**:
    1.  **Position Attention Branch**:
        * 입력 `x`가 `conv_in_pa`를 통과하여 `inter_channels`로 줄어든 `feat_pa` 생성됩니다.
        * `feat_pa`가 `PAM_Module`(`self.pam`)을 통과하여 공간적 주의(spatial attention)가 적용된 `sa_feat`이 생성됩니다. (`use_checkpoint`가 True이고 학습 중이면 메모리 절약을 위해 `checkpoint` 사용)
        * `sa_feat`이 `conv_out_pa`를 통과하여 `sa_conv`가 생성됩니다.
    2.  **Channel Attention Branch**:
        * 입력 `x`가 `conv_in_ca`를 통과하여 `inter_channels`로 줄어든 `feat_ca` 생성됩니다.
        * `feat_ca`가 `CAM_Module`(`self.cam`)을 통과하여 채널 주의(channel attention)가 적용된 `sc_feat`이 생성됩니다. (`checkpoint` 사용 가능)
        * `sc_feat`이 `conv_out_ca`를 통과하여 `sc_conv`가 생성됩니다.
    3.  **결합 및 최종 출력**:
        * `sa_conv`와 `sc_conv`를 **element-wise 덧셈**으로 결합하여 `feat_sum`을 만듭니다.
        * `feat_sum`이 최종 `conv_out` (Dropout + 1x1 Conv)을 통과하여 원래 채널 수(`in_channels`)를 가진 최종 `output`을 생성합니다.
    4.  최종 `output`을 반환합니다. 이 출력은 공간적, 채널적 문맥 정보가 보강된 특징 맵입니다.

**2. `PAM_Module` (Position Attention Module)**

* **목표**: 특징 맵 내의 임의의 두 위치 간의 공간적 상관관계를 모델링하여 공간적 문맥 정보를 포착합니다.
* **초기화 (`__init__`)**:
    * `in_dim`: 입력 채널 수 (`inter_channels`)를 받습니다.
    * `query_conv`, `key_conv`, `value_conv`: 입력 특징 맵으로부터 Query, Key, Value를 생성하기 위한 1x1 Convolution 레이어입니다. Query와 Key는 계산 효율성을 위해 채널 수를 `in_dim // 8`로 줄입니다.
    * `gamma`: Attention 결과의 중요도를 학습하기 위한 스케일링 파라미터 (0으로 초기화).
    * `softmax`: Attention map을 생성하기 위한 Softmax 함수.
* **순전파 (`forward`)**:
    1.  입력 `x` (B, C', H', W')에 `query_conv`, `key_conv`, `value_conv`를 각각 적용합니다.
    2.  Query와 Key를 (B, H'W', C''/8) 및 (B, C''/8, H'W') 형태로 reshape하고 batch matrix multiplication (`torch.bmm`)을 수행하여 `energy` (B, H'W', H'W')를 계산합니다. 이는 모든 위치 쌍 간의 유사도를 나타냅니다.
    3.  `energy`에 Softmax를 적용하여 **Spatial Attention Map** (`attention`, B, H'W', H'W')을 얻습니다.
    4.  Value를 (B, C', H'W') 형태로 reshape하고, `attention`의 전치(transpose)와 batch matrix multiplication을 수행합니다. 이는 Attention Map의 가중치에 따라 Value 특징들을 가중합하는 과정입니다.
    5.  결과를 다시 (B, C', H', W') 형태로 reshape합니다.
    6.  `gamma * out + x` : Attention이 적용된 결과에 `gamma`를 곱하고 원본 입력 `x`를 더합니다 (Residual Connection). 이를 통해 네트워크는 Attention 결과와 원본 특징 중 어느 것에 더 비중을 둘지 학습할 수 있습니다.
    7.  최종 결과를 반환합니다.

**3. `CAM_Module` (Channel Attention Module)**

* **목표**: 모든 채널 간의 상호 의존성을 모델링하여 채널 관점에서의 문맥 정보를 포착합니다.
* **초기화 (`__init__`)**:
    * `in_dim`: 입력 채널 수 (`inter_channels`)를 받습니다.
    * `gamma`: Attention 결과의 중요도를 학습하기 위한 스케일링 파라미터 (0으로 초기화).
    * `softmax`: Attention map을 생성하기 위한 Softmax 함수.
* **순전파 (`forward`)**:
    1.  입력 `x` (B, C', H', W')를 Query, Key, Value로 사용하기 위해 reshape 합니다.
    2.  Query (B, C', H'W')와 Key의 전치 (B, H'W', C')를 batch matrix multiplication (`torch.bmm`)하여 `energy` (B, C', C')를 계산합니다. 이는 모든 채널 쌍 간의 유사도(상관관계)를 나타냅니다.
    3.  `energy`에 특정 정규화(코드 내 `energy_new` 계산 부분)를 적용한 후 Softmax를 적용하여 **Channel Attention Map** (`attention`, B, C', C')을 얻습니다.
    4.  Value (B, C', H'W')에 `attention`을 batch matrix multiplication합니다. 이는 Attention Map의 가중치에 따라 채널 특징들을 가중합하는 과정입니다.
    5.  결과를 다시 (B, C', H', W') 형태로 reshape합니다.
    6.  `gamma * out + x` : Attention이 적용된 결과에 `gamma`를 곱하고 원본 입력 `x`를 더합니다 (Residual Connection).
    7.  최종 결과를 반환합니다.

**요약**:

`DualAttentionModule`은 입력 특징 맵을 받아 PAM과 CAM을 병렬적으로 통과시킵니다. PAM은 "어떤 위치가 중요한가?"를 학습하고, CAM은 "어떤 채널이 중요한가?"를 학습합니다. 각 모듈의 결과는 별도의 컨볼루션을 거친 후 합쳐지고, 마지막으로 1x1 컨볼루션을 통해 원래 채널 수로 복원되어 공간적, 채널적 문맥 정보가 풍부해진 특징 맵을 출력합니다. 중간 채널 수를 줄이고(`reduction_factor`), 필요시 gradient checkpointing (`use_checkpoint`)을 사용하여 메모리 효율성을 높입니다.
