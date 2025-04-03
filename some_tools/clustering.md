# Python 코드 설명(Clustering Analysis)
특정 기준(DA_Level, LP)에 따라 데이터를 그룹화하고, 각 그룹 내에서 K-평균(K-Means) 군집화를 수행하는 분석 코드입니다. 
* 분석 결과를 시각화하고 파일로 저장하는 기능도 포함되어 있습니다.

## 코드 주요 부분 설명

### 특성 선택 (Feature Selection):

``` python 
orig_metrics = [col for col in df.columns if col.endswith('_Orig')]
```
* DataFrame의 컬럼 이름 중 _Orig로 끝나는 모든 컬럼을 찾아 orig_metrics 리스트에 저장합니다.
* 이 컬럼들이 군집화에 사용될 특성(feature)이 됩니다.
  * Clip, MinMax는 Orig와 별 차이가 없다고 생각했습니다.

### 그룹별 군집화:

``` python
grouped = df.groupby(['DA_Level', 'LP'])
```
* DA_Level과 LP 컬럼의 값 조합을 기준으로 DataFrame을 그룹화합니다.
* 각 그룹에 대해 반복 작업을 수행합니다.

### 전처리 (Preprocessing):
``` python 
scaler = StandardScaler()
```
* 표준화 객체를 생성합니다.

``` python 
scaled_features = scaler.fit_transform(features)
```
* 선택된 특성 데이터를 표준화합니다 (평균 0, 분산 1). 이는 각 특성의 스케일 차이가 군집화 결과에 미치는 영향을 줄이기 위함입니다.


### 최적 군집 개수(k) 결정 (Elbow Method):
``` python
max_k = min(10, len(group) - 1)
```
* Elbow Method를 시도할 최대 k값을 설정합니다.


n_clusters = visualizer.elbow_value_: 시각화 결과에서 자동으로 'elbow' 지점(최적의 k값으로 간주되는 지점)을 찾습니다.

Elbow 값 부재 처리: 명확한 elbow 지점을 찾지 못하면 경고 메시지를 출력하고, 기본값(예: 3 또는 가능한 최대 k값)으로 n_clusters를 설정합니다.

Elbow Plot 저장: Elbow Method 시각화 그래프를 PNG 파일로 output_dir에 저장합니다. (elbow_plot_DA{da_level}_LP{lp}.png)


### K-평균 군집화 수행:

kmeans = KMeans(n_clusters=n_clusters, ...): 결정된 n_clusters 개수만큼의 군집을 찾도록 K-평균 모델을 초기화합니다. random_state=42는 결과 재현성을 위함이고, n_init=10은 다른 초기 중심점으로 10번 실행하여 가장 좋은 결과를 선택하도록 합니다.

cluster_labels = kmeans.fit_predict(scaled_features): 표준화된 데이터에 K-평균 알고리즘을 적용하여 각 데이터 포인트가 속하는 군집 번호(label)를 예측합니다.

### 군집 레이블 할당:

원본 그룹 데이터(group)를 복사하고, Cluster라는 새 컬럼에 예측된 군집 레이블(cluster_labels)을 추가합니다.

이 결과를 results_list에 추가합니다.

### 차원 축소 (PCA) 및 시각화:

조건 확인: 군집 개수가 1개 초과이고 그룹 내 데이터가 2개 이상일 때만 PCA 및 시각화를 수행합니다. (단일 군집이거나 데이터가 너무 적으면 PCA 의미가 없거나 불가능)

pca = PCA(n_components=2, ...): 데이터를 2차원으로 축소하기 위한 PCA 모델을 설정합니다.

pca_components = pca.fit_transform(scaled_features): 표준화된 데이터에 PCA를 적용하여 2개의 주성분(PCA1, PCA2)을 추출합니다.

PCA 결과 DataFrame 생성: 추출된 PCA 결과를 DataFrame (pca_df)으로 만듭니다.

결합: 원본 그룹 정보, 군집 레이블, PCA 결과를 합쳐 시각화용 DataFrame (group_pca_vis)을 만듭니다. 이 결과를 pca_results_list에 추가합니다.


### 군집 특성 분석

df_clustered.groupby(['DA_Level', 'LP', 'Cluster'])[orig_metrics].mean(): 각 DA_Level, LP, Cluster 조합별로 원본 메트릭(orig_metrics)들의 평균값을 계산합니다. 이를 통해 각 군집의 특성을 파악할 수 있습니다.

cluster_analysis['Size'] = ...: 각 군집의 크기(데이터 포인트 수)를 계산하여 추가합니다.

분석 결과를 출력하고 CSV 파일로 저장합니다. (cluster_analysis_means.csv)

특정 군집 파일 식별 예시:

PSNR_Orig 컬럼이 분석 결과에 있다면:

### 결과물
이 코드를 실행하면 output_dir (clustering_analysis_160) 디렉토리에 다음과 같은 파일들이 생성됩니다 (단, 처리된 그룹이 있는 경우):

elbow_plot_DA{...}_LP{...}.png: 각 DA_Level, LP 그룹별 Elbow Method 시각화 그래프 (최적 k 결정 과정).

pca_clusters_DA{...}_LP{...}.png: 각 DA_Level, LP 그룹별 PCA 기반 군집 시각화 그래프 (2개 이상의 군집이 있고 데이터가 충분한 경우).

data_with_clusters.csv: 원본 데이터에 각 데이터 포인트가 속한 군집 번호 (Cluster 컬럼)가 추가된 전체 데이터 파일.

data_with_pca_and_clusters.csv: data_with_clusters.csv 내용에 PCA 결과 (PCA1, PCA2 컬럼)가 추가된 데이터 파일 (PCA가 수행된 경우).

cluster_analysis_means.csv: 각 DA_Level, LP, Cluster 조합별 원본 메트릭 평균값과 군집 크기를 보여주는 분석 요약 파일.

