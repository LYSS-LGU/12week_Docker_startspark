# Spark MLlib

> // Spark_MLlib.md

## Machine Learning Library

머신러닝을 쉽고 확장성 있게 적용할 수 있는 라이브러리이다. 데이터 프레임과 연동해 효율적으로 머신러닝 모델을 개발할 수 있다.

![image.png](todaylearn/Spark%20MLlib/image.png)

## MLLib 으로 할 수 있는 것들

![image.png](todaylearn/Spark%20MLlib/image%201.png)

- Feature Engineering
- 통계적인 연산 EDA
- 일반적인 ML 알고리즘(딥러닝 제외)
    - 회귀(Regression): Linear 예측, Logistic 분류 알고리즘이다.
    - Decision Tree, SVM(Support Vector Machine): 분류와 회귀에 모두 사용 가능한 알고리즘이다.
    - Naïve Bayes: 조건부확률 기반 분류로 스팸감지와 텍스트 분류에 효과적이다.
    - 군집 알고리즘: K-Means 클러스터링이다.
    - 추천: Alternating Least Squares 알고리즘이다.
- 파이프라인: 전처리부터 튜닝까지 전체 과정을 지원하며, 필요시 알고리즘을 교체할 수 있다.
- 피처엔지니어링: 데이터에서 필요한 정보를 추출하고 변형하여 인사이트를 찾는 과정이다. 머신러닝 모델이 사용할 데이터를 준비하는 단계이다.
- 유틸스: 선형대수, 통계 등 수학적 공식을 모아놓은 라이브러리이다.

## *머신러닝 파이프라인

![image.png](todaylearn/Spark%20MLlib/image%202.png)

- 각 단계를 스테이지라고 한다.
- 데이터 로딩 > 전처리 > 학습 > 평가 스테이지를 연결해 플로우를 만드는 것이 파이프라인
- 모델 평가 결과 모델이 충분한 성능을 발휘하지 못하면?
- 모델평가 이후 하이퍼 파라미터를 진행(튜닝) 한 다음 다시 시도한다.

머신러닝의 파라미터 종류 :

1)모델 파라미터: 머신러닝 알고리즘이 학습,  즉 학습대상

2)하이퍼 파라미터: 사람이 모델에게 넣어주는 알고리즘  > 튜닝대상

### MLlib의 주요 Component들

- DataFrame
- Transformer.transform()
- Data Normalization, Tokenization, Categorical Numeric Encoding, One Hot Encoding 등
- Estimator.fit()
- Evaluator
- BinaryClassificationEvaluator, CrossValidator
- **Pipeline**
    
    ![image.png](todaylearn/Spark%20MLlib/image%203.png)
    

# 파이프라인

## **데이터 준비**

- 입력 데이터는 DataFrame 형태로 사용되며, 각 행은 하나의 데이터 포인트, 각 열은 특성(Feature) 또는 레이블(Label)을 나타내는 구조이다.
- 입력 데이터는 일반적으로 CSV, JSON, Parquet, Avro 등의 형식으로 제공되는 파일이다.
- 학습용과 테스트용 데이터 세트를 분리하여 생성하는 과정이다.

![ 학습/테스트 분할](todaylearn/Spark%20MLlib/image%204.png)

 학습/테스트 분할

## **변환기 Transformaer**

- Feature Transformation을 담당한다. 피처 트랜스폼은 보통 하나 이상의 컬럼을 추가하는 작업이다.
- 원본 DataFrame을 머신러닝이 가능한 새로운 DataFrame으로 변환한다.
- 머신러닝 가능 DataFrame은 숫자데이터로 구성된다.
- 문자데이터는 처리가 불가능하므로 전처리를 수행한다.
- 모든 Transformer는 transform() 함수를 가지고 있다.
- 변환기는 데이터를 변환하는 정적인 연산을 수행한다.
- 입력 데이터(DataFrame) → 변환된 데이터(DataFrame).
- 예시 : Data Normalization, Tokenization, Categorical Numeric Encoding, One Hot Encoding 등

- 원핫인코딩(One hot encoding) > 카테고리 형식의 데이터를 피처로 넣을 때 입력한 데이터 중 한 개만 1, 나머지는 0으로 설정한다.
- **StStringIndexer**: 범주형 데이터를 숫자로 변환.
    - **StringIndexer의 주요 역할**
        
        1.**범주형 데이터의 수치화**: StringIndexer는 범주형 변수의 각 고유 값에 숫자를 할당하여 모델링 가능한 수치형 데이터로 변환하는 도구이다. 예를 들어, "red", "blue", "green"과 같은 색상 범주가 있다면, 이들 각각에 0, 1, 2와 같은 숫자를 할당할 수 있다.
        
        2.**빈도 기반 인덱싱**: 기본적으로 StringIndexer는 가장 빈도가 높은 범주부터 시작하여 순서대로 낮은 숫자를 할당하는 방식이다. 이는 가장 일반적인 범주가 낮은 숫자 인덱스를 받아 일부 모델에서 더 빠른 학습과 예측이 가능하다.
        
        3.**모델 입력 준비**: 많은 머신러닝 알고리즘들, 특히 선형 모델과 트리 기반 모델들은 입력 데이터로 숫자형 벡터를 요구한다. StringIndexer를 사용하여 범주형 데이터를 수치형 데이터로 변환함으로써 이러한 요구사항을 충족시킬 수 있다.
        
- **VectorAssembler**: 여러 특성을 단일 벡터 컬럼으로 결합.
- **StandardScaler**: 데이터를 정규화하여 스케일 차이를 제거.

## **추정기 Estimator**

- 모델의 학습을 담당하는 역할이다.
- 모든 Estimator는 fit() 함수를 가지고 있다.
- fit()은 DataFrame을 입력 받아 학습한 다음, Model을 반환한다.
- 입력 데이터(DataFrame) → 학습된 모델(Transformer)로 변환된다.
- 모델은 하나의 Transformer라고 볼 수 있다.

## **평가 Evaluator**

- 모델의 성능을 평가 방식(metric)을 기반으로 평가하는 도구이다.
- RMSE, MSE, MAE, Cross Entropy Error 등의 평가 지표가 있다.
- 여러 모델의 성능을 평가하여 최적의 모델을 선택함으로써 모델 튜닝을 자동화할 수 있다.
- BinaryClassificationEvaluator, CrossValidator 등의 평가기가 있다.

## **파이프라인 (Pipeline) 객체**

- 여러 변환기와 추정기를 연결하여 머신러닝 워크플로우를 정의하는 도구이다.
- 여러 stage를 담고 있으며, 머신러닝의 전체적인 워크플로우를 연결시켜 주는 역할이다.
- **Pipeline** 객체는 입력 데이터(DataFrame)를 받아 변환 및 학습을 자동으로 처리하는 객체이다.
- 각 stage마다 담당하는 과정이 있으며, stage는 로딩, 전처리, 학습, 모델 평가 등 각각의 과정들을 담당하는 요소이다.
- 파이프라인을 이용해 Evaluator까지 거치면 모델이 완성되는 구조이다.

![image.png](todaylearn/Spark%20MLlib/image%205.png)

# MLlib-ALS 기반 협업필터링시스템

## 추천 알고리즘의 주요 유형

추천 시스템의 주요 유형은 크게 협업 필터링과 콘텐츠 기반 필터링이다.

### 협업 필터링

협업필터링은 사용자들의 기호(taste) 정보를 기반으로 관심사를 자동으로 예측하는 방식이다. 비슷한 취향을 가진 사용자들에게 서로 아직 구매하지 않은 상품을 교차 추천하거나, 사용자의 취향과 생활 패턴에 맞는 상품을 추천하는 서비스에 활용된다.

- 사용자와 아이템 간의 **과거 상호작용(평점, 클릭 등)**만 사용
- **아이템의 내용 정보 필요 없음**

<aside>
💡

“**비슷한 사용자나 아이템의 행동을 이용해서 추천**하는 방식”

</aside>

| 방법 | 설명 | 대표 알고리즘 |
| --- | --- | --- |
| **메모리 기반** | 유사 사용자/아이템 찾아서 직접 추천 | User-based, Item-based CF |
| **모델 기반** | 행렬 분해 등 수학 모델로 예측 | ✅ **ALS**, SVD, NMF |

#### 협업필터링의 작동 원리

- 사용자 A는 일부 데이터만 있고, 사용자 B는 더 많은 데이터가 있다. 그러나 두 사용자의 영화 평가 패턴이 유사하다.
- A와 B가 동일한 영화에 비슷한 평점을 준다면, 두 사람의 영화 취향은 유사하다고 볼 수 있다. 따라서 B가 '위쳐'에 높은 평점을 주었다면, A에게도 '위쳐'를 추천하는 것이 협업 필터링의 기본 원리이다.

![image.png](todaylearn/MLlib-ALS%20기반%20협업필터링시스템/image%201.png)

실제 데이터에서는 영화와 사용자 수가 매우 많다.

- 아래 표와 같이 빈 칸이 많은 것은 한 사용자가 모든 영화를 볼 수 없기 때문이다. 추천 시스템은 이러한 미시청 영화의 예상 평점을 예측하는 것이다.
- 예측된 평점을 높은 순으로 정렬하여 사용자에게 제공하는 것이 추천 시스템의 기본 원리이다.

![image.png](todaylearn/MLlib-ALS%20기반%20협업필터링시스템/9b7f26f7-7099-4c99-8fa6-095c8de3c55b.png)

![image.png](todaylearn/MLlib-ALS%20기반%20협업필터링시스템/0915c94b-828c-404c-acab-512bae15a8c2.png)

### **콘텐츠 기반 필터링 (Content-Based Filtering)**

콘텐츠 기반 필터링은 사용자가 과거에 좋아했거나 관심을 보인 아이템의 특성을 분석하여 유사한 특성을 가진 다른 아이템을 추천하는 방식이다. 이 방법은 아이템 자체의 속성(예: 영화 장르, 배우, 감독, 줄거리 등)을 기반으로 추천을 제공한다. 따라서 새로운 아이템이 시스템에 추가되었을 때도 그 아이템의 특성만 있으면 즉시 추천이 가능하다.

- 아이템의 **속성 정보(장르, 키워드, 설명 등)** 필요
- 사용자와 아이템을 개별적으로 매칭
- 협업 필터링보다 **Cold Start에 강함**

- 유저 A가 “액션 영화”를 좋아했으면, 또 다른 액션 영화를 추천
- TF-IDF, Word2Vec, Cosine Similarity 활용

### **하이브리드 필터링 (Hybrid Filtering)**

> 협업 + 콘텐츠 기반을 결합한 방식
> 

- 예측 결과를 **가중 평균하거나**
- ALS 벡터와 콘텐츠 벡터를 **합쳐서 딥러닝 모델에 입력**
- 넷플릭스: 사용자 평점 + 영화 장르/태그 + 시청 로그 결합

## ALS 알고리즘

ALS(Alternating Least Squares)는 행렬 분해(Matrix Factorization) 기반의 협업 필터링 알고리즘이다.

<aside>
💡

ALS는 실제 평점 행렬(R)과 가장 비슷하게 되도록 두 개의 **잠재 요인 행렬(Latent Factor Matrices)**을 **분해하고 다시 곱**해서 예측하는 방식

</aside>

사용자-아이템 평점 행렬을 두 개의 저차원 행렬(사용자 잠재 요인과 아이템 잠재 요인)으로 분해한다. 이 알고리즘은 두 행렬을 번갈아가며 최적화하는 방식으로 작동한다.

- ALS 알고리즘
    
    실제 계산에서는 특정 차원(latent factors)의 행렬로 변환하여 계산한다. 
    
    예를 들어 m×n 크기의 Rating Matrix는 m×k 크기의 User Matrix와 k×n 크기의 Item Matrix로 분해된다. 여기서 k는 모델의 복잡성을 결정하는 하이퍼파라미터이다.
    
    이때 반복을 통해 최적화가 진행될수록 예측값과 실제값 사이의 오차는 점점 줄어들게 된다. 즉, 잠재 요인 행렬들은 실제 평점 데이터를 더 잘 표현할 수 있게 된다. 학습이 완료된 후에는 기존에 평가하지 않은 항목에 대한 예측이 가능해진다.
    
    결과적으로 User Matrix와 Item Matrix의 곱으로 Rating Matrix와 최대한 가까운 Matrix가 생성되며, 이 Matrix는 빈 칸에 있는 값들이 모두 채워진 형태이다.
    
    이는 특정 사용자의 영화 목록 또는 특정 영화를 선택한 사용자 목록을 행렬곱하여 Rating Matrix를 만드는 방식으로, 열벡터와 행벡터의 행렬곱으로 완성된다.
    
- ALS의 작동 원리
    
    1.**초기화**: 사용자 행렬(U)와 아이템 행렬(I)을 무작위 값으로 초기화합니다. 이 두 행렬의 곱은 원래의 평가 행렬을 근사하게 됩니다.
    
    2.**교대 최소 제곱법**: ALS는 교대로 한 행렬을 고정시키고 다른 행렬을 최적화하는 방식으로 작동합니다. 예를 들어, 사용자 행렬을 고정시킨 상태에서 아이템 행렬을 최적화하고, 이후 아이템 행렬을 고정시킨 상태에서 사용자 행렬을 최적화합니다. 이 과정은 평가 행렬과 사용자 행렬 및 아이템 행렬의 곱 사이의 차이(오차)를 최소화합니다.
    
    3.**정규화**: 과적합(overfitting)을 방지하기 위해 정규화 항을 추가할 수 있습니다. 이는 모델이 훈련 데이터에 너무 정확히 맞춰지는 것을 방지하여, 일반화 성능을 향상시킵니다.
    
    4.**반복**: 위의 과정을 반복하여, 모델의 예측 성능이 더 이상 개선되지 않을 때까지 진행합니다.
    
    - 사용자 행렬과 아이템 행렬의 최적화가 완료되면, 이 두 행렬의 곱을 사용하여 비어 있는 평점들을 예측합니다. 이 곱셈 결과는 완성된 평점 행렬로, 모든 사용자와 모든 아이템 간의 평점 예측 값을 포함합니다.
    - 이 예측 행렬을 바탕으로 사용자별로 가장 높은 평점을 받을 것으로 예측되는 아이템을 추천합니다.
    
    ![image.png](todaylearn/MLlib-ALS%20기반%20협업필터링시스템/image.png)
    
    1. User Matrix의 값과 Item Matrix의 값은 랜덤하게 채워집니다.
    2. Item 행렬을 고정 시키고 User 행렬을 최적화 합니다.
    3. User Matrix의 값과 Item Matrix의 값을 곱했을 때 Rating Matrix에 있는 값과 비슷하게 최적화가 됩니다.
    4. User 행렬을 고정 시키고 Item 행렬을 최적화 합니다.
    

ALS는 넷플릭스나 아마존 같은 대형 온라인 서비스에서 개인화된 추천을 제공하는 강력한 도구이다. 이 알고리즘은 사용자의 과거 행동 패턴을 분석하여 사용자가 좋아할 만한 영화나 TV 프로그램을 정확하게 예측하거나, 개인의 쇼핑 선호도와 구매 이력에 기반하여 가장 적합한 제품을 추천하는 데 폭넓게 활용된다.

대규모 사용자 데이터베이스에서도 효율적으로 작동하며, 실시간으로 개인화된 추천을 생성할 수 있는 능력 덕분에 현대 전자상거래 및 스트리밍 플랫폼에서 필수적인 요소로 쓰이고  있다.

## Spark MLlib의 ALS(Alternating Least Squares)

**`pyspark.ml.recommendation.ALS`** 클래스로 ALS를 구현하고 있다.  

스파크의 이 클래스는 대용량 데이터에서도 확장성이 뛰어나 실시간 추천 시스템에 적합하다.

<aside>
💡

MLlib의 ALS (Alaternating Least Squares) 알고리즘 기반 구현

</aside>

ALS는 사용자-아이템 행렬의 누락된 값을 예측해 아직 평가되지 않은 아이템에 대한 사용자 선호도를 추정한다. 이 방식은 특히 대규모 데이터셋에서 Spark의 분산 컴퓨팅 능력을 활용해 높은 성능을 보인다.

## ALS 구축 순서

```bash
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# 1. 세션 생성
spark = SparkSession.builder.appName("ALS-example").getOrCreate()

# 2. 데이터 로드 (예: userId, itemId, rating)
ratings = spark.read.csv("ratings.csv", header=True, inferSchema=True)

# 3. 훈련/테스트 분할
(training, test) = ratings.randomSplit([0.8, 0.2])

# 4. ALS 모델 정의
als = ALS(
    userCol="userId",
    itemCol="itemId",
    ratingCol="rating",
    rank=10,
    maxIter=10,
    regParam=0.1,
    coldStartStrategy="drop"  # NaN 방지
)

# 5. 모델 훈련
model = als.fit(training)

# 6. 예측
predictions = model.transform(test)

# 7. 평가 (RMSE)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Test RMSE = {rmse:.4f}")

```