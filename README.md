# 스파크 RDD 핵심 요약 정리

---

### **1. RDD란 무엇인가요? (RDD의 특징)**

*   **RDD (Resilient Distributed Dataset, 탄력적 분산 데이터셋)**: 스파크의 핵심 데이터 개념입니다. 여러 컴퓨터(노드)에 **분산 저장**된 데이터 모음이라고 생각하시면 됩니다.
*   **주요 특징:**
    *   **분산 및 병렬 처리**: 데이터가 여러 조각(파티션)으로 나뉘어 여러 컴퓨터에서 **병렬로 처리**되므로 매우 빠릅니다.
    *   **탄력성 (Resilience)**: 특정 컴퓨터에 문제가 생겨도 다른 컴퓨터의 복제본으로 데이터를 **자동으로 복구**할 수 있어 안정적입니다.
    *   **불변성 (Immutability)**: 한 번 만들어진 RDD는 **변경할 수 없습니다**. 데이터를 바꾸려면 기존 RDD를 기반으로 **새로운 RDD를 만드는 '변환(Transformation)'** 과정을 거칩니다.
    *   **지연 연산 (Lazy Evaluation)**: `map` 같은 변환 작업은 즉시 실행되지 않고, 결과를 실제로 요청하는 **'액션(Action)'이 호출될 때 한꺼번에 실행**됩니다. 이는 불필요한 계산을 줄여 최적화에 유리합니다.

---

### **2. RDD 작업 흐름 (RDD 워크플로우)**

RDD 작업은 항상 **생성 → 변환 → 액션** 3단계로 이루어집니다.

1.  **생성 (Creation)**:
    *   Python 리스트 같은 기존 데이터로 만들거나 (`sc.parallelize()`)
    *   HDFS, 로컬 파일 등 외부 저장소에서 데이터를 읽어와 만듭니다 (`sc.textFile()`).

2.  **변환 (Transformation)**:
    *   기존 RDD를 가공해서 **새로운 RDD를 만드는** 모든 작업입니다.
    *   예: `map`, `filter`, `join`, `groupByKey`

3.  **액션 (Action)**:
    *   지금까지 계획된 모든 변환 작업을 **실제로 실행**시키고, 결과를 반환받거나 파일로 저장하는 작업입니다.
    *   예: `count`, `collect`, `take`, `saveAsTextFile`

---

### **3. 가장 중요한 연산: 트랜스포메이션과 액션**

*   **트랜스포메이션 (Transformation, 변환)**:
    *   **Narrow (좁은) 변환**:
        *   1:1 변환. 다른 파티션의 데이터 교환(셔플)이 필요 없습니다. **매우 빠릅니다.**
        *   예: `map` (각 요소를 변환), `filter` (조건에 맞는 요소만 남김)
    *   **Wide (넓은) 변환**:
        *   여러 파티션의 데이터를 섞어야 하는(셔플 발생) 변환. **비용이 비싸고 느릴 수 있습니다.**
        *   예: `groupByKey` (같은 키끼리 그룹화), `reduceByKey` (키별로 집계), `join` (두 RDD를 키 기준으로 합침), `sortBy` (정렬)
        *   **성능 팁**: `groupByKey()` 후 집계하는 것보다 **`reduceByKey()`**를 쓰는 것이 훨씬 효율적입니다. 셔플 전에 각 노드에서 미리 집계를 하기 때문입니다.

*   **액션 (Action, 실행)**:
    *   `collect()`: 모든 데이터를 드라이버(내 컴퓨터)로 가져옵니다. **데이터가 크면 메모리 부족으로 시스템이 멈출 수 있으니 매우 주의해야 합니다.**
    *   `take(n)`: `n`개의 데이터만 가져옵니다. (`collect`의 안전한 버전)
    *   `count()`: 데이터 개수를 셉니다.
    *   `reduce()`: 모든 데이터를 하나의 값으로 합칩니다. (예: 총합 구하기)
    *   `saveAsTextFile()`: 결과를 파일로 저장합니다.

---

### **4. 그룹화와 집계의 핵심: Key-Value RDD**

*   `(Key, Value)` 형태의 쌍으로 이루어진 RDD입니다. 단어 개수 세기, 장르별 영화 평점 계산 등 **그룹화 및 집계 작업에 특화**되어 있습니다.
*   **만드는 법**: 일반 RDD에 `map`을 사용해 `(키, 값)` 형태로 만들어줍니다.
    *   `rdd.map(lambda x: (x, 1))` → 단어 `x`를 `(x, 1)` 쌍으로 만들어 카운팅을 준비합니다.
*   **주요 연산**:
    *   `reduceByKey(func)`: **(가장 중요)** 같은 키를 가진 값들을 `func` 함수로 합칩니다.
    *   `groupByKey()`: 같은 키를 가진 값들을 리스트로 묶습니다. (메모리 사용량이 클 수 있음)
    *   `mapValues(func)`: 키는 그대로 두고 값에만 함수를 적용합니다.
    *   `sortByKey()`: 키를 기준으로 정렬합니다.
    *   `join()`: 두 개의 Key-Value RDD를 같은 키를 기준으로 합칩니다.

#### **대표 예시: 단어 개수 세기**
```python
# 1. 생성
lines = sc.parallelize(["hello world", "hello spark"])

# 2. 변환
words = lines.flatMap(lambda line: line.split())  # ["hello", "world", "hello", "spark"]
pairs = words.map(lambda word: (word, 1))         # [("hello", 1), ("world", 1), ("hello", 1), ("spark", 1)]
counts = pairs.reduceByKey(lambda a, b: a + b)    # [("hello", 2), ("world", 1), ("spark", 1)]

# 3. 액션
result = counts.collect()
print(result)
```
