# %%
# C:\githome\12week_Docker_startspark\spark_miniproject\pet_supplies_analysis_v3.py
# -*- coding: utf-8 -*-
"""
# AliExpress 반려동물 용품 데이터 분석 (노트북 버전)

**분석 목표:**
1.  로그 스케일을 활용하여 유의미한 판매량 차이를 시각화하고 비즈니스 인사이트를 도출한다.
2.  "평점" 데이터의 신뢰도와 영향력에 대한 가설을 검증한다.
"""

# %%
# ## 1. 환경 설정 및 SparkSession 생성
# 이 셀은 분석에 필요한 모든 도구를 불러옵니다.
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import IntegerType
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# %%
# --- 폰트 설정 ---
# Docker 컨테이너에 설치된 나눔고딕 폰트를 사용하도록 설정합니다.
# 이 코드는 폰트가 있을 때만 적용되므로, 오류 없이 안전하게 실행됩니다.
try:
    if any("NanumGothic" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = 'NanumGothic'
    else:
        print("나눔고딕 폰트를 찾을 수 없습니다. 기본 폰트로 실행됩니다.")
except Exception:
    print("폰트 설정 중 오류가 발생했습니다.")
plt.rcParams['axes.unicode_minus'] = False
# --- 폰트 설정 끝 ---

# %%
# SparkSession을 생성합니다. 모든 Spark 애플리케이션의 시작점입니다.
spark = SparkSession.builder.appName("PetSuppliesAnalysisNotebook").getOrCreate()
print("Spark Session이 성공적으로 생성되었습니다.")


# %%
# C:/githome/12week_Docker_startspark/spark_miniproject/pet_supplies_analysis_v3.ipynb
# ## 1. 환경 설정 및 SparkSession 생성
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import IntegerType
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# --- 폰트 설정 ---
try:
    if any("NanumGothic" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.family'] = 'NanumGothic'
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False
# --- 폰트 설정 끝 ---

spark = SparkSession.builder.appName("PetSuppliesInteractiveAnalysis").getOrCreate()
print("✅ Spark Session이 성공적으로 생성되었습니다.")

# %%
# ## 2. 원본 데이터 로드 및 확인
file_path = 'data/aliexpress_pet_supplies.csv'
df = spark.read.csv(file_path, header=True, inferSchema=False, multiLine=True, escape="\"")
df.show(3)


# %%
# ## 3. 데이터 정제 및 전처리
def clean_trade_amount(amount_str):
    if amount_str is None: return 0
    try:
        cleaned_str = amount_str.replace(',', '')
        numbers = re.findall(r'\d+', cleaned_str)
        return int(numbers[0]) if numbers else 0
    except (ValueError, TypeError): return 0

clean_trade_amount_udf = udf(clean_trade_amount, IntegerType())

cleaned_df = df.withColumn("sales", clean_trade_amount_udf(col("tradeAmount")))\
               .withColumn("averageStar", col("averageStar").cast('float'))\
               .withColumn("wishedCount", col("wishedCount").cast('int'))

# 여기서 final_df가 생성됩니다!
final_df = cleaned_df.select("title", "averageStar", "sales", "wishedCount")

print("✅ 데이터 정제 완료. final_df가 준비되었습니다.")
final_df.show(3)

# %%
# C:/githome/12week_Docker_startspark/spark_miniproject/pet_supplies_analysis_v3.ipynb
# ## 5. '찜하기 수' 기준 Top 10 분석 및 최종 시각화 (범례 추가)
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns

# 1. '찜하기 수(wishedCount)'가 높은 순서대로 진짜 인기 상품 Top 10을 추출합니다.
true_top_10_df = final_df.orderBy(col("wishedCount").desc()).limit(10)

# 2. 시각화를 위해 Pandas DataFrame으로 변환하고 순위를 매깁니다.
top_10_wished = true_top_10_df.toPandas()
# 10위부터 1위 순으로 차트를 그리기 위해 데이터 순서를 뒤집습니다.
top_10_wished = top_10_wished.iloc[::-1].reset_index(drop=True)
top_10_wished['rank'] = top_10_wished.index + 1

print("--- '찜하기 수' 기준 진짜 Top 10 ---")
display(top_10_wished)


# 3. 최종 차트 그리기 (선/막대 조합, 파스텔톤, 두 개의 범례)
legend_labels = [f"#{row['rank']}: {row['title'][:50]}..." for index, row in top_10_wished.iterrows()]

# fig와 ax1은 차트의 전체 도화지와 첫 번째 Y축(왼쪽)을 의미합니다.
fig, ax1 = plt.subplots(figsize=(16, 10))
plt.title('가장 많이 찜한 상품 Top 10과 그 평점', fontsize=22, pad=20, weight='bold')

# '찜하기 수'를 나타내는 막대그래프 (왼쪽 Y축 사용)
plot = sns.barplot(x='rank', y='wishedCount', data=top_10_wished, ax=ax1, palette='pastel', hue='rank', legend=False)
ax1.set_xlabel('찜하기 순위', fontsize=16)
ax1.set_ylabel('찜하기 수 (개)', fontsize=16, color='skyblue')
ax1.tick_params(axis='y', labelsize=12)
# X축 눈금을 순위에 맞춥니다.
ax1.set_xticks(range(len(top_10_wished['rank'])))
ax1.set_xticklabels(top_10_wished['rank'], fontsize=12)

# 막대 위에 '찜하기 수' 텍스트 추가
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=11, color='gray', xytext=(0, 9),
                 textcoords='offset points')

# '평점'을 나타내는 선그래프를 위한 보조 Y축(오른쪽)을 만듭니다.
ax2 = ax1.twinx()
# --- 핵심 수정: label='평균 평점'이 범례에 표시될 내용입니다 ---
ax2.plot(range(len(top_10_wished['rank'])), top_10_wished['averageStar'], color='salmon', marker='o', linestyle='--', linewidth=2.5, markersize=8, label='평균 평점')
ax2.set_ylabel('평균 평점', fontsize=16, color='salmon')
ax2.set_ylim(4, 5) # 평점 범위를 4~5점으로 설정하여 변화를 잘 보이게 합니다.
ax2.tick_params(axis='y', labelsize=12)

# --- 핵심 수정: 선 그래프에 대한 범례를 차트 안쪽 우측 상단에 추가합니다 ---
ax2.legend(loc='upper right', fontsize=12)

# 상품명에 대한 범례를 차트 바깥쪽 오른쪽에 예쁘게 표시합니다.
product_legend = ax1.legend(handles=plot.patches, labels=legend_labels, title='상품명 (Top 10)',
                            bbox_to_anchor=(1.1, 1), loc='upper left', borderaxespad=0., fontsize=12)
plt.setp(product_legend.get_title(), fontsize=14, weight='bold')

# 범례가 잘리지 않도록 레이아웃을 조정하고 저장합니다.
fig.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig("top_10_wished_final_with_legend.png")
print("\n✅ 'top_10_wished_final_with_legend.png' 파일로 최종 차트를 저장했습니다.")
plt.show()

# %%
# ## 4. '찜하기 수'로 진짜 Top 10 순위 결정 및 최종 시각화
from pyspark.sql.functions import col

# 1. 판매량(sales)으로 먼저 정렬하고, 판매량이 같으면(10000) 찜하기 수(wishedCount)로 다시 정렬합니다.
# 이것이 진짜 인기 순위를 찾는 핵심입니다!
true_top_10_df = final_df.orderBy(col("sales").desc(), col("wishedCount").desc()).limit(10)

# 2. 시각화를 위해 Pandas DataFrame으로 변환합니다.
top_10_sold = true_top_10_df.toPandas()
top_10_sold = top_10_sold.iloc[::-1].reset_index(drop=True)
top_10_sold['rank'] = top_10_sold.index + 1

print("--- '찜하기 수'로 다시 정렬한 진짜 Top 10 ---")
# display()는 Jupyter Notebook에서 표를 예쁘게 보여주는 명령어입니다.
display(top_10_sold)


# 3. 최종 차트 그리기 (FutureWarning 해결 및 로그 스케일 적용)
legend_labels = [f"#{row['rank']}: {row['title'][:50]}..." for index, row in top_10_sold.iterrows()]

plt.figure(figsize=(14, 10))

# FutureWarning 해결: y축 변수인 'rank'를 hue에 지정하고 legend=False를 추가합니다.
plot = sns.barplot(x='sales', y='rank', data=top_10_sold, palette='magma', orient='h', hue='rank', legend=False)

# X축을 로그 스케일로 변경하여 값의 차이를 명확하게 보여줍니다.
plot.set_xscale("log")

plt.title('가장 많이 팔린 반려동물 용품 Top 10 (판매량: 로그 스케일)', fontsize=20, pad=20, weight='bold')
plt.xlabel('판매량 (개) - 로그 스케일', fontsize=14)
plt.ylabel('순위', fontsize=14)
plt.yticks([])

# 막대 위에 실제 판매량 텍스트 추가
for index, row in top_10_sold.iterrows():
    plot.text(row.sales, index, f' {row.sales:,}개', color='black', ha="left", va="center")

legend = plt.legend(handles=plot.patches, labels=legend_labels, title='상품명 (Top 10)',
                    bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=11)
plt.setp(legend.get_title(), fontsize=13, weight='bold')

plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig("top_10_sold_final.png")
print("\n✅ 'top_10_sold_final.png' 파일로 최종 차트를 저장했습니다.")
plt.show()

# %%
# ## 6. Spark SQL을 사용한 원본 데이터 전체 확인

# 1. 원본 CSV를 읽어온 Spark DataFrame(df)을 'raw_pet_supplies'라는 SQL 테이블로 등록합니다.
# (이전에 정제한 final_df가 아닌, 원본 df를 사용합니다.)
df.createOrReplaceTempView("raw_pet_supplies")

print("✅ 원본 데이터가 'raw_pet_supplies' SQL 테이블로 성공적으로 등록되었습니다.")


# 2. spark.sql() 함수 안에 'SELECT *' 쿼리를 작성하여 모든 데이터를 조회합니다.
sql_query = """
    SELECT
        *
    FROM
        raw_pet_supplies
"""

# 3. SQL 쿼리를 실행하고, 그 결과를 새로운 Spark DataFrame에 저장합니다.
all_data_sql_df = spark.sql(sql_query)

print("\n--- SQL 쿼리로 조회한 전체 데이터 ---")
# Jupyter Notebook에서 모든 내용을 편하게 보기 위해 Pandas로 변환합니다.
all_data_pd = all_data_sql_df.toPandas()

# Jupyter Notebook에서 모든 행과 열이 보이도록 출력 옵션을 설정합니다.
pd.set_option('display.max_rows', 100) # 너무 길어지지 않게 100개로 제한 (None으로 바꾸면 전체)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# display()는 Jupyter Notebook에서 표를 예쁘게 보여주는 명령어입니다.
display(all_data_pd)

# %%
# ## 원본 CSV 데이터 전체 확인 (Pandas 사용)
import pandas as pd

# Pandas를 사용해 CSV 파일을 직접 읽어옵니다.
# Spark와는 별개로, 탐색을 위해 Pandas로만 데이터를 불러오는 것입니다.
file_path = 'data/aliexpress_pet_supplies.csv'

try:
    # Pandas DataFrame으로 csv_df 변수에 데이터를 저장합니다.
    csv_df = pd.read_csv(file_path)

    # Jupyter Notebook에서 모든 행과 열이 보이도록 출력 옵션을 설정합니다.
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    print(f"--- '{file_path}' 파일 전체 내용 ---")
    # display()는 Jupyter Notebook에서 표를 예쁘게 보여주는 명령어입니다.
    display(csv_df)

except FileNotFoundError:
    print(f"'{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")

# %%
# ## 데이터 전체 요약 및 최대 판매량 확인
from pyspark.sql.functions import col

# 이전에 실행했던 3번 셀(데이터 정제)의 코드를 다시 한번 실행하여 final_df를 준비합니다.
# 이미 실행했다면 이 부분은 건너뛰어도 되지만, 안전을 위해 포함합니다.
def clean_trade_amount(amount_str):
    if amount_str is None: return 0
    try:
        cleaned_str = amount_str.replace(',', '')
        numbers = re.findall(r'\d+', cleaned_str)
        return int(numbers[0]) if numbers else 0
    except (ValueError, TypeError): return 0

clean_trade_amount_udf = udf(clean_trade_amount, IntegerType())
cleaned_df = df.withColumn("sales", clean_trade_amount_udf(col("tradeAmount")))\
               .withColumn("averageStar", col("averageStar").cast('float'))\
               .withColumn("wishedCount", col("wishedCount").cast('int'))
final_df = cleaned_df.select("title", "averageStar", "sales", "wishedCount")


# describe() 함수로 모든 숫자 컬럼의 통계 요약을 확인합니다.
print("--- 숫자 데이터 전체 요약 ---")
# .show()를 실행하면 바로 아래에 결과 표가 나타납니다.
final_df.describe().show()


# 판매량(sales)이 가장 높은 순서대로 상위 10개 상품을 확인합니다.
print("\n--- 판매량 Top 10 상품 ---")
final_df.orderBy(col("sales").desc()).show(10, truncate=False)


# %%
# ## 3. 데이터 정제 및 전처리
# '1,000+ sold'와 같이 문자와 숫자가 섞인 판매량(tradeAmount) 컬럼을
# 분석 가능한 순수 숫자로 바꿔주는 함수를 정의합니다.
def clean_trade_amount(amount_str):
    if amount_str is None: return 0
    try:
        cleaned_str = amount_str.replace(',', '')
        numbers = re.findall(r'\d+', cleaned_str)
        return int(numbers[0]) if numbers else 0
    except (ValueError, TypeError): return 0

# %%
# 위에서 만든 함수를 Spark에서 사용할 수 있도록 UDF(사용자 정의 함수)로 등록합니다.
clean_trade_amount_udf = udf(clean_trade_amount, IntegerType())

# %%
# 필요한 컬럼들의 데이터 타입을 숫자로 변환하고, 'sales' 컬럼을 새로 만듭니다.
cleaned_df = df.withColumn("sales", clean_trade_amount_udf(col("tradeAmount")))\
               .withColumn("averageStar", col("averageStar").cast('float'))\
               .withColumn("wishedCount", col("wishedCount").cast('int'))

# %%
# 분석에 사용할 최종 컬럼들만 선택합니다.
final_df = cleaned_df.select("title", "averageStar", "sales", "wishedCount")

# %%
print("\n데이터 정제 완료. 처리된 데이터 상위 5개:")
final_df.show(5, truncate=False)
print("\n처리된 데이터 스키마 (데이터 타입 확인):")
final_df.printSchema()


# %%
# ## 4. Top 10 판매 상품 분석 (로그 스케일 적용)
# 판매량(sales) 기준으로 상위 10개 상품을 추출하여 시각화하기 좋은 Pandas DataFrame으로 변환합니다.
top_10_sold = final_df.orderBy(col("sales").desc()).limit(10).toPandas()
# 차트를 10위부터 1위 순으로 그리기 위해 데이터 순서를 뒤집습니다.
top_10_sold = top_10_sold.iloc[::-1].reset_index(drop=True)
# 순위 컬럼을 추가합니다.
top_10_sold['rank'] = top_10_sold.index + 1

# %%
# 범례에 표시할 텍스트를 만듭니다. (상품명이 너무 길면 50자까지만 표시)
legend_labels = [f"#{row['rank']}: {row['title'][:50]}..." for index, row in top_10_sold.iterrows()]

# %%
# --- 핵심 수정: X축을 로그 스케일로 변경 ---
# 이 한 줄이 판매량 차이를 명확하게 보여주는 마법입니다.
plot.set_xscale("log")
# --- 핵심 수정 끝 ---

# %%
# --- 막대 위에 실제 판매량 텍스트 추가 ---
for index, row in top_10_sold.iterrows():
    plot.text(row.sales,       # 텍스트를 표시할 x 좌표 (판매량)
              index,           # 텍스트를 표시할 y 좌표 (인덱스)
              f' {row.sales:,}개', # 표시할 텍스트 (세 자리 콤마 추가)
              color='black',   # 글자색
              ha="left",       # 수평 정렬 (왼쪽)
              va="center")     # 수직 정렬 (중앙)
# --- 텍스트 추가 끝 ---

# %%
plt.tight_layout(rect=[0, 0, 0.75, 1])
plt.savefig("top_10_sold_log_scale.png")

# %%
print("\n✅ 'top_10_sold_log_scale.png' 파일로 로그 스케일이 적용된 차트를 저장했습니다.")
# Jupyter Notebook에서는 이 셀을 실행하면 바로 아래에 차트가 보입니다.
plt.show()


# %%
# ## 5. 분석 종료
# SparkSession을 종료하여 모든 리소스를 해제합니다.
spark.stop()
print("\n분석이 완료되었습니다. Spark Session이 종료되었습니다.")

# %%

# %%

# %%

# %%
