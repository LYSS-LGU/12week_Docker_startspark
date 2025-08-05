# C:\githome\12week_Docker_startspark\untitled.py
from pyspark.sql import SparkSession
from pyspark.sql import Row

print("🚀 Spark 세션 생성 중...")
spark = SparkSession.builder \
    .appName("WriteToMySQL") \
    .getOrCreate()

print("✅ Spark 세션 생성 완료")

# 예시 데이터프레임 생성
data = [Row(name="David"), Row(name="Emma"), Row(name="Frank")]
df = spark.createDataFrame(data)

print("📦 데이터프레임 생성:")
df.show()

# MySQL로 저장
print("📤 MySQL에 데이터 쓰는 중...")
df.write.format("jdbc").options(
    url="jdbc:mysql://mysql:3306/mydb",
    driver="com.mysql.cj.jdbc.Driver",
    dbtable="test_table",
    user="root",
    password="root"
).mode("append").save()

print("✅ MySQL 저장 완료")