from preparation.utils import spark_daily_sales

df = spark_daily_sales(begin="2018-09-24", end="2020-09-01")

print(df.head())
