import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName("kaggle_hm_app").getOrCreate()
transactions_train_df = spark.read.option("header", True).csv(
    "./00_data/transactions_train.csv"
)
articles_df = spark.read.option("header", True).csv("./00_data/articles.csv")
customers_df = spark.read.option("header", True).csv("./00_data/customers.csv")


def spark_daily_sales(customer_id=None, begin=None, end=None):
    if all(v is None for v in [customer_id, begin, end]):
        print("All the dataset will be returned.")
        return (
            transactions_train_df.groupby(["t_dat", "customer_id"]).count().toPandas()
        )

    if (customer_id is not None) & (begin is None) & (end is None):
        print(f"All the dataset will be returned for customer : {customer_id}")
        return (
            transactions_train_df.filter(
                transactions_train_df["customer_id"] == customer_id
            )
            .groupBy(["t_dat", "customer_id"])
            .count()
            .toPandas()
        )

    if (customer_id == "All") & (begin is not None) & (end is not None):
        print(f"Return all sales between {begin} and {end}")
        return (
            transactions_train_df.filter(
                (transactions_train_df["t_dat"] >= begin)
                & (transactions_train_df["t_dat"] <= end)
            )
            .groupBy(["t_dat", "customer_id"])
            .count()
            .toPandas()
        )

    if (customer_id is None) & (begin is not None) & (end is not None):
        print(f"Return all sales between {begin} and {end}")
        return (
            transactions_train_df.filter(
                (transactions_train_df["t_dat"] >= begin)
                & (transactions_train_df["t_dat"] <= end)
            )
            .groupBy(["t_dat"])
            .count()
            .toPandas()
        )

    if all(v is not None for v in [customer_id, begin, end]):
        print(f"All the dataset will be returned for customer : {customer_id}")
        print(f"Between {begin} and {end}")
        return transactions_train_df.filter(
            transactions_train_df["customer_id"]
            == customer_id & transactions_train_df["t_dat"]
            >= begin & transactions_train_df["t_dat"]
            <= end
        ).toPandas()


def get_transaction_train_df():
    transaction_train_path = "00_data/transactions_train.csv"
    transaction_train_df = pd.read_csv(
        transaction_train_path,
        index_col=["customer_id", "article_id"],
        parse_dates=["t_dat"],
    )

    return transaction_train_df


def get_number_articles_sold(customer, period_begin, period_end):
    train = get_transaction_train_df()

    return train


def get_number_sales_per_day(
    transaction_train_df=None, customer=None, begin=None, end=None
):
    if transaction_train_df is None:
        transaction_train_df = get_transaction_train_df()

    if end is not None and begin is not None:
        sales = transaction_train_df[
            (transaction_train_df["t_dat"] >= begin)
            & (transaction_train_df["t_dat"] <= end)
        ]
        return sales.groupby(["t_dat"]).t_dat.count()
    else:
        return transaction_train_df.groupby(["t_dat"]).t_dat.count()


def get_number_sales_per_month(
    transaction_train_df=None, customer=None, begin=None, end=None
):
    if transaction_train_df is None:
        transaction_train_df = get_transaction_train_df()

    add_year_month_column(transaction_train_df, "t_dat")

    return transaction_train_df.groupby(["year-month"]).t_dat.count()


def add_year_month_column(df, date_column):
    df["year_month"] = df[date_column].apply(
        lambda x: str(x.year) + "-" + str(x.month).zfill(2)
    )

    return df


def adf_test(series, title=""):
    print("Dicker-Fuller Test: ")
    result = adfuller(series.dropna(), autolag="AIC")

    labels = ["ADF test", "p-value", "# lags used", "# observations"]
    out = pd.Series(result[0:4], index=labels)

    for key, val in result[4].items():
        out[f"critical value ({key})"] = val

    print(out.to_string())
    print("*" * 100)
    if result[1] <= 0.05:
        print("Data has no unit root and is stationnary")
    else:
        print("Data has unit root and is non-stationnary")
