import pandas as pd
from statsmodels.tsa.stattools import adfuller


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


def get_number_sales_per_day(customer=None, begin=None, end=None):
    train = get_transaction_train_df()

    if end is not None and begin is not None:
        sales = train[(train["t_dat"] >= begin) & (train["t_dat"] <= end)]
        return sales.groupby(["t_dat"]).t_dat.count()
    else:
        return train.groupby(["t_dat"]).t_dat.count()


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
