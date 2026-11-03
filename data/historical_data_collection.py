import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as web

url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

sp500 = pd.read_csv(url)

print(sp500.head())

print(sp500["GICS Sector"].unique())

print(sp500["GICS Sector"].value_counts())

sectors = sp500["GICS Sector"].unique()

selected_companies = []

for i, sector in enumerate(sectors):
    if i < 7:
        num_companies = 4
    else:
        num_companies = 3

    sector_companies = sp500[sp500["GICS Sector"] == sector]

    selected = sector_companies.sample(
        n=num_companies,
        random_state=42
    )
    selected_companies.append(selected)

    print(sector, selected["Symbol"].tolist())

stock_list = pd.concat(selected_companies, ignore_index=True)

print(len(stock_list))

stock_list = stock_list[["Symbol", "GICS Sector"]]
print(stock_list.head())

tickers = stock_list["Symbol"].tolist()

print(tickers)

test_data = yf.download(
    "FTV",
    start="2014-12-01",
    end="2026-01-01",
    auto_adjust=False
)

print(test_data.head())

print("First date:", test_data.index.min())
print("Last date:", test_data.index.max())
print("Number of observations:", len(test_data))


invalid_tickers = []

for ticker in tickers:
    data = yf.download(
        ticker,
        start="2014-12-01",
        end="2026-01-01",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        print(ticker, "DOWNLOAD FAILED")
        invalid_tickers.append(ticker)

    elif data.index.min() > pd.Timestamp("2015-01-01"):
        print(ticker, "INSUFFICIENT HISTORY")
        invalid_tickers.append(ticker)

    else:
        print(ticker, "OK")


print("Companies to replace:", invalid_tickers)

industrial_candidates = sp500[
    (sp500["GICS Sector"] == "Industrials") &
    (~sp500["Symbol"].isin(tickers))
]

print(industrial_candidates[["Symbol", "GICS Sector"]].head())

replacement_ftv = industrial_candidates.sample(
    n=1,
    random_state=42
)

print(replacement_ftv["Symbol"])

new_ticker = replacement_ftv["Symbol"].iloc[0]

print("Replacement candidate:", new_ticker)

replacement_data = yf.download(
    new_ticker,
    start="2014-12-01",
    end="2026-01-01",
    auto_adjust=False,
    progress=False
)

print("First date:", replacement_data.index.min())
print("Last date:", replacement_data.index.max())
print("Observations:", len(replacement_data))

stock_list["Symbol"] = stock_list["Symbol"].replace("FTV", "GD")
print(stock_list.head())

communication_candidates = sp500[
    (sp500["GICS Sector"] == "Communication Services") &
    (~sp500["Symbol"].isin(tickers)) &
    (~sp500["Symbol"].isin(["GOOG", "APP"]))
]
print(communication_candidates[["Symbol", "GICS Sector"]].head())

replacement_fox = communication_candidates.sample(
    n=1,
    random_state=42
)

new_fox_ticker = replacement_fox["Symbol"].iloc[0]

print("FOX replacement candidate:", new_fox_ticker)

fox_data = yf.download(
    new_fox_ticker,
    start="2014-12-01",
    end="2026-01-01",
    auto_adjust=False,
    progress=False
)

print("First date:", fox_data.index.min())
print("Last date:", fox_data.index.max())
print("Observations:", len(fox_data))

stock_list["Symbol"] = stock_list["Symbol"].replace("FOX", "T")

tickers = stock_list["Symbol"].tolist()

print("Total companies:", len(stock_list))
print("Unique tickers:", stock_list["Symbol"].nunique())
print("Total sectors:", stock_list["GICS Sector"].nunique())
print("Final tickers:", tickers)

stock_list.to_csv("HistoricalData/stock_list.csv", index=False)

all_stock_data = []

for ticker in tickers:
    data = yf.download(
        ticker,
        start="2014-12-01",
        end="2026-01-01",
        auto_adjust=False,
        progress=False
    )
    data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data["Ticker"] = ticker

    all_stock_data.append(data)

print("Number of companies downloaded:", len(all_stock_data))

for ticker, data in zip(tickers, all_stock_data):
    if data.empty:
        print("Missing data for:", ticker)

stock_prices = pd.concat(all_stock_data, ignore_index=True)

print(stock_prices.head())
print("Total rows:", len(stock_prices))
print("Unique companies:", stock_prices["Ticker"].nunique())

stock_prices.to_csv("HistoricalData/stock_prices.csv", index=False)

print("Missing Close prices:", stock_prices["Close"].isna().sum())
print("Missing Adj Close prices:", stock_prices["Adj Close"].isna().sum())
print("Duplicate ticker-date rows:",
      stock_prices.duplicated(subset=["Ticker", "Date"]).sum())

treasury_rates = web.DataReader(
    ["DGS1MO", "DGS3MO", "DGS6MO"],
    "fred",
    start="2016-01-01",
    end="2025-12-31"
)

print(treasury_rates.head())
print(treasury_rates.shape)

treasury_rates = treasury_rates / 100

print(treasury_rates.head())

treasury_rates = treasury_rates.ffill()

print("Missing Treasury rates:")
print(treasury_rates.isna().sum())

treasury_rates = treasury_rates.dropna()

print("Remaining missing values:")
print(treasury_rates.isna().sum())

print("Total Treasury observations:", len(treasury_rates))

treasury_rates.index.name = "Date"

treasury_rates.to_csv("HistoricalData/treasury_rates.csv")

stock_prices = stock_prices.sort_values(["Ticker", "Date"])

stock_prices["log_return"] = (
    np.log(
        stock_prices["Adj Close"] /
        stock_prices.groupby("Ticker")["Adj Close"].shift(1)
    )
)

print(stock_prices[["Date", "Ticker", "Adj Close", "log_return"]].head())

stock_prices["sigma"] = (
    stock_prices.groupby("Ticker")["log_return"]
    .transform(lambda x: x.rolling(window=252, min_periods=252).std())
    * np.sqrt(252)
)

print(stock_prices[["Date", "Ticker", "log_return", "sigma"]].iloc[250:256])

print("Valid sigma observations:", stock_prices["sigma"].notna().sum())

print("Companies with valid sigma:",
      stock_prices.loc[stock_prices["sigma"].notna(), "Ticker"].nunique())

print("Minimum sigma:", stock_prices["sigma"].min())
print("Maximum sigma:", stock_prices["sigma"].max())

treasury_rates = treasury_rates.reset_index()

stock_prices["Date"] = pd.to_datetime(stock_prices["Date"]).astype("datetime64[ns]")
treasury_rates["Date"] = pd.to_datetime(treasury_rates["Date"]).astype("datetime64[ns]")

stock_prices = stock_prices.sort_values("Date")
treasury_rates = treasury_rates.sort_values("Date")

print(treasury_rates.head())

historical_data = pd.merge_asof(
    stock_prices,
    treasury_rates,
    on="Date",
    direction="backward"
)

print(historical_data[
    ["Date", "Ticker", "Close", "sigma", "DGS1MO", "DGS3MO", "DGS6MO"]
].tail())

historical_data = historical_data.dropna(
    subset=["Close", "Adj Close", "sigma", "DGS1MO", "DGS3MO", "DGS6MO"]
)

print("Clean historical rows:", len(historical_data))
print("Companies remaining:", historical_data["Ticker"].nunique())

print("Missing values:")
print(historical_data[
    ["Close", "Adj Close", "sigma", "DGS1MO", "DGS3MO", "DGS6MO"]
].isna().sum())

historical_data.to_csv("data/historical_data.csv", index=False)

sampled_groups = []

for ticker, group in historical_data.groupby("Ticker"):
    group = group.sort_values("Date")

    indices = np.linspace(0, len(group) - 1, 139, dtype=int)

    sampled_groups.append(group.iloc[indices])

sampled_data = pd.concat(sampled_groups, ignore_index=True)

print("Sampled observations:", len(sampled_data))
print("Companies sampled:", sampled_data["Ticker"].nunique())

option_types = ["call", "put"]
strike_ratios = [0.90, 1.00, 1.10]
maturities = [30, 90, 180]

scenarios = []

for _, row in sampled_data.iterrows():
    S = row["Close"]
    sigma = row["sigma"]
    ticker = row["Ticker"]
    date = row["Date"]


    for option_type in option_types:
        for strike_ratio in strike_ratios:
            for days in maturities:
                K = S * strike_ratio
                T = days / 365

                if days == 30:
                    r = row["DGS1MO"]
                elif days == 90:
                    r = row["DGS3MO"]
                else:
                    r = row["DGS6MO"]

                scenario = {
                    "scenario_id": len(scenarios) + 1,
                    "ticker": ticker,
                    "date": date.strftime("%Y-%m-%d"),
                    "S": S,
                    "K": K,
                    "T": T,
                    "sigma": sigma,
                    "r": r,
                    "option_type": option_type
                }

                scenarios.append(scenario)

print("Total scenarios generated:", len(scenarios))
print("First scenario:", scenarios[0])

pricing_scenarios = pd.DataFrame(scenarios)

print(pricing_scenarios.head())
print(pricing_scenarios.shape)
print(pricing_scenarios.columns.tolist())

print("Missing values:")
print(pricing_scenarios.isna().sum())

print("Duplicate scenario IDs:",
      pricing_scenarios["scenario_id"].duplicated().sum())

print("Invalid positive values:",
      (pricing_scenarios[["S", "K", "T", "sigma"]] <= 0).sum())

print("Option types:",
      pricing_scenarios["option_type"].unique())

print("Unique companies:",
      pricing_scenarios["ticker"].nunique())

pricing_scenarios.to_csv("data/pricing_scenarios.csv", index=False)

pricing_scenarios.head(20).to_csv("data/pricing_scenarios_sample.csv", index=False)