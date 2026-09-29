import pandas as pd

def combine_log_returns(stocks):
    """
    Combines log returns of stocks in portfolio into one dataframe.

    Args:
        stocks: list
            List of stock ticker symbol.

    Returns:
        returns_combined: pd.DataFrame
            A DataFrame containing the daily log returns for each stock.
    """

    series = {}
    for stock in stocks: 
        try: 
            df = pd.read_csv(f"regime_portfolio_optimiser/data/{stock}_historical_data_engineered.csv", 
                             parse_dates=["Date"])            
        except Exception as e: 
            print(f"Error fetching data for {stock}: {e}")
            raise
        series[stock] = df.set_index("Date")["Daily Log Return"] 

    returns_combined = pd.concat(series, axis=1, join="inner").sort_index()
    returns_combined = returns_combined.dropna()
    return returns_combined

def normalise_index(df1, df2, method="reindex"): 
    """
    Normalise index of datetime indexes

    Args:
        df1: pd.DataFrame
            Pandas dataframes containing datetime index. 
        df2: pd.DataFrame
            Pandas dataframes containing datetime index. 
        method: str
            Method of normalisation. 
            reindex - Drops differing dates
            ffill   - Forward fills dates (backward fills missing values which lead)

    Returns:
        df1_normalised (pd.DataFrame), df2_normalised (pd.DataFrame): Pandas dataframes with matching indexes
    """
    def row_valid(x):
        m = x.notna()
        return m.all(axis=1) if m.ndim == 2 else m

    # normalise datetime index conventions
    df1_normalised = df1.copy()
    df2_normalised = df2.copy()
    df1_normalised.index = df1.index.tz_localize(None).normalize() 
    df2_normalised.index = df2.index.tz_localize(None).normalize()

    # deal with index mismatches based on method. just realised filling data is probably not viable for returns, probably reindex 
    if (method == "reindex"):
        df1_normalised, df2_normalised = df1_normalised.align(df2_normalised, join="inner", axis=0)
    elif (method == "ffill"):
        max_fill = 3
        df1_normalised, df2_normalised = df1_normalised.align(df2_normalised, join="outer", axis=0)
        df1_normalised = df1_normalised.ffill(axis=0, limit=max_fill).bfill(axis=0, limit=max_fill)
        df2_normalised = df2_normalised.ffill(axis=0, limit=max_fill).bfill(axis=0, limit=max_fill)
    else:
        raise ValueError(f"Method should be ffill or reindex, got {method}")

    valid = row_valid(df1_normalised) & row_valid(df2_normalised) # filter for valid fills  
    df1_normalised = df1_normalised[valid]
    df2_normalised = df2_normalised[valid]

    return df1_normalised, df2_normalised
