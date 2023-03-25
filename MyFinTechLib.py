# Declare CONSTANTS using ALLCAPS as convention
TRADING_DAYS = 252
    
def load_csv_to_df(path_to_csv, report_nulls=True, drop_nulls=True):
    """
    Load a CSV file into a Pandas DataFrame
    
    Parameters, [Optional]:
    -----------------------
    
        path_to_csv (file):       CSV file path
        [report_nulls] (boolean): print out nulls found in DataFrame? Default: True
        [drop_nulls] (boolean):   remove all null values from DataFrame? Default: True
        
    Returns:
    --------
        
        df_obj (DataFrame):     Pandas DataFrame representation of CSV file
        
    """

    # Import necessary libraries to use locally
    import pandas as pd
    from pathlib import Path
        
    # check that the file exists so that we don't crash
    try:
        csv_file = Path(path_to_csv)
        df_obj = pd.read_csv(csv_file, index_col='Date', infer_datetime_format=True, parse_dates=True)
        # print out useful data to user
        if report_nulls:
            print(f"Nulls found:\n--------------------------------\n{df_obj.isna().sum()}\n--------------------------------")
        if drop_nulls:
            df_obj.dropna(inplace=True)
            print("All null values removed!")
        return df_obj
    except FileNotFoundError:
        print(f"File: {path_to_csv}, not found!")

def sharpe_ratio(data_set, risk_free=[], annualized=True):
    """
    Calculate Sharpe Ratio for a DataFrame of returns (pct_change)
    
    Parameters, [Optional]:
    -----------------------
    
        data_set (DataFrame):    Pandas DataFrame of returns (pct_change)
        [risk_free] (DataFrame): Pandas DataFrame of risk free rates in 1 column. Default: 0
        [annualized] (Boolean):  Annualize? Default: True
        
    Returns:
    --------
    
        sr (DataFrame): Pandas DataFrame of Sharpe Ratios
        
    """
    
    # Import necessary libraries to use locally
    import pandas as pd
    import numpy as np
    
    risk_free_rate = pd.DataFrame({'risk_free':[0] * len(data_set)}, index=data_set.index) if not risk_free else risk_free
    all_data = pd.concat([data_set, risk_free_rate], axis='columns', join='inner')
    trading_days = 252 if annualized else 1
    sr = ((all_data.mean() - all_data['risk_free'].mean()) * trading_days) / (data_set.std() * np.sqrt(trading_days))
    sr.drop('risk_free', inplace=True)
    return sr

def beta(data1, data2, roll_win=0):
    """
    Calculate Beta of data1 relative to data2. data1 would typically be a portfolio or stock
    and data2 would be the market or index. Beta measures the volatility of a portfolio or security
    against a market or index.
    
    Parameters, [Optional]:
    -----------------------
    
        data1 (DataFrame): Pandas DataFrame of returns (pct_change) for a stock or portfolio
        data2 (DataFrame): Pandas DataFrame of returns (pct_change) for a market or index
        [roll_win] (int):  Number of datapoints to calculate a rolling beta over. Default: 0, no rolling
        
    Returns:
    --------
    
        beta (float): measure of data1's volatility relative to data2
        
    """
    
    # Import necessary libraries to use locally
    import pandas as pd
    
    if roll_win > 0:
        variance = data2.rolling(window=roll_win).var()
        covariance = data1.rolling(window=roll_win).cov(data2)
        beta = covariance / variance
    else:
        variance = data2.var()
        covariance = data1.cov(data2)
        beta = covariance / variance

    return beta
