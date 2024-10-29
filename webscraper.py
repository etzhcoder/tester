import bs4 as bs
import datetime as dt
import os
import yfinance as yf
import pickle
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')




def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []
    for row in table.findAll('tr')[1:]:
        cols = row.findAll('td')
        if cols:
            ticker = cols[1].text.strip()
            if ticker:
                tickers.append(ticker)
    
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    print(tickers)
    return tickers

def get_data_from_yahoo(reload_sp500 = False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    
    start = dt.datetime(2019, 10, 4)
    end = dt.datetime.now()
    for ticker in tickers:
        # in case connection breaks
        if not os.path.exists(f'stock_dfs/{ticker}.csv'):
            try:
                # Attempt to fetch data using yfinance
                df = yf.download(ticker, start=start, end=end)
                
                # Check if the dataframe contains actual data (not just headers)
                if len(df) > 0 and not df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].isnull().all().all():
                    df.to_csv(f'stock_dfs/{ticker}.csv')
                    print(f'Data saved for {ticker}')
                else:
                    print(f'No valid data for {ticker}, skipping...')

            except Exception as e:
                # Print error message but continue with the next ticker
                print(f'Failed to download {ticker}: {e}')
        else:
            print(f'Already have {ticker}')
#get_data_from_yahoo()

# compiles the data from get_data_from_yahoo into one dataframe
def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
    
    main_df = pd.DataFrame()
    for count, ticker in enumerate(tickers):
        file_path = f"stock_dfs/{ticker}.csv"

        if os.path.exists(file_path):
            df = pd.read_csv(f"stock_dfs/{ticker}.csv")
            df.set_index('Date', inplace=True)
            df.rename(columns={'Adj Close':ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer', rsuffix=f'_{ticker}')

        else:
            print(f"No data for {ticker}, skipping...")

        if count % 10 == 0:
            print(count)
    main_df.ffill(inplace=True)
    main_df.bfill(inplace=True)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

#compile_data()

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    df = df.drop('Date', axis=1)
    df_corr = df.corr()
    print(df_corr.head())
    df_corr.to_csv('sp500corr.csv')

    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    #plt.savefig("correlations.png",dpi = (300))
    plt.show()


visualize_data()


