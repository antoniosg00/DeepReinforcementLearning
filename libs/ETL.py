import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def ADX(df):
    n_adx = 14  # The period for ADX calculation
    n = 14

    # Calculate True Range (TR)
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close-Prev'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-Close-Prev'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-Close-Prev', 'Low-Close-Prev']].max(axis=1)

    # Calculating the Positive Directional Indicator (+DI) and Negative Directional Indicator (-DI)
    df['High-Prev'] = df['High'].shift(1)
    df['Low-Prev'] = df['Low'].shift(1)

    df['+DM'] = 0.0
    df['-DM'] = 0.0

    df.loc[(df['High'] - df['High-Prev'] > df['Low-Prev'] - df['Low']), '+DM'] = df['High'] - df['High-Prev']
    df.loc[(df['Low-Prev'] - df['Low'] > df['High'] - df['High-Prev']), '-DM'] = df['Low-Prev'] - df['Low']

    df['+DI'] = (df['+DM'].rolling(window=n, min_periods=1).sum() / df['TR'].rolling(window=n, min_periods=1).sum()) * 100
    df['-DI'] = (df['-DM'].rolling(window=n, min_periods=1).sum() / df['TR'].rolling(window=n, min_periods=1).sum()) * 100

    # Calculating the Average Directional Movement Index (ADX)
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(window=n_adx, min_periods=1).mean()

    # Delete intermediate columns
    ADX_results = np.array(df[['ADX']].values.tolist()).squeeze()
    df.drop(['High-Low', 'High-Close-Prev', 'Low-Close-Prev', 'TR', 'High-Prev', 'Low-Prev', '+DM', '-DM', '+DI', '-DI', 'DX', 'ADX'], axis=1, inplace=True)

    return ADX_results

class ETLclass: # Extract, transform and load data
    def __init__(self, target, val_size=0.1, test_size=0.1) -> None:
        self.symbols = ['SAB.MC', 'SAN.MC', 'BKT.MC', 'BBVA.MC', 'CABK.MC', 'UNI.MC', '^IBEX', '^IXIC','^DJI', '^GSPC','^N225', '^STOXX50E', 'EURUSD=X', 'EURGBP=X','EURJPY=X'] # The 6 IBEX 35 banks and other related assets
        self.assets_used = []
        if target in self.symbols:
            self.df = yf.Ticker(target).history(period='max').reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']] # Pandas df
            self.df['Date'] = [datetime.strptime(a.strftime('%Y-%m-%d'),'%Y-%m-%d') for a in self.df['Date']] # Change date format to YYYYY-MM-DD 
        else: raise Exception('Create an instance with the symbol of an IBEX 35 bank')

        self.tech_df = self.technical_indicators()
        self.fft_df = self.fourier_transform()
        self.assets_df, self.assets_used = self.related_assets(target)

        self.df = self.df[-len(self.tech_df.index):].reset_index().drop(['index'], axis=1)    # I adjust the 3 dataframes to the same number of examples (5000 at the moment).
        self.fft_df = self.fft_df[-len(self.tech_df.index):].reset_index().drop(['index'], axis=1)   # since the first tech_df's were NaN and were eliminated.
        self.dates = np.array(self.df['Date']).reshape(-1,1)[-len(self.tech_df.index):]

        self.combined_df, self.train_df, self.val_df, self.test_df = self.create_and_split(val_size=val_size, test_size=test_size) # We create the dataframe and split it into train/val/test
        self.scaled_train_tensor, self.scaled_val_tensor, self.scaled_test_tensor = self.scale() # In case it is useful

    def related_assets(self, target): # We obtain df from related companies
        assets_used = []
        assets_df = pd.DataFrame()
        for symb in self.symbols:
            if symb != target:
                obj = yf.Ticker(symb).history(period='max').reset_index()[['Close']]
                if len(obj.index) > len(self.tech_df.index): 
                    assets_df[symb+'Close'] = obj[-len(self.tech_df.index):].reset_index().drop(['index'], axis=1)
                    assets_used.append(symb)
                # Assets with less data I will not use them.
        return assets_df, assets_used

    def technical_indicators(self): # We calculate the technical indicators of our bank
        tech_df = self.df[['Date']].sort_values(by='Date') # Sort the DataFrame by date if not already sorted
        
        # Calculation of the Simple Moving Average (SMA) of 7 and 21 days
        tech_df['SMA_7'] = self.df['Close'].rolling(window=7).mean()
        tech_df['SMA_21'] = self.df['Close'].rolling(window=21).mean()

        # Calculation of the Exponential Moving Average (EMA) = EWMA with 10-day period (short term)
        tech_df['EMA'] =self.df['Close'].ewm(span=10).mean()

        # Calculation of Bollinger bands
        tech_df['sigma_20'] = self.df['Close'].rolling(window=20).std() # Standard deviation in 20-day period
        tech_df['uBB'] = tech_df['SMA_21'] + 2 * tech_df['sigma_20'] # Upper Bollinger Band
        tech_df['lBB'] = tech_df['SMA_21'] - 2 * tech_df['sigma_20'] # Lower Bollinger Band

        # Calculation of Moving Average Convergence/Divergence (MACD) (by definition EMA26-EMA12)
        ema_short = self.df['Close'].ewm(span=12, adjust=False).mean() # Short-term EMA
        ema_long = self.df['Close'].ewm(span=26, adjust=False).mean() # Long-term EMA
        tech_df['MACD'] = ema_short - ema_long
        tech_df['SignalLine'] = tech_df['MACD'].ewm(span=9, adjust=False).mean() # 9-day signal line (definition)

        # Calculation of Relative Strength Index (RSI) in 14 days (https://es.wikipedia.org/wiki/%C3%8Dndice_de_fuerza_relativa)
        tech_df['Diff'] = self.df['Close'].diff(1) # Differences between consecutive prices
        
        tech_df['Gain'] = np.where(tech_df['Diff'] > 0, tech_df['Diff'], 0) # Positive values (gains) and negative values (losses)
        tech_df['Lose'] = np.where(tech_df['Diff'] < 0, -tech_df['Diff'], 0) # np.where(condition, if complied with, if not)

        tech_df['EMA_Gain'] = tech_df['Gain'].rolling(window=14).mean() # (EMA) of profit and loss
        tech_df['EMA_Lose'] = tech_df['Lose'].rolling(window=14).mean() 
 
        tech_df['RS'] = tech_df['EMA_Gain'] / tech_df['EMA_Lose'] # Calculation of RSI
        tech_df['RSI'] = 100 - (100 / (1 + tech_df['RS'])) 

        tech_df = tech_df.drop(['Diff', 'Gain', 'Lose', 'EMA_Gain', 'EMA_Lose', 'RS'], axis=1) # Eliminating unnecessary columns

        # Stochastic Oscillator calculation --> compares current with max and min of a given period (14 days)
        tech_df['min'] = self.df['Close'].rolling(window=14).min()
        tech_df['max'] = self.df['Close'].rolling(window=14).max()
        tech_df['%K'] = 100 * ((self.df['Close'] - tech_df['min']) / (tech_df['max'] - tech_df['min']))
        tech_df['%D'] = tech_df['%K'].rolling(window=3).mean() # SMA of %K
        tech_df = tech_df.drop(['min', 'max'], axis=1) # Eliminate unnecessary columns 

        # ADX calculation (more complex with external function). Period of 14 days.
        tech_df['ADX'] = ADX(self.df)

        return tech_df[-5000:].reset_index().drop(['index'],axis=1)    # I will keep the last 5000 data to have assets related to that amount of data.
    
    def fourier_transform(self): # Calculation of Fourier transforms (by FFT)
        # We will calculate it with different number of components to extract long and short distance trends.
        fft_values = np.fft.fft(self.df['Close'].values)
        fft_df = self.df[['Date']].sort_values(by='Date')

        for comp in [3, 6, 9,25, 100]:
            fft_freqs= np.copy(fft_values)
            fft_freqs[comp:-comp]=0 # I will keep only the 'comp' number of main frequencies by eliminating the intermediate ones.
            fft_df['ifft_'+str(comp)] = np.fft.ifft(fft_freqs) # Inverse transform on the frequencies I have saved
        
        return fft_df
    
    def create_and_split(self, val_size, test_size):
        lista_dfs = [self.df[['Date' ,'Close', 'Open', 'High', 'Low', 'Volume']], self.tech_df, self.fft_df.apply(np.real), self.assets_df]
        combined_df = pd.concat(lista_dfs, axis=1).drop(['Date'], axis=1) # We create a dataframe with all the data we have available
        combined_df.index = self.dates.reshape(-1,) # Add the date as an index

        return combined_df, combined_df[:int(len(combined_df)*(1-val_size-test_size))], combined_df[int(len(combined_df)*(1-val_size-test_size)):int(len(combined_df)*(1-test_size))], combined_df[int(len(combined_df)*(1-test_size)):]

    def scale(self):
        scaler = MinMaxScaler()  # Let's normalize the data with min-max method
        scaled_train_tensor = scaler.fit_transform(self.train_df) # Generate tensor to train the DRL algorithm
        scaled_val_tensor = scaler.transform(self.val_df) # Generate tensor to evaluate the DRL algorithm
        scaled_test_tensor = scaler.transform(self.test_df) # Generate tensor for final testing of the DRL algorithm
        
        return scaled_train_tensor, scaled_val_tensor, scaled_test_tensor