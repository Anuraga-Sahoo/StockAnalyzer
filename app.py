#################################################################################################
# File Name - app.py
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This file contains the code for the Flask application that serves the frontend
#################################################################################################

from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import click
import time
import requests


# Initialize Flask app  and set up logging  

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#################################################################################################
# Class Name - IndianStockAnalyzer
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This class contains the logic for analyzing Indian stocks using technical indicators
#################################################################################################

class IndianStockAnalyzer:
   

    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 risk_free_rate: float = 0.05):
        """
        Initialize the Indian Stock Analyzer.
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.risk_free_rate = risk_free_rate
        self.stock_data = None
        self.stock_info = None
        self.recommendationTopGainer = []

#################################################################################################
# Function Name - fetch_stock_data
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function fetches stock data from Yahoo Finance
#################################################################################################

    def fetch_stock_data(self, symbol: str, exchange: str = "NS", 
                        start_date: str = None,
                        interval: str = "1d") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        """
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                
            stock_symbol = f"{symbol}.{exchange}"
            stock = yf.Ticker(stock_symbol)
            
            data = stock.history(start=start_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {stock_symbol}")
                
            self.stock_data = data
            self.stock_info = stock.info
            
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise
    # def recomendationTopGainer(self) -> list:
    #     recomendationTopGainers = []
    #     return recomendationTopGainers
#################################################################################################
# Function Name - calculate_rsi
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates the Relative Strength Index (RSI) technical indicator
#################################################################################################

    def calculate_rsi(self, data: pd.Series) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

#################################################################################################
# Function Name - calculate_macd
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates the Moving Average Convergence Divergence (MACD) technical indicator
#################################################################################################

    def calculate_macd(self, data: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal line, and Histogram."""
        exp1 = data.ewm(span=12, adjust=False).mean()
        exp2 = data.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

#################################################################################################
# Function Name - calculate_bollinger_bands
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates the Bollinger Bands technical indicator
#################################################################################################

    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle_band = data.rolling(window=window).mean()
        std_dev = data.rolling(window=window).std()
        upper_band = middle_band + (std_dev * 2)
        lower_band = middle_band - (std_dev * 2)
        return upper_band, middle_band, lower_band

#################################################################################################
# Function Name - calculate_technical_indicators
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates various technical indicators
#################################################################################################

    def calculate_technical_indicators(self) -> pd.DataFrame:
        """Calculate various technical indicators."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        df = self.stock_data.copy()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Volume Indicators
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        
        return df

#################################################################################################
# Function Name - calculate_risk_metrics
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function calculates various risk metrics
#################################################################################################

    def calculate_risk_metrics(self) -> dict:
        """Calculate various risk metrics."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        daily_returns = self.stock_data['Close'].pct_change().dropna()
        
        # Sharpe Ratio
        excess_returns = daily_returns - (self.risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Maximum Drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Beta (using Nifty 50 as benchmark) 
        try:
            nifty = yf.download('^NSEI', start=self.stock_data.index[0])
            nifty_returns = nifty['Close'].pct_change().dropna()
            beta = np.cov(daily_returns, nifty_returns)[0][1] / np.var(nifty_returns)
        except:
            beta = None
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'beta': beta
        }

#################################################################################################
# Function Name - generate_trading_signals
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function generates trading signals based on technical indicators
#################################################################################################
    

    def generate_trading_signals(self) -> tuple[str, list, list]:
        """Generate trading signals based on technical indicators."""
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        df = self.calculate_technical_indicators()
        current = df.iloc[-1]
        prev = df.iloc[-2]

        signals = []
        score = 0
        predictTopgainer = []
        
        # RSI Signals
        if current['RSI'] < self.rsi_oversold:
            signals.append(f"Oversold (RSI: {current['RSI']:.2f})")
            score += 1
        elif current['RSI'] > self.rsi_overbought:
            signals.append(f"Overbought (RSI: {current['RSI']:.2f})")
            score -= 1
            
        # MACD Signals
        if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals.append("MACD Bullish Crossover")
            score += 1
        elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals.append("MACD Bearish Crossover")
            score -= 1
            
        # Moving Average Signals
        if current['Close'] > current['SMA_200']:
            signals.append("Price above 200 SMA (Bullish)")
            score += 0.5
        else:
            signals.append("Price below 200 SMA (Bearish)")
            score -= 0.5
            
        # Bollinger Bands Signals
        if current['Close'] < current['BB_Lower']:
            signals.append("Price below lower Bollinger Band (Potential Buy)")
            score += 1
        elif current['Close'] > current['BB_Upper']:
            signals.append("Price above upper Bollinger Band (Potential Sell)")
            score -= 1
            
        # Generate recommendation based on score
        if score >= 2:
            recommendation = "Strong Buy"
            predictTopgainer.append(score)
        elif score > 0:
            recommendation = "Buy"
            predictTopgainer.append(score)

        elif score == 0:
            recommendation = "Hold"
            predictTopgainer.append(score)

        elif score > -2:
            recommendation = "Sell" 
            # predictTopgainer.append(score)


        else:
            recommendation = "Strong Sell"
            
        return recommendation, signals, predictTopgainer
    
    def fetch_top_gainers(self) -> str:
            # get top gainer data
            # Define URLs
        nse_url = "https://www.nseindia.com"  # NSE homepage (to get cookies)
        api_url_topgainer = "https://www.nseindia.com/api/live-analysis-variations?index=gainers"
        api_url_toplooser = "https://www.nseindia.com/api/live-analysis-variations?index=loosers"


# Define headers
        headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    }

# Create a session
        session = requests.Session()
        self.recommendationTopGainer = {}

        try:
             # Step 1: Get cookies from the NSE homepage
            session.get(nse_url, headers=headers)

            # Step 2: Use the session with cookies to fetch API data of topgainer
            gainer_response = session.get(api_url_topgainer, headers=headers)
            gainer_response.raise_for_status()  # Raise an error if the request fails

            # Step 3: Use the session with cookies to fetch API data of topgainer
            looser_response = session.get(api_url_toplooser, headers=headers)
            looser_response.raise_for_status()  # Raise an error if the request fails

            # Step 4: Print the data
            top_gainer_data = gainer_response.json()
            top_looser_data = looser_response.json()

            top_gainer_symbols = [item["symbol"] for item in top_gainer_data["NIFTY"].get("data", [])]
            logger.info("Symbols extracting from top gainer api : %s", top_gainer_symbols)
            

            top_looser_symbols = [items["symbol"] for items in top_looser_data["NIFTY"].get("data", [])]
            logger.info("Symbols extracting from top looser api : %s", top_looser_symbols)
            
            symbols = top_gainer_symbols + top_looser_symbols
            print(symbols)
            for symbol in symbols:
                try:
                    logger.info("Symbol running in loop %s", symbol)
                    print(symbol)
                    self.fetch_stock_data(symbol)
                    self.calculate_technical_indicators()
                    _, _, score_list = self.generate_trading_signals()
                    if score_list:
                        # self.recommendationTopGainer.append(score_list[0])
                        self.recommendationTopGainer[symbol] = score_list[0]
                        print(score_list[0])
                except Exception as e :
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    self.recommendationTopGainer[symbol] = None
            return symbols
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching top gainers: {str(e)}")
            return []


#################################################################################################
# Function Name - index
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function renders the index.html template
#################################################################################################

@app.route('/')
def index():
    return render_template('index.html')

#################################################################################################
# Function Name - analyze
# Author - Ojas Ulhas Dighe
# Date - 3rd Mar 2025
# Description - This function analyzes the stock data and returns the analysis results
#################################################################################################

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        exchange = data.get('exchange', 'NS')
        start_date = data.get('startDate')

        analyzer = IndianStockAnalyzer()
        stock_data = analyzer.fetch_stock_data(symbol, exchange, start_date)
        
        # Calculate main symbol
        tech_data = analyzer.calculate_technical_indicators()
        recommendation, signals, predictTopgainer = analyzer.generate_trading_signals()
        risk_metrics = analyzer.calculate_risk_metrics()
        #  process top gainers
        topSymbol = analyzer.fetch_top_gainers()
        
        # recomendationTopGainers = analyzer.recomendationTopGainer()
          
        print("Top symbols : = ",topSymbol)
        print("predict", predictTopgainer)
        print("signals" , signals)
        # print(analyzer.recomendationTopGainer)


        # Prepare chart data
        chart_data = []
        for date, row in tech_data.iterrows():
            chart_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': round(float(row['Close']), 2),
                'sma20': round(float(row['SMA_20']), 2) if pd.notnull(row['SMA_20']) else None,
                'sma50': round(float(row['SMA_50']), 2) if pd.notnull(row['SMA_50']) else None,
                'rsi': round(float(row['RSI']), 2) if pd.notnull(row['RSI']) else None,
                'macd': round(float(row['MACD']), 2) if pd.notnull(row['MACD']) else None,
                'signal': round(float(row['MACD_Signal']), 2) if pd.notnull(row['MACD_Signal']) else None
            })

        response = {
            'success': True,
            'data': {
                'recommendation': recommendation,
                # 'currentPrice': round(float(stock_data['Close'][-1]), 2),
                'currentPrice': round(float(stock_data['Close'].iloc[-1]), 2),

                'signals': signals,
                'riskMetrics': {
                    'sharpeRatio': round(float(risk_metrics['sharpe_ratio']), 2),
                    'volatility': round(float(risk_metrics['volatility']), 2),
                    'maxDrawdown': round(float(risk_metrics['max_drawdown']), 2),
                    'beta': round(float(risk_metrics['beta']), 2) if risk_metrics['beta'] else None
                },
                'chartData': chart_data,
                'topGainerScores': analyzer.recommendationTopGainer
            }
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

# updated