from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import requests

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndianStockAnalyzer:
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 risk_free_rate: float = 0.05):
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.risk_free_rate = risk_free_rate
        self.stock_data = None
        self.stock_info = None

    def fetch_stock_data(self, symbol: str, exchange: str = "NSE", 
                        start_date: str = None) -> pd.DataFrame:
        try:
            if start_date:
                start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
                days = (datetime.now() - start_date_dt).days
            else:
                days = 365  # Default to 1 year

            stock_symbol = f"{exchange}:{symbol}-EQ"
            api_url = "http://127.0.0.1:5000/historical/quotes"
            
            response = requests.get(
                api_url,
                params={'symbol': stock_symbol, 'days': days}
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            if df.empty:
                raise ValueError(f"No data found for symbol {stock_symbol}")
                
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            self.stock_data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            return self.stock_data
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise

    def calculate_rsi(self, data: pd.Series) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, data: pd.Series) -> tuple:
        exp1 = data.ewm(span=12, adjust=False).mean()
        exp2 = data.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal, (macd - signal)

    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20) -> tuple:
        middle_band = data.rolling(window=window).mean()
        std_dev = data.rolling(window=window).std()
        return (middle_band + (std_dev * 2), middle_band, middle_band - (std_dev * 2))

    def calculate_technical_indicators(self) -> pd.DataFrame:
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        df = self.stock_data.copy()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        return df

    def calculate_risk_metrics(self) -> dict:
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        daily_returns = self.stock_data['Close'].pct_change().dropna()
        excess_returns = daily_returns - (self.risk_free_rate / 252)
        sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
        volatility = daily_returns.std() * np.sqrt(252) * 100
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        max_drawdown = ((cumulative_returns - rolling_max) / rolling_max).min() * 100

        beta = None
        try:
            nifty_response = requests.get(
                "http://127.0.0.1:5000/historical/quotes",
                params={'symbol': 'NSE:SBIN-EQ', 'days': 365}
            )
            nifty_response.raise_for_status()
            nifty_df = pd.DataFrame(nifty_response.json())
            nifty_df['timestamp'] = pd.to_datetime(nifty_df['timestamp'])
            nifty_df.set_index('timestamp', inplace=True)
            nifty_returns = nifty_df['close'].pct_change().dropna()
            aligned_returns = daily_returns.align(nifty_returns, join='inner')
            beta = np.cov(aligned_returns[0], aligned_returns[1])[0][1] / np.var(aligned_returns[1])
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")

        return {
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'beta': beta
        }

    def generate_trading_signals(self) -> tuple:
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data first.")
            
        df = self.calculate_technical_indicators()
        current = df.iloc[-1]
        prev = df.iloc[-2]
        signals = []
        score = 0

        if current['RSI'] < self.rsi_oversold:
            signals.append(f"Oversold (RSI: {current['RSI']:.2f})")
            score += 1
        elif current['RSI'] > self.rsi_overbought:
            signals.append(f"Overbought (RSI: {current['RSI']:.2f})")
            score -= 1

        if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals.append("MACD Bullish Crossover")
            score += 1
        elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals.append("MACD Bearish Crossover")
            score -= 1

        if current['Close'] > current['SMA_200']:
            signals.append("Price above 200 SMA (Bullish)")
            score += 0.5
        else:
            signals.append("Price below 200 SMA (Bearish)")
            score -= 0.5

        if current['Close'] < current['BB_Lower']:
            signals.append("Price below lower Bollinger Band (Potential Buy)")
            score += 1
        elif current['Close'] > current['BB_Upper']:
            signals.append("Price above upper Bollinger Band (Potential Sell)")
            score -= 1

        recommendation = "Strong Buy" if score >= 2 else "Buy" if score > 0 else \
                        "Hold" if score == 0 else "Sell" if score > -2 else "Strong Sell"
        return recommendation, signals

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        analyzer = IndianStockAnalyzer()
        stock_data = analyzer.fetch_stock_data(
            symbol=data.get('symbol'),
            exchange=data.get('exchange', 'NSE'),
            start_date=data.get('startDate')
        )

        tech_data = analyzer.calculate_technical_indicators()
        recommendation, signals = analyzer.generate_trading_signals()
        risk_metrics = analyzer.calculate_risk_metrics()

        chart_data = [{
            'date': date.strftime('%Y-%m-%d'),
            'price': round(row['Close'], 2),
            'sma20': round(row['SMA_20'], 2) if pd.notnull(row['SMA_20']) else None,
            'sma50': round(row['SMA_50'], 2) if pd.notnull(row['SMA_50']) else None,
            'rsi': round(row['RSI'], 2) if pd.notnull(row['RSI']) else None,
            'macd': round(row['MACD'], 2) if pd.notnull(row['MACD']) else None,
            'signal': round(row['MACD_Signal'], 2) if pd.notnull(row['MACD_Signal']) else None
        } for date, row in tech_data.iterrows()]

        return jsonify({
            'success': True,
            'data': {
                'recommendation': recommendation,
                'currentPrice': round(stock_data['Close'][-1], 2),
                'signals': signals,
                'riskMetrics': {
                    'sharpeRatio': round(risk_metrics['sharpe_ratio'], 2),
                    'volatility': round(risk_metrics['volatility'], 2),
                    'maxDrawdown': round(risk_metrics['max_drawdown'], 2),
                    'beta': round(risk_metrics['beta'], 2) if risk_metrics['beta'] else None
                },
                'chartData': chart_data
            }
        })

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)