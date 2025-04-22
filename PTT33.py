# pair_trading_system.py - Sistema completo di pair trading

import os
import pandas as pd
import numpy as np
import streamlit as st
import sqlite3
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import multiprocessing
from functools import partial
import itertools
import time
import random
from datetime import datetime, timedelta

# ------ Database Manager ------
class DatabaseManager:
    def __init__(self, db_path='data/stock_data.db'):
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        self.db_path = db_path
        self.initialize_database()
    
    def get_connection(self):
        """Create a database connection"""
        return sqlite3.connect(self.db_path)
    
    def initialize_database(self):
        """Initialize the database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Stock prices table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            date TEXT,
            ticker TEXT,
            price REAL,
            PRIMARY KEY (date, ticker)
        )
        ''')
        
        # Pairs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock1 TEXT,
            stock2 TEXT,
            sector1 TEXT,
            sector2 TEXT,
            hedge_ratio REAL,
            correlation REAL,
            coint_pvalue REAL,
            last_updated TEXT,
            UNIQUE(stock1, stock2)
        )
        ''')
        
        # Trading signals table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trading_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair_id INTEGER,
            direction TEXT,
            zscore REAL,
            entry_date TEXT,
            entry_spread REAL,
            target_spread REAL,
            exit_date TEXT,
            exit_spread REAL,
            profit_loss REAL,
            status TEXT,
            FOREIGN KEY(pair_id) REFERENCES pairs(id)
        )
        ''')
        
        # Backtest results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair_id INTEGER,
            start_date TEXT,
            end_date TEXT,
            entry_threshold REAL,
            exit_threshold REAL,
            total_return REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            trade_count INTEGER,
            win_rate REAL,
            last_updated TEXT,
            FOREIGN KEY(pair_id) REFERENCES pairs(id)
        )
        ''')
        
        # Sectors table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sectors (
            sector TEXT PRIMARY KEY,
            description TEXT
        )
        ''')
        
        # Stock-sector mapping table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_sectors (
            ticker TEXT PRIMARY KEY,
            sector TEXT,
            FOREIGN KEY(sector) REFERENCES sectors(sector)
        )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    
    def save_stock_data(self, data_df):
        """
        Save stock price data to the database
        
        Parameters:
            data_df (DataFrame): DataFrame with DateTimeIndex and columns as tickers
        """
        conn = self.get_connection()
        
        # Reshape the data for SQLite storage
        for ticker in data_df.columns:
            stock_data = data_df[ticker].reset_index()
            stock_data.columns = ['date', 'price']
            stock_data['ticker'] = ticker
            stock_data['date'] = stock_data['date'].dt.strftime('%Y-%m-%d')
            
            # Insert data
            for _, row in stock_data.iterrows():
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO stock_prices (date, ticker, price) VALUES (?, ?, ?)",
                        (row['date'], row['ticker'], row['price'])
                    )
                except Exception as e:
                    print(f"Error inserting data for {ticker} on {row['date']}: {e}")
        
        conn.commit()
        conn.close()
        print(f"Saved price data for {len(data_df.columns)} stocks")
    
    def get_stock_data(self, tickers=None, start_date=None, end_date=None):
        """
        Retrieve stock price data from the database
        
        Parameters:
            tickers (list): List of stock tickers to retrieve
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame: DataFrame with DateTimeIndex and columns as tickers
        """
        conn = self.get_connection()
        
        query = "SELECT date, ticker, price FROM stock_prices"
        conditions = []
        
        if tickers:
            placeholders = ','.join(['?'] * len(tickers))
            conditions.append(f"ticker IN ({placeholders})")
        
        if start_date:
            conditions.append("date >= ?")
        
        if end_date:
            conditions.append("date <= ?")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        params = []
        if tickers:
            params.extend(tickers)
        if start_date:
            params.append(start_date)
        if end_date:
            params.append(end_date)
        
        # Execute query
        df = pd.read_sql_query(query, conn, params=params)
        
        # Reshape to wide format
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df_pivot = df.pivot(index='date', columns='ticker', values='price')
            return df_pivot
        else:
            return pd.DataFrame()
    
    def save_pairs(self, pairs_list):
        """
        Save analyzed pairs to the database
        
        Parameters:
            pairs_list (list): List of dictionaries with pair information
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        for pair in pairs_list:
            try:
                cursor.execute('''
                INSERT INTO pairs 
                (stock1, stock2, sector1, sector2, hedge_ratio, correlation, coint_pvalue, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stock1, stock2) 
                DO UPDATE SET 
                    hedge_ratio=excluded.hedge_ratio,
                    correlation=excluded.correlation,
                    coint_pvalue=excluded.coint_pvalue,
                    last_updated=excluded.last_updated
                ''', (
                    pair['stock1'],
                    pair['stock2'],
                    pair['sector1'],
                    pair['sector2'],
                    pair['hedge_ratio'],
                    pair['correlation'],
                    pair['coint_pvalue'],
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
            except Exception as e:
                print(f"Error saving pair {pair['stock1']}/{pair['stock2']}: {e}")
        
        conn.commit()
        conn.close()
        print(f"Saved {len(pairs_list)} pairs to database")
    
    def get_pairs(self, sector=None, max_pvalue=0.05, min_correlation=0.5):
        """
        Retrieve pairs from the database
        
        Parameters:
            sector (str): Filter by sector
            max_pvalue (float): Maximum p-value for cointegration
            min_correlation (float): Minimum correlation
            
        Returns:
            list: List of dictionaries with pair information
        """
        conn = self.get_connection()
        
        query = '''
        SELECT id, stock1, stock2, sector1, sector2, hedge_ratio, correlation, coint_pvalue, last_updated 
        FROM pairs 
        WHERE coint_pvalue <= ? AND ABS(correlation) >= ?
        '''
        
        params = [max_pvalue, min_correlation]
        
        if sector:
            query += " AND (sector1 = ? OR sector2 = ?)"
            params.extend([sector, sector])
        
        # Execute query
        df = pd.read_sql_query(query, conn, params=params)
        
        # Convert to list of dictionaries
        pairs = []
        for _, row in df.iterrows():
            pairs.append(dict(row))
        
        conn.close()
        return pairs
    
    def save_trading_signal(self, signal):
        """
        Save a trading signal to the database
        
        Parameters:
            signal (dict): Dictionary with signal information
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO trading_signals 
        (pair_id, direction, zscore, entry_date, entry_spread, target_spread, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['pair_id'],
            signal['direction'],
            signal['zscore'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            signal['entry_spread'],
            signal['target_spread'],
            signal['status']
        ))
        
        signal_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return signal_id
    
    def update_trading_signal(self, signal_id, exit_date, exit_spread, profit_loss, status):
        """
        Update a trading signal when it's closed
        
        Parameters:
            signal_id (int): Signal ID
            exit_date (str): Exit date
            exit_spread (float): Exit spread
            profit_loss (float): Profit/loss
            status (str): Signal status
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE trading_signals
        SET exit_date = ?, exit_spread = ?, profit_loss = ?, status = ?
        WHERE id = ?
        ''', (
            exit_date,
            exit_spread,
            profit_loss,
            status,
            signal_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_signals(self):
        """
        Get all active trading signals
        
        Returns:
            list: List of dictionaries with signal information
        """
        conn = self.get_connection()
        
        query = '''
        SELECT s.id, s.pair_id, s.direction, s.zscore, s.entry_date, s.entry_spread, s.target_spread, s.status,
               p.stock1, p.stock2, p.sector1, p.sector2, p.hedge_ratio
        FROM trading_signals s
        JOIN pairs p ON s.pair_id = p.id
        WHERE s.status = 'active'
        '''
        
        # Execute query
        df = pd.read_sql_query(query, conn)
        
        # Convert to list of dictionaries
        signals = []
        for _, row in df.iterrows():
            signals.append(dict(row))
        
        conn.close()
        return signals
    
    def save_backtest_result(self, result):
        """
        Save backtest result to the database
        
        Parameters:
            result (dict): Dictionary with backtest result
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO backtest_results 
        (pair_id, start_date, end_date, entry_threshold, exit_threshold, 
         total_return, sharpe_ratio, max_drawdown, trade_count, win_rate, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['pair_id'],
            result['start_date'],
            result['end_date'],
            result['entry_threshold'],
            result['exit_threshold'],
            result['total_return'],
            result['sharpe_ratio'],
            result['max_drawdown'],
            result['trade_count'],
            result['win_rate'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        conn.commit()
        conn.close()
    
    def save_sectors(self, sectors):
        """
        Save sector information to the database
        
        Parameters:
            sectors (dict): Dictionary of sector name -> description
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        for sector, description in sectors.items():
            cursor.execute('''
            INSERT OR REPLACE INTO sectors (sector, description)
            VALUES (?, ?)
            ''', (sector, description))
        
        conn.commit()
        conn.close()
    
    def save_stock_sectors(self, stock_sectors):
        """
        Save stock-sector mappings to the database
        
        Parameters:
            stock_sectors (dict): Dictionary of ticker -> sector
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        for ticker, sector in stock_sectors.items():
            cursor.execute('''
            INSERT OR REPLACE INTO stock_sectors (ticker, sector)
            VALUES (?, ?)
            ''', (ticker, sector))
        
        conn.commit()
        conn.close()

# ------ Data Collector ------
class DataCollector:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.sector_stocks = self.get_sector_stocks()
    
    def get_sector_stocks(self):
        """
        Returns a dictionary of stocks organized by sector
        """
        sectors = {
            'Technology': [
                'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CSCO', 'IBM', 'PLTR', 'ACN', 'AMD', 
                'INTC', 'TSM', 'ADBE', 'CRM', 'PYPL', 'QCOM', 'TXN', 'UBER', 'MU', 'AMAT'
            ],
            'Healthcare': [
                'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
                'UNH', 'ISRG', 'GILD', 'REGN', 'BIIB', 'VRTX', 'MRNA', 'HCA', 'CVS', 'ZTS'
            ],
            'Financial': [
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'SCHW', 'V',
                'MA', 'SPGI', 'CME', 'ICE', 'CB', 'PNC', 'TFC', 'USB', 'PRU', 'MET'
            ],
            'Consumer Discretionary': [
                'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT', 'BKNG', 'TJX', 'EBAY',
                'MAR', 'YUM', 'DG', 'DLTR', 'BBY', 'ROST', 'ETSY', 'ULTA', 'DPZ', 'RCL'
            ],
            'Energy': [
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'OXY', 'PSX', 'MPC', 'VLO',
                'KMI', 'WMB', 'DVN', 'HAL', 'BKR', 'OKE', 'FANG', 'HES', 'MRO', 'APA'
            ],
            'Industrial': [
                'HON', 'UNP', 'UPS', 'BA', 'CAT', 'DE', 'LMT', 'RTX', 'GE', 'MMM',
                'CSX', 'NOC', 'ITW', 'EMR', 'ETN', 'GD', 'NSC', 'WM', 'PH', 'CMI'
            ]
        }
        
        # Save sector information to database
        sector_descriptions = {
            'Technology': 'Technology companies including software, hardware, and IT services',
            'Healthcare': 'Healthcare companies including pharmaceuticals, medical devices, and healthcare providers',
            'Financial': 'Financial institutions including banks, insurance, and investment services',
            'Consumer Discretionary': 'Consumer discretionary companies including retail, automotive, and leisure',
            'Energy': 'Energy companies including oil, gas, and renewable energy',
            'Industrial': 'Industrial companies including aerospace, machinery, and transportation'
        }
        self.db_manager.save_sectors(sector_descriptions)
        
        # Create stock-sector mapping
        stock_sectors = {}
        for sector, stocks in sectors.items():
            for stock in stocks:
                stock_sectors[stock] = sector
        
        self.db_manager.save_stock_sectors(stock_sectors)
        
        return sectors
    
    def fetch_sector_data(self, sectors=None, period="6mo", interval="1d"):
        """
        Fetches data for stocks in specified sectors
        
        Parameters:
            sectors (list): List of sector names to fetch data for
            period (str): Time period to fetch data for
            interval (str): Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame: Historical price data
        """
        if sectors is None:
            sectors = list(self.sector_stocks.keys())
        
        all_stocks = []
        sector_mapping = {}
        
        # Create a flat list of all stocks in specified sectors
        for sector in sectors:
            if sector in self.sector_stocks:
                sector_stocks = self.sector_stocks[sector]
                all_stocks.extend(sector_stocks)
                
                for stock in sector_stocks:
                    sector_mapping[stock] = sector
        
        # Remove duplicates while preserving order
        all_stocks = list(dict.fromkeys(all_stocks))
        
        # Check if we already have recent data in the database
        today = datetime.now().date()
        latest_data = self.db_manager.get_stock_data(
            tickers=all_stocks,
            start_date=(today - timedelta(days=7)).strftime('%Y-%m-%d')
        )
        
        latest_date = latest_data.index.max() if not latest_data.empty else None
        
        # If we have data from today or it's a weekend/holiday, use the database data
        if latest_date is not None and (
            latest_date.date() == today or 
            today.weekday() >= 5 or  # Weekend
            (today - latest_date.date()).days <= 1  # Recent data (handles holidays)
        ):
            print(f"Using recent data from database (latest: {latest_date})")
            
            # Get the full period data from the database
            start_date = (today - pd.Timedelta(period)).strftime('%Y-%m-%d')
            return self.db_manager.get_stock_data(
                tickers=all_stocks, 
                start_date=start_date
            ), sector_mapping
        
        # Fetch new data from Yahoo Finance
        print(f"Fetching new data from Yahoo Finance for {len(all_stocks)} stocks")
        
        # Split into batches to avoid API limitations
        batch_size = 50
        all_data = []
        
        for i in range(0, len(all_stocks), batch_size):
            batch = all_stocks[i:i+batch_size]
            print(f"Fetching batch {i//batch_size + 1}/{(len(all_stocks)+batch_size-1)//batch_size}")
            
            try:
                batch_data = yf.download(batch, period=period, interval=interval)
                
                # If only one stock, fix the format
                if len(batch) == 1:
                    batch_data = batch_data.iloc[:, batch_data.columns.get_level_values(0) == 'Adj Close']
                    batch_data.columns = [batch[0]]
                else:
                    batch_data = batch_data['Adj Close']
                
                all_data.append(batch_data)
                
                # Add a delay to avoid API rate limits
                time.sleep(1 + random.random())
                
            except Exception as e:
                print(f"Error fetching batch: {e}")
                # Continue with the next batch
        
        if not all_data:
            return pd.DataFrame(), sector_mapping
        
        # Combine all batches
        data = pd.concat(all_data, axis=1)
        
        # Save to database
        self.db_manager.save_stock_data(data)
        
        return data, sector_mapping
    
    def update_data(self):
        """
        Update data for all sectors
        
        Returns:
            tuple: (price_data, sector_mapping)
        """
        print(f"Updating data at {datetime.now()}")
        return self.fetch_sector_data()

# ------ Pair Analyzer ------
class PairAnalyzer:
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def analyze_pair(self, pair, data, min_correlation=0.5):
        """
        Analyze a single pair for cointegration
        
        Parameters:
            pair (tuple): Tuple of (stock1, stock2)
            data (DataFrame): Historical price data
            min_correlation (float): Minimum correlation threshold
            
        Returns:
            dict: Pair analysis results or None if not cointegrated
        """
        stock1, stock2 = pair
        
        # Skip if data is missing
        if stock1 not in data.columns or stock2 not in data.columns:
            return None
            
        if data[stock1].isnull().any() or data[stock2].isnull().any():
            return None
            
        # Calculate correlation
        correlation = data[stock1].corr(data[stock2])
        
        if abs(correlation) < min_correlation:
            return None
            
        # Perform cointegration test
        series1 = data[stock1]
        series2 = data[stock2]
        
        try:
            # Engle-Granger test
            _, pvalue, _ = coint(series1, series2)
            
            # Calculate hedge ratio
            X = sm.add_constant(series1)
            model = sm.OLS(series2, X).fit()
            hedge_ratio = model.params[1]
            
            if pvalue < 0.05:  # Using default threshold
                return {
                    'stock1': stock1,
                    'stock2': stock2,
                    'correlation': correlation,
                    'coint_pvalue': pvalue,
                    'hedge_ratio': hedge_ratio
                }
            return None
        except:
            return None
    
    def analyze_pairs(self, data, sector_mapping, min_correlation=0.5, use_parallel=True):
        """
        Analyze all possible pairs for cointegration
        
        Parameters:
            data (DataFrame): Historical price data
            sector_mapping (dict): Mapping of tickers to sectors
            min_correlation (float): Minimum correlation threshold
            use_parallel (bool): Whether to use parallel processing
            
        Returns:
            dict: Dictionary of cointegrated pairs by sector
        """
        # Get all possible stock combinations
        all_pairs = list(itertools.combinations(data.columns, 2))
        
        print(f"Analyzing {len(all_pairs)} potential pairs")
        
        # Use parallel processing if enabled
        if use_parallel and len(all_pairs) > 100:
            # Set up parallel processing
            max_workers = multiprocessing.cpu_count() - 1
            
            # Create a pool of workers
            with multiprocessing.Pool(processes=max_workers) as pool:
                # Create a partial function with fixed parameters
                analyze_func = partial(self.analyze_pair, data=data, min_correlation=min_correlation)
                
                # Map the function to all pairs
                results = pool.map(analyze_func, all_pairs)
        else:
            # Sequential processing
            results = [self.analyze_pair(pair, data, min_correlation) for pair in all_pairs]
        
        # Filter out None results
        valid_pairs = [r for r in results if r is not None]
        
        # Add sector information
        for pair in valid_pairs:
            stock1 = pair['stock1']
            stock2 = pair['stock2']
            
            pair['sector1'] = sector_mapping.get(stock1, 'Unknown')
            pair['sector2'] = sector_mapping.get(stock2, 'Unknown')
        
        # Sort by p-value
        valid_pairs = sorted(valid_pairs, key=lambda x: x['coint_pvalue'])
        
        # Categorize pairs
        result = {
            'within_sector': {},
            'cross_sector': []
        }
        
        # Initialize sector categories
        sectors = set(sector_mapping.values())
        for sector in sectors:
            result['within_sector'][sector] = []
        
        # Categorize pairs
        for pair in valid_pairs:
            if pair['sector1'] == pair['sector2']:
                result['within_sector'][pair['sector1']].append(pair)
            else:
                result['cross_sector'].append(pair)
        
        # Save to database
        self.db_manager.save_pairs(valid_pairs)
        
        return result
    
    def run_analysis(self):
        """
        Run the full pair analysis pipeline
        
        Returns:
            dict: Dictionary of cointegrated pairs by sector
        """
        # Get recent data from database
        data = self.db_manager.get_stock_data(
            start_date=(pd.Timestamp.now() - pd.Timedelta(days=180)).strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print("No data available for analysis")
            return None
        
        # Get sector mapping
        query = "SELECT ticker, sector FROM stock_sectors"
        conn = self.db_manager.get_connection()
        sector_df = pd.read_sql_query(query, conn)
        conn.close()
        
        sector_mapping = dict(zip(sector_df['ticker'], sector_df['sector']))
        
        # Analyze pairs
        return self.analyze_pairs(data, sector_mapping)

# ------ Signal Generator ------
class SignalGenerator:
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    def calculate_spread_zscore(self, data, pair, window=20):
        """
        Calculates the z-score of the spread between two stocks
        
        Parameters:
            data (DataFrame): Historical price data
            pair (dict): Stock pair information
            window (int): Rolling window size for z-score calculation
            
        Returns:
            DataFrame: Z-scores for the spread
        """
        stock1 = pair['stock1']
        stock2 = pair['stock2']
        hedge_ratio = pair['hedge_ratio']
        
        # Skip if data is missing
        if stock1 not in data.columns or stock2 not in data.columns:
            return pd.DataFrame()
            
        # Calculate spread
        spread = data[stock2] - hedge_ratio * data[stock1]
        
        # Calculate rolling mean and standard deviation
        mean = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std()
        
        # Calculate z-score
        zscore = (spread - mean) / std
        
        return pd.DataFrame({
            'spread': spread,
            'mean': mean,
            'std': std,
            'zscore': zscore
        })
    
    def generate_signals(self, entry_threshold=2.0, exit_threshold=0.0):
        """
        Generate trading signals for all pairs
        
        Parameters:
            entry_threshold (float): Entry threshold in standard deviations
            exit_threshold (float): Exit threshold in standard deviations
            
        Returns:
            list: List of trade signals
        """
        # Get pairs from database
        pairs = self.db_manager.get_pairs()
        
        if not pairs:
            print("No pairs available for signal generation")
            return []
        
        # Get stock data
        tickers = set()
        for pair in pairs:
            tickers.add(pair['stock1'])
            tickers.add(pair['stock2'])
        
        data = self.db_manager.get_stock_data(
            tickers=list(tickers),
            start_date=(pd.Timestamp.now() - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
        )
        
        if data.empty:
            print("No price data available for signal generation")
            return []
        
        # Check active signals that may need to be closed
        active_signals = self.db_manager.get_active_signals()
        
        for signal in active_signals:
            spread_data = self.calculate_spread_zscore(data, signal)
            
            if spread_data.empty or spread_data['zscore'].isnull().all():
                continue
            
            latest_zscore = spread_data['zscore'].iloc[-1]
            
            # Check if signal should be closed
            if abs(latest_zscore) <= exit_threshold:
                latest_spread = spread_data['spread'].iloc[-1]
                entry_spread = signal['entry_spread']
                
                # Calculate P&L
                if signal['direction'] == 'BUY':
                    profit_loss = latest_spread - entry_spread
                else:
                    profit_loss = entry_spread - latest_spread
                
                # Update signal in database
                self.db_manager.update_trading_signal(
                    signal['id'],
                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    latest_spread,
                    profit_loss,
                    'closed'
                )
                
                print(f"Closed signal for {signal['stock1']}/{signal['stock2']} with P&L: {profit_loss:.2f}")
        
        # Generate new signals
        new_signals = []
        
        for pair in pairs:
            spread_data = self.calculate_spread_zscore(data, pair)
            
            if spread_data.empty or spread_data['zscore'].isnull().all():
                continue
            
            latest_zscore = spread_data['zscore'].iloc[-1]
            
            # Check if there's a trading opportunity
            if abs(latest_zscore) >= entry_threshold:
                direction = "BUY" if latest_zscore <= -entry_threshold else "SELL"
                
                current_spread = spread_data['spread'].iloc[-1]
                mean_spread = spread_data['mean'].iloc[-1]
                
                # Create signal
                signal = {
                    'pair_id': pair['id'],
                    'direction': direction,
                    'zscore': latest_zscore,
                    'entry_spread': current_spread,
                    'target_spread': mean_spread,
                    'status': 'active'
                }
                
                # Save to database
                signal_id = self.db_manager.save_trading_signal(signal)
                
                # Add to results
                signal['id'] = signal_id
                signal
