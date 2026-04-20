import os
import io
import pandas as pd
import numpy as np
import yfinance as yf
import requests               # <--- For connecting to Google
from bs4 import BeautifulSoup # <--- For reading Google's HTML
import concurrent.futures
import sqlite3
from flask import Flask, request, jsonify, send_from_directory
from scipy.optimize import newton

app = Flask(__name__, static_folder='static')
DB_PATH = 'portfolio.db'

# --- Helper: Newton-Raphson XIRR ---
def calculate_xirr(cashflows):
    """Cashflows is a list of tuples: (Date, Amount)"""
    cashflows = sorted(cashflows, key=lambda x: x[0])
    if not cashflows: return None
    t0 = cashflows[0][0]
    
    def xnpv(rate):
        return sum([cf / (1 + rate)**((t - t0).days / 365.0) for t, cf in cashflows])
    
    try:
        rate = newton(xnpv, 0.1) 
        return rate if np.isfinite(rate) else None
    except (RuntimeError, OverflowError):
        return None

@app.route('/')
def root():
    return send_from_directory('static', 'index V4.html')

@app.route('/api/upload', methods=['POST'])
def upload_data():
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # 1. Handle Transactions Sync (Overwrite Mode)
        if 'transactions' in request.files and request.files['transactions'].filename != '':
            df_tx = pd.read_csv(request.files['transactions'])
            # We removed the merge logic. Now it just completely overwrites the old table!
            df_tx.to_sql('transactions', conn, if_exists='replace', index=False)

        # 2. Handle Prices Sync (Overwrite Mode)
        if 'prices' in request.files and request.files['prices'].filename != '':
            df_px = pd.read_csv(request.files['prices'])
            # We removed the merge logic here too
            df_px.to_sql('prices', conn, if_exists='replace', index=False)
        
        # 3. Metadata Sync (NEW)
        if 'metadata' in request.files and request.files['metadata'].filename != '':
            df_meta = pd.read_csv(request.files['metadata'])
            df_meta.to_sql('metadata', conn, if_exists='replace', index=False)

        conn.close()
        return jsonify({"message": "Database successfully synced!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fetch_dividends', methods=['POST'])
def fetch_dividends():
    try:
        conn = sqlite3.connect(DB_PATH)
        # 1. Find all unique stocks you've ever owned
        df_tx = pd.read_sql("SELECT DISTINCT ScripName FROM transactions", conn)
        tickers = df_tx['ScripName'].tolist()
        
        div_records = []
        for scrip in tickers:
            # Format ticker for Indian market
            symbol = f"{scrip}.NS" if not scrip.endswith('.NS') and not scrip.endswith('.BO') else scrip
            try:
                stock = yf.Ticker(symbol)
                divs = stock.dividends
                if not divs.empty:
                    for date, amount in divs.items():
                        # Strip timezones and save exact Ex-Dividend dates
                        clean_date = date.tz_localize(None).strftime('%Y-%m-%d')
                        div_records.append({
                            "ScripName": scrip,
                            "Date": clean_date,
                            "Dividend": float(amount)
                        })
            except Exception as e:
                print(f"Skipping {scrip}: {e}")
        
        # 2. Save permanently to database
        if div_records:
            df_divs = pd.DataFrame(div_records)
            df_divs.to_sql('dividends', conn, if_exists='replace', index=False)
            msg = f"Success! Fetched {len(div_records)} historical dividend events."
        else:
            msg = "No dividends found for these tickers (or ETFs)."
            
        conn.close()
        return jsonify({"message": msg})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_portfolio():
    # Attempt to read from the SQLite database
    try:
        conn = sqlite3.connect(DB_PATH)
        df_tx = pd.read_sql("SELECT * FROM transactions", conn)
        df_px = pd.read_sql("SELECT * FROM prices", conn)
        conn.close()
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        return jsonify({"error": "Database is empty. Please sync your CSV files using the Database Sync panel."}), 400

    if df_tx.empty or df_px.empty:
        return jsonify({"error": "Database tables are empty. Please sync your CSV files."}), 400

    # 2. Clean Data
    df_tx['Date'] = pd.to_datetime(df_tx['Date'], format='%d/%m/%Y')
    df_px['Date'] = pd.to_datetime(df_px['Date'], format='%d/%m/%Y')
    df_tx['ShareNos'] = pd.to_numeric(df_tx['ShareNos'], errors='coerce').fillna(0)
    df_tx['Price'] = pd.to_numeric(df_tx['Price'], errors='coerce').fillna(0)
    df_px['Close Price'] = pd.to_numeric(df_px['Close Price'], errors='coerce').fillna(0)
    
    # Ensure Demat column exists
    if 'Demat' not in df_tx.columns: df_tx['Demat'] = 'Unknown'

    # Cashflow: Buy is negative cash out, Sell is positive cash in
    df_tx['Cashflow'] = -1 * df_tx['ShareNos'] * df_tx['Price']

    # 3. Extract Unique Options for Dropdowns (Unfiltered)
    folios_list = sorted(df_tx['Folio'].dropna().unique().tolist())
    
    # Inject our virtual 'SmallCase' folio into the dropdown list
    if 'SmallCase' not in folios_list:
        folios_list.append('SmallCase')
        folios_list.sort() # Optional: keeps the dropdown alphabetical

    filters_options = {
        "folios": folios_list,
        "investors": sorted(df_tx['Investor'].dropna().unique().tolist()),
        "demats": sorted(df_tx['Demat'].dropna().unique().tolist())
    }

    # 4. Apply User Filters
    f_folio = request.form.get('folio', 'All')
    f_investor = request.form.get('investor', 'All')
    f_demat = request.form.get('demat', 'All')

    if f_folio == 'SmallCase':
        # If SmallCase is selected, filter for transactions in any of these 3 folios
        df_tx = df_tx[df_tx['Folio'].isin(['NiCT', 'NiMS', 'NiGE'])]
    elif f_folio != 'All': 
        # Standard behavior for any other individual folio
        df_tx = df_tx[df_tx['Folio'] == f_folio]
    if f_investor != 'All': df_tx = df_tx[df_tx['Investor'] == f_investor]
    if f_demat != 'All': df_tx = df_tx[df_tx['Demat'] == f_demat]

    if df_tx.empty:
        return jsonify({"error": "No transactions match these filters.", "filters": filters_options})

    # 5. Core Calculations (on filtered data)
    fifo_costs = {}
    realized_gains = {}
    residual_shares = {}
    active_lots = {} # Tracks the exact active lots for tax purposes
    
    # Process transactions chronologically to match Sells with oldest Buys
    for scrip in df_tx['ScripName'].unique():
        scrip_tx = df_tx[df_tx['ScripName'] == scrip].sort_values(by='Date', ascending=True)
        
        buy_queue = []
        realized_pnl = 0.0
        
        for _, row in scrip_tx.iterrows():
            action = str(row['Transaction']).strip().upper()
            qty = abs(float(row['ShareNos']))
            price = float(row['Price'])
            
            if action == 'BUY':
                # Add to queue
                buy_queue.append({'date': row['Date'], 'qty': qty, 'price': price})
            elif action in ['SELL', 'SALE']:
                qty_to_sell = qty
                # Match sell against oldest available buys (FIFO)
                while qty_to_sell > 0 and buy_queue:
                    oldest_buy = buy_queue[0]
                    if oldest_buy['qty'] <= qty_to_sell:
                        # Consume entire buy lot
                        matched_qty = oldest_buy['qty']
                        realized_pnl += matched_qty * (price - oldest_buy['price'])
                        qty_to_sell -= matched_qty
                        buy_queue.pop(0)
                    else:
                        # Consume partial buy lot
                        matched_qty = qty_to_sell
                        realized_pnl += matched_qty * (price - oldest_buy['price'])
                        oldest_buy['qty'] -= matched_qty
                        qty_to_sell = 0
                        
        realized_gains[scrip] = realized_pnl

        # Save the exact remaining lots for the Tax Harvester
        active_lots[scrip] = buy_queue
        
        # What remains in the queue is the active residual position
        residual_qty = sum(item['qty'] for item in buy_queue)
        residual_cost = sum(item['qty'] * item['price'] for item in buy_queue)
        
        residual_shares[scrip] = residual_qty
        fifo_costs[scrip] = residual_cost / residual_qty if residual_qty > 0 else 0

    # Build holdings dataframe from calculated FIFO dictionaries
    holdings_summary = pd.DataFrame({
        'ScripName': list(residual_shares.keys()),
        'TotalShares': list(residual_shares.values()),
        'AvgBuyPrice': [fifo_costs[k] for k in residual_shares.keys()],
        'RealizedGain': [realized_gains[k] for k in residual_shares.keys()]
    })
    
    # Filter to show only active positions
    current_holdings = holdings_summary[holdings_summary['TotalShares'] > 0].copy()
    current_holdings['NetInvested'] = current_holdings['TotalShares'] * current_holdings['AvgBuyPrice']
    
    # Merge with Latest Prices for live valuation
    latest_prices = df_px.sort_values('Date').groupby('Stock').last().reset_index()
    current_holdings = current_holdings.merge(latest_prices, left_on='ScripName', right_on='Stock', how='left')
    
    current_holdings['CurrentValue'] = current_holdings['TotalShares'] * current_holdings['Close Price']

    # Merge Metadata for Sector & Market Cap
    try:
        conn = sqlite3.connect(DB_PATH)
        df_meta = pd.read_sql("SELECT * FROM metadata", conn)
        conn.close()
        
        # Make it bulletproof by forcing the exact capitalization we need
        rename_map = {col: 'MarketCap' for col in df_meta.columns if col.lower() == 'marketcap'}
        rename_map.update({col: 'Ticker' for col in df_meta.columns if col.lower() == 'ticker'})
        rename_map.update({col: 'Sector' for col in df_meta.columns if col.lower() == 'sector'})
        df_meta.rename(columns=rename_map, inplace=True)

        current_holdings = current_holdings.merge(df_meta, left_on='ScripName', right_on='Ticker', how='left')
        
        # Use simple assignment to avoid warnings
        current_holdings['Sector'] = current_holdings['Sector'].fillna('Unclassified')
        current_holdings['MarketCap'] = current_holdings['MarketCap'].fillna('Unclassified')
        
    except Exception as e:
        print(f"Metadata Merge Error: {e}") # This will print the actual error to your terminal so it doesn't fail silently!
        current_holdings['Sector'] = 'Unclassified'
        current_holdings['MarketCap'] = 'Unclassified'

    # Gain here represents Unrealized (Paper) Gain
    current_holdings['Gain'] = current_holdings['CurrentValue'] - current_holdings['NetInvested']
    current_holdings.fillna(0, inplace=True)

    # Group data for the Command Center Charts
    sector_dist = current_holdings.groupby('Sector')['CurrentValue'].sum().reset_index().to_dict('records')
    mcap_dist = current_holdings.groupby('MarketCap')['CurrentValue'].sum().reset_index().to_dict('records')

    # 6. Time Series for Chart
    px_pivot = df_px.pivot_table(index='Date', columns='Stock', values='Close Price').ffill()
    tx_pivot = df_tx.groupby(['Date', 'ScripName'])['ShareNos'].sum().reset_index().pivot_table(index='Date', columns='ScripName', values='ShareNos').fillna(0).cumsum()
    
    start_date = df_tx['Date'].min()
    end_date = df_px['Date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    px_pivot = px_pivot.reindex(all_dates).ffill().fillna(0)
    tx_pivot = tx_pivot.reindex(all_dates).ffill().fillna(0)
    
    common_stocks = list(set(px_pivot.columns) & set(tx_pivot.columns))
    daily_value = (px_pivot[common_stocks] * tx_pivot[common_stocks]).sum(axis=1)

    # NIFTY SIMULATION: What if you bought Nifty instead?
    # Capital Inflow: Positive when buying (adding money), Negative when selling
    df_tx['Capital_Inflow'] = df_tx['ShareNos'] * df_tx['Price']
    daily_inflows = df_tx.groupby('Date')['Capital_Inflow'].sum().reindex(all_dates).fillna(0)

    # --- Historical FIFO Cost Basis for Chart ---
    inventory = {}
    daily_cost_basis = {}
    current_total_cost = 0.0
    
    # Replay all transactions day-by-day to reconstruct the exact FIFO cost basis in history
    for date, daily_txs in df_tx.sort_values(by='Date').groupby('Date'):
        for _, row in daily_txs.iterrows():
            scrip = row['ScripName']
            action = str(row['Transaction']).strip().upper()
            qty = abs(float(row['ShareNos']))
            price = float(row['Price'])
            
            if scrip not in inventory:
                inventory[scrip] = []
                
            if action == 'BUY':
                inventory[scrip].append({'qty': qty, 'price': price})
                current_total_cost += (qty * price)
            elif action in ['SELL', 'SALE']:
                qty_to_sell = qty
                # Match sell against oldest available buys (FIFO)
                while qty_to_sell > 0 and inventory[scrip]:
                    oldest_buy = inventory[scrip][0]
                    if oldest_buy['qty'] <= qty_to_sell:
                        matched_qty = oldest_buy['qty']
                        current_total_cost -= (matched_qty * oldest_buy['price'])
                        qty_to_sell -= matched_qty
                        inventory[scrip].pop(0)
                    else:
                        matched_qty = qty_to_sell
                        current_total_cost -= (matched_qty * oldest_buy['price'])
                        inventory[scrip][0]['qty'] -= matched_qty
                        qty_to_sell = 0
                        
        daily_cost_basis[date] = current_total_cost
        
    # Convert to a daily time series and forward-fill the gaps between trading days
    if daily_cost_basis:
        fifo_cost_series = pd.Series(daily_cost_basis)
        cumulative_invested = fifo_cost_series.reindex(all_dates).ffill().fillna(0)
    else:
        cumulative_invested = pd.Series(0, index=all_dates)
    # --------------------------------------------

    try:
        nifty = yf.download("^NSEI", start=start_date - pd.Timedelta(days=5), end=end_date + pd.Timedelta(days=1))
        nifty_close = nifty['Close'].squeeze() if isinstance(nifty.columns, pd.MultiIndex) else nifty['Close']
        nifty_reindexed = nifty_close.reindex(all_dates).bfill().ffill()
        
        # Simulate buying Nifty units with your daily cash flows
        nifty_units_bought = (daily_inflows / nifty_reindexed).fillna(0)
        cumulative_nifty_units = nifty_units_bought.cumsum()
        
        # Simulated Daily Nifty Portfolio Value
        simulated_nifty_value = cumulative_nifty_units * nifty_reindexed
    except Exception as e:
        print(f"Nifty Error: {e}")
        simulated_nifty_value = pd.Series(index=all_dates, data=0)

    # --- TODAY'S GAIN & TIMESTAMPS MODULE ---
    # 1. Get Timestamps
    last_tx_date = df_tx['Date'].max().strftime('%d-%b-%Y') if not df_tx.empty else 'N/A'
    last_px_date = df_px['Date'].max().strftime('%d-%b-%Y') if not df_px.empty else 'N/A'
    
    try:
        conn = sqlite3.connect(DB_PATH)
        df_div_meta = pd.read_sql("SELECT MAX(Date) as max_date FROM dividends", conn)
        conn.close()
        last_div_date = pd.to_datetime(df_div_meta['max_date'].iloc[0]).strftime('%d-%b-%Y') if not df_div_meta.empty and pd.notna(df_div_meta['max_date'].iloc[0]) else "Never Synced"
    except Exception:
        last_div_date = "Never Synced"

    # --- LIVE DATA IS NOW HANDLED ASYNCHRONOUSLY ---
    # We pass empty placeholders so the frontend loads instantly
    last_live_time = "Fetching..."
    todays_gain_data = []

    # --- ALPHA MODULE: Portfolio Beta Calculation ---
    try:
        # Calculate daily returns, stripping out cash deposits to find true volatility
        v_prev = daily_value.shift(1).fillna(0)
        port_returns = pd.Series(
            np.where(v_prev > 0, (daily_value - daily_inflows - v_prev) / v_prev, 0), 
            index=all_dates
        ).replace([np.inf, -np.inf], 0)
        
        nifty_returns = nifty_reindexed.pct_change().fillna(0)
        
        # Align dates and calculate Covariance / Variance
        aligned_returns = pd.concat([port_returns, nifty_returns], axis=1, join='inner').dropna()
        aligned_returns.columns = ['Portfolio', 'Nifty']
        
        if len(aligned_returns) > 30:
            covar = aligned_returns.cov().iloc[0, 1]
            nifty_var = aligned_returns['Nifty'].var()
            portfolio_beta = covar / nifty_var if nifty_var != 0 else 1.0
            if pd.isna(portfolio_beta): portfolio_beta = 1.0 # <--- FIX: Catch NaN Beta
        else:
            portfolio_beta = 1.0
            nifty_var = 0

        # Individual Stock Betas for the Risk Heatmap
        individual_betas = {}
        if len(aligned_returns) > 30 and nifty_var != 0:
            stock_returns = px_pivot.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
            for col in stock_returns.columns:
                valid_data = pd.concat([stock_returns[col], nifty_returns], axis=1).dropna()
                if len(valid_data) > 30:
                    scov = valid_data.cov().iloc[0, 1]
                    val = scov / nifty_var
                    individual_betas[col] = val if pd.notna(val) else 1.0 # <--- FIX: Catch NaN Beta

    except Exception as e:
        print(f"Beta Calculation Error: {e}")
        portfolio_beta = 1.0
        individual_betas = {}

    # --- ALPHA MODULE: Multi-Period Rolling Returns ---
    try:
        # Create true NAV indices (Base 100) to ignore cash deposits/withdrawals
        port_nav = 100 * (1 + port_returns).cumprod()
        nifty_nav = 100 * (1 + nifty_returns).cumprod()
        
        def calc_rolling_cagr(nav_series, days, years):
            if len(nav_series) > days:
                roll = ((nav_series / nav_series.shift(days)) ** (1/years)) - 1
                valid_mask = roll.notna()
                dates = roll[valid_mask].index.strftime('%Y-%m-%d').tolist()
                vals = (roll[valid_mask] * 100).round(2).tolist()
                return dates, vals
            return [], []

        # Calculate 2Y, 3Y, and 5Y Rolling CAGRs
        r_dates_2y, r_port_2y = calc_rolling_cagr(port_nav, 730, 2)
        _, r_nifty_2y = calc_rolling_cagr(nifty_nav, 730, 2)

        r_dates_3y, r_port_3y = calc_rolling_cagr(port_nav, 1095, 3)
        _, r_nifty_3y = calc_rolling_cagr(nifty_nav, 1095, 3)

        r_dates_5y, r_port_5y = calc_rolling_cagr(port_nav, 1825, 5)
        _, r_nifty_5y = calc_rolling_cagr(nifty_nav, 1825, 5)

        rolling_data = {
            "2Y": {"dates": r_dates_2y, "port": r_port_2y, "nifty": r_nifty_2y},
            "3Y": {"dates": r_dates_3y, "port": r_port_3y, "nifty": r_nifty_3y},
            "5Y": {"dates": r_dates_5y, "port": r_port_5y, "nifty": r_nifty_5y}
        }
    except Exception as e:
        print(f"Rolling Returns Error: {e}")
        rolling_data = {"2Y": {}, "3Y": {}, "5Y": {}}
    # ----------------------------------------------

    chart_dates = all_dates.strftime('%Y-%m-%d').tolist()
    chart_portfolio_abs = daily_value.fillna(0).tolist()
    chart_nifty_abs = simulated_nifty_value.fillna(0).tolist()

    # 7. XIRR & Reports Logic
    total_current_value = current_holdings['CurrentValue'].sum()
    cashflows = list(zip(df_tx['Date'], df_tx['Cashflow']))
    cashflows.append((end_date, total_current_value))
    portfolio_xirr = calculate_xirr(cashflows)

    # Extract 30-Day Trend Data for Sparklines
    last_30_dates = all_dates[-30:] if len(all_dates) >= 30 else all_dates
    trend_data = px_pivot.reindex(last_30_dates).ffill().fillna(0)

    holdings_list = []
    for index, row in current_holdings.iterrows():
        scrip = row['ScripName']
        # Grab the 30-day price array for this specific stock
        trend = trend_data[scrip].tolist() if scrip in trend_data.columns else []
        holdings_list.append({
            "ScripName": scrip,
            "Sector": row.get('Sector', 'Unclassified'),
            "MarketCap": row.get('MarketCap', 'Unclassified'),
            "TotalShares": row['TotalShares'],
            "AvgBuyPrice": row.get('AvgBuyPrice', 0),
            "Close Price": row['Close Price'],
            "NetInvested": row['NetInvested'],
            "CurrentValue": row['CurrentValue'],
            "UnrealizedGain": row['Gain'],
            "RealizedGain": row.get('RealizedGain', 0),
            "Trend": trend,
            "Beta": float(individual_betas.get(scrip, 1.0))
        })
        
    # Helper for Reports Tab
    def build_report(group_col):
        report_data = []
        for name, group in df_tx.groupby(group_col):
            inv = -group['Cashflow'].sum()
            
            # Find current value for this specific group's holdings
            grp_shares = group.groupby('ScripName')['ShareNos'].sum()
            val = 0
            for scrip, shares in grp_shares.items():
                if shares > 0:
                    px = latest_prices[latest_prices['Stock'] == scrip]['Close Price'].max()
                    val += shares * (px if pd.notna(px) else 0)
            
            # Group XIRR
            grp_cfs = list(zip(group['Date'], group['Cashflow']))
            grp_cfs.append((end_date, val))
            gxirr = calculate_xirr(grp_cfs)
            
            report_data.append({
                "Group": str(name),
                "Invested": float(inv),
                "CurrentValue": float(val),
                "Gain": float(val - inv),
                "XIRR": float(gxirr * 100) if gxirr else None
            })
        return report_data

    # --- DEMAT ALLOCATION MATRIX LOGIC ---
    demat_matrix_data = {}
    for name, group in df_tx.groupby(['ScripName', 'Demat']):
        scrip, demat = name
        group = group.sort_values(by='Date', ascending=True)
        
        buy_queue = []
        for _, row in group.iterrows():
            action = str(row['Transaction']).strip().upper()
            qty = abs(float(row['ShareNos']))
            price = float(row['Price'])
            
            if action == 'BUY':
                buy_queue.append({'qty': qty, 'price': price})
            elif action in ['SELL', 'SALE']:
                qty_to_sell = qty
                while qty_to_sell > 0 and buy_queue:
                    oldest_buy = buy_queue[0]
                    if oldest_buy['qty'] <= qty_to_sell:
                        qty_to_sell -= oldest_buy['qty']
                        buy_queue.pop(0)
                    else:
                        oldest_buy['qty'] -= qty_to_sell
                        qty_to_sell = 0
                        
        residual_qty = sum(item['qty'] for item in buy_queue)
        if residual_qty > 0:
            residual_cost = sum(item['qty'] * item['price'] for item in buy_queue)
            avg_price = residual_cost / residual_qty
            
            if scrip not in demat_matrix_data:
                demat_matrix_data[scrip] = {"ScripName": scrip}
            
            # Map the exact shares and average cost to this specific Demat's column keys
            demat_matrix_data[scrip][f"{demat}_Shares"] = residual_qty
            demat_matrix_data[scrip][f"{demat}_Avg"] = avg_price

    matrix_list = list(demat_matrix_data.values())
    demats_list = sorted(df_tx['Demat'].dropna().unique().tolist())


    # --- SUNBURST CHART LOGIC (Investor -> Sector) ---
    try:
        # Group raw shares by Investor and Ticker
        inv_scrip_shares = df_tx.groupby(['Investor', 'ScripName'])['ShareNos'].sum().reset_index()
        inv_scrip_shares = inv_scrip_shares[inv_scrip_shares['ShareNos'] > 0]
        
        # Merge with Live Prices and Metadata
        inv_scrip_shares = inv_scrip_shares.merge(latest_prices, left_on='ScripName', right_on='Stock', how='left')
        inv_scrip_shares = inv_scrip_shares.merge(df_meta, left_on='ScripName', right_on='Ticker', how='left')
        
        # Calculate Value and group by Investor -> Sector
        inv_scrip_shares['CurrentValue'] = inv_scrip_shares['ShareNos'] * inv_scrip_shares['Close Price']
        if 'Sector' not in inv_scrip_shares.columns:
            inv_scrip_shares['Sector'] = 'Unclassified'
        inv_scrip_shares['Sector'] = inv_scrip_shares['Sector'].fillna('Unclassified')
        
        sunburst_raw = inv_scrip_shares.groupby(['Investor', 'Sector'])['CurrentValue'].sum().reset_index()
        sunburst_data = sunburst_raw.to_dict('records')
    except Exception as e:
        print(f"Sunburst Error: {e}")
        sunburst_data = []

    # --- RISK CORRELATION MATRIX ---
    try:
        # 1. Isolate the top 12 holdings by Current Value
        top_holdings = current_holdings.sort_values('CurrentValue', ascending=False).head(12)
        top_stocks = top_holdings['ScripName'].tolist()
        
        # 2. Ensure they exist in our historical price pivot table
        valid_stocks = [s for s in top_stocks if s in px_pivot.columns]
        
        if len(px_pivot) > 30 and len(valid_stocks) > 1:
            # 3. Calculate daily percentage returns
            top_px = px_pivot[valid_stocks].ffill()
            returns = top_px.pct_change().replace([np.inf, -np.inf], 0).dropna()
            
            # 4. Calculate the correlation matrix
            corr_matrix = returns.corr().round(2).fillna(0)
            
            # 5. Format for Plotly (Lists of Lists)
            corr_data = {
                "tickers": corr_matrix.columns.tolist(),
                "z_values": corr_matrix.values.tolist()
            }
        else:
            corr_data = None
    except Exception as e:
        print(f"Correlation Error: {e}")
        corr_data = None

    # --- TAX-LOSS HARVESTER LOGIC (Strict FIFO Enforced) ---
    try:
        tax_lots = []
        stcl_total = 0.0
        ltcl_total = 0.0
        analysis_date = end_date # Anchor to the latest price date
        
        for index, row in current_holdings.iterrows():
            scrip = row['ScripName']
            ltp = row['Close Price']
            
            if scrip in active_lots and ltp > 0:
                # STRICT FIFO RULE: We must evaluate shares strictly from oldest to newest.
                for lot in active_lots[scrip]:
                    if lot['qty'] <= 0:
                        continue
                        
                    if ltp < lot['price']: 
                        # This lot is at a loss. Because it's at the front of the queue, we can safely harvest it!
                        loss_per_share = lot['price'] - ltp
                        total_loss = loss_per_share * lot['qty']
                        days_held = (analysis_date - lot['date']).days
                        term = "LT" if days_held > 365 else "ST"
                        
                        tax_lots.append({
                            "ScripName": scrip,
                            "BuyDate": lot['date'].strftime('%d-%b-%Y'),
                            "DaysHeld": int(days_held),
                            "Term": term,
                            "Qty": float(lot['qty']),
                            "BuyPrice": float(lot['price']),
                            "LTP": float(ltp),
                            "TotalLoss": float(-total_loss) # Send as a negative number
                        })
                        
                        if term == "ST":
                            stcl_total -= total_loss
                        else:
                            ltcl_total -= total_loss
                    else:
                        # THE FIFO WALL TRIGGERED!
                        # This lot is currently in profit. We cannot legally reach any newer losing shares 
                        # behind it without selling this one first and paying Capital Gains Tax.
                        # Therefore, we MUST STOP scanning this stock to protect the user.
                        break 
                            
        # Sort by biggest loss (most negative) at the top
        tax_lots = sorted(tax_lots, key=lambda x: x['TotalLoss'])
        tax_data = {"summary": {"STCL": stcl_total, "LTCL": ltcl_total}, "lots": tax_lots}
    except Exception as e:
        print(f"Tax Harvester Error: {e}")
        tax_data = {"summary": {"STCL": 0, "LTCL": 0}, "lots": []}

    # --- DIVIDEND TIME MACHINE LOGIC ---
    try:
        conn = sqlite3.connect(DB_PATH)
        df_divs = pd.read_sql("SELECT * FROM dividends", conn)
        conn.close()
        
        df_divs['Date'] = pd.to_datetime(df_divs['Date'])
        total_collected = 0.0
        yearly_dividends = {}
        
        # Calculate recent annual dividend per stock to find "Yield on Cost"
        one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
        recent_divs = df_divs[df_divs['Date'] >= one_year_ago]
        annual_div_per_stock = recent_divs.groupby('ScripName')['Dividend'].sum().to_dict()
        
        # Match shares held on the exact Ex-Dividend Date
        for _, div_row in df_divs.iterrows():
            scrip = div_row['ScripName']
            ex_date = div_row['Date']
            amount = div_row['Dividend']
            
            # tx_pivot holds our daily cumulative share balance
            if ex_date in tx_pivot.index and scrip in tx_pivot.columns:
                shares_held = tx_pivot.at[ex_date, scrip]
                if shares_held > 0:
                    payout = shares_held * amount
                    total_collected += payout
                    
                    year = str(ex_date.year)
                    yearly_dividends[year] = yearly_dividends.get(year, 0) + payout
                    
        div_chart = [{"Year": y, "Amount": float(a)} for y, a in sorted(yearly_dividends.items())]
        
        # Inject Yield on Cost into the Live Positions table
        for holding in holdings_list:
            scrip = holding['ScripName']
            avg_cost = holding['AvgBuyPrice']
            ann_div = annual_div_per_stock.get(scrip, 0.0)
            holding['YieldOnCost'] = (ann_div / avg_cost * 100) if avg_cost > 0 else 0.0
            
        div_summary = {"TotalCollected": float(total_collected), "History": div_chart}
    except Exception as e:
        print(f"Dividend Logic Error: {e}")
        div_summary = {"TotalCollected": 0.0, "History": []}
        for holding in holdings_list: holding['YieldOnCost'] = 0.0

    # 8. Package Response (Update Chart variables)
    response = {
            "meta": {                          
                "last_tx_date": last_tx_date,
                "last_px_date": last_px_date,
                "last_div_date": last_div_date,
                "last_live_time": last_live_time
            },
            "todays_gain": todays_gain_data,
            "filters_options": filters_options,
            "summary": {
                "TotalInvested": float(current_holdings['NetInvested'].sum()),
                "CurrentValue": float(total_current_value),
                "Gain": float(current_holdings['Gain'].sum()),
                "XIRR": float(portfolio_xirr * 100) if portfolio_xirr else None,
                "Beta": float(portfolio_beta)
            },
            "holdings": holdings_list,
            "transactions": df_tx[['Date', 'ScripName', 'Transaction', 'ShareNos', 'Price', 'Folio', 'Investor', 'Demat']].assign(Date=df_tx['Date'].dt.strftime('%Y-%m-%d')).to_dict(orient='records'),
            "reports": {
                "folio": build_report('Folio'),
                "investor": build_report('Investor'),
                "demat": build_report('Demat'),
                "stock_demat_matrix": matrix_list,
                "demat_names": demats_list          
            },
            "chart": {
                "dates": chart_dates,
                "portfolio_abs": chart_portfolio_abs,
                "nifty_abs": chart_nifty_abs,
                "invested_abs": cumulative_invested.fillna(0).tolist(),
                "rolling": rolling_data
            },
            "sector_dist": sector_dist,
            "mcap_dist": mcap_dist,
            "sunburst_data": sunburst_data,
            "correlation": corr_data,
            "tax_harvesting": tax_data,
            "dividends": div_summary
        }
    
    return jsonify(response)

@app.route('/api/live_prices', methods=['POST'])
def get_live_prices():
    try:
        conn = sqlite3.connect(DB_PATH)
        df_tx = pd.read_sql("SELECT * FROM transactions", conn)
        df_px = pd.read_sql("SELECT * FROM prices", conn)
        conn.close()
    except Exception as e:
        return jsonify({"error": "Database error"}), 500

    df_tx['Date'] = pd.to_datetime(df_tx['Date'], format='%d/%m/%Y')
    df_tx['ShareNos'] = pd.to_numeric(df_tx['ShareNos'], errors='coerce').fillna(0)
    
    # 1. Apply user filters
    f_folio = request.form.get('folio', 'All')
    f_investor = request.form.get('investor', 'All')
    f_demat = request.form.get('demat', 'All')

    if f_folio == 'SmallCase':
        df_tx = df_tx[df_tx['Folio'].isin(['NiCT', 'NiMS', 'NiGE'])]
    elif f_folio != 'All': 
        df_tx = df_tx[df_tx['Folio'] == f_folio]
    if f_investor != 'All': df_tx = df_tx[df_tx['Investor'] == f_investor]
    if f_demat != 'All': df_tx = df_tx[df_tx['Demat'] == f_demat]

    # 2. Get currently active tickers
    folio_holdings = df_tx.groupby(['Folio', 'ScripName'])['ShareNos'].sum().reset_index()
    folio_holdings = folio_holdings[folio_holdings['ShareNos'] > 0]
    active_tickers = folio_holdings['ScripName'].unique().tolist()
    
    # --- GOOGLE FINANCE PARALLEL SCRAPER ---
    import concurrent.futures
    import requests
    from bs4 import BeautifulSoup
    
    live_ltp = {}
    live_changes = {}
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def fetch_google_price(scrip):
        clean_symbol = scrip.replace('.NS', '').replace('.BO', '')
        GOOGLE_TICKER_MAP = {
            "EMBASSY": "542602:BOM",
            "MINDSPACE": "543217:BOM",  
            "BIRET": "542660:BOM"
        }
        if clean_symbol in GOOGLE_TICKER_MAP:
            google_query = GOOGLE_TICKER_MAP[clean_symbol]
        else:
            exchange = "BOM" if scrip.endswith('.BO') else "NSE"
            google_query = f"{clean_symbol}:{exchange}"
            
        url = f"https://www.google.com/finance/quote/{google_query}"
        
        try:
            response = requests.get(url, headers=HEADERS, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                price_div = soup.find(class_="YMlKec fxKbKc")
                if not price_div: return scrip, None, None, None
                ltp = float(price_div.text.replace('₹', '').replace(',', '').strip())
                
                prev_close = None
                stat_blocks = soup.find_all(class_="gyFHrc")
                for block in stat_blocks:
                    if 'Previous close' in block.text:
                        val_div = block.find(class_="P6K39c")
                        if val_div:
                            prev_close = float(val_div.text.replace('₹', '').replace(',', '').strip())
                        break
                
                # --- Extract Google's Exact Market Timestamp ---
                market_time = None
                time_div = soup.find(class_="ygUjEc")
                if time_div:
                    # Grabs "Apr 20, 3:30:00 PM UTC+5:30" and strips away the " · INR"
                    market_time = time_div.text.split(' · ')[0].strip() 
                else:
                    # Bulletproof Fallback: Search for any div containing "UTC"
                    for div in soup.find_all('div'):
                        if 'UTC' in div.text and len(div.text) < 40:
                            market_time = div.text.split(' · ')[0].strip()
                            break

                return scrip, ltp, prev_close, market_time
        except Exception: pass
        return scrip, None, None, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        results = executor.map(fetch_google_price, active_tickers)
        
    # Get offline prices for fallback
    df_px['Date'] = pd.to_datetime(df_px['Date'], format='%d/%m/%Y')
    df_px['Close Price'] = pd.to_numeric(df_px['Close Price'], errors='coerce').fillna(0)
    latest_prices = df_px.sort_values('Date').groupby('Stock').last()
    
    valid_market_times = [] # <--- NEW: Array to store successful Google timestamps
    
    for scrip, ltp, prev_close, market_time in results: # <--- NEW: Unpack 4 variables
        
        if market_time:
            valid_market_times.append(market_time)

        try: offline_px = float(latest_prices.loc[scrip, 'Close Price'])
        except KeyError: offline_px = 0.0

        if ltp is not None:
            live_ltp[scrip] = ltp
            if prev_close is not None: live_changes[scrip] = ltp - prev_close
            else: live_changes[scrip] = ltp - offline_px
        else:
            # Fallback to Yahoo if Google is blocked
            try:
                yf_symbol = f"{scrip}.NS" if not scrip.endswith('.NS') and not scrip.endswith('.BO') else scrip
                ticker_live = yf.download(yf_symbol, period="5d", progress=False)
                if not ticker_live.empty:
                    close_col = ticker_live['Close']
                    px_series = close_col.iloc[:, 0].dropna() if isinstance(close_col, pd.DataFrame) else close_col.dropna()
                    if len(px_series) >= 2:
                        live_changes[scrip] = float(px_series.iloc[-1] - px_series.iloc[-2])
                        live_ltp[scrip] = float(px_series.iloc[-1])
                    else:
                        live_changes[scrip] = 0.0
                        live_ltp[scrip] = float(px_series.iloc[-1])
                else:
                    live_ltp[scrip] = offline_px
                    live_changes[scrip] = 0.0
            except Exception:
                live_ltp[scrip] = offline_px
                live_changes[scrip] = 0.0

    # --- NEW: Inject True Market Time instead of Local Computer Time ---
    if valid_market_times:
        # Just grab the first successful timestamp fetched from Google
        last_live_time = valid_market_times[0] + " (Google)" 
    else:
        # Absolute fallback if no internet
        last_live_time = pd.Timestamp.now(tz='Asia/Kolkata').strftime('%d-%b-%Y %I:%M %p') + " (Local Time)"
    
    # 3. Calculate Final Gain Data
    folio_holdings['DayChange'] = folio_holdings['ScripName'].map(live_changes).fillna(0)
    folio_holdings['TodayGain'] = folio_holdings['ShareNos'] * folio_holdings['DayChange']
    folio_holdings['LTP'] = folio_holdings['ScripName'].map(live_ltp).fillna(0)
    
    todays_gain_data = []
    for folio, group in folio_holdings.groupby('Folio'):
        folio_total = group['TodayGain'].sum()
        group = group.sort_values(by='TodayGain', ascending=False)
        stocks = group[['ScripName', 'ShareNos', 'LTP', 'DayChange', 'TodayGain']].to_dict(orient='records')
        todays_gain_data.append({
            "Folio": str(folio),
            "TotalTodayGain": float(folio_total),
            "Stocks": stocks
        })
    todays_gain_data = sorted(todays_gain_data, key=lambda x: x['TotalTodayGain'], reverse=True)
    
    return jsonify({
        "todays_gain": todays_gain_data,
        "last_live_time": last_live_time
    })

@app.route('/api/fundamentals', methods=['POST'])
def run_screener():
    data = request.json
    tickers = data.get('tickers', [])
    
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    results = []
    
    def score_company(roe, debt_eq, margin, growth):
        score = 0
        # Ignore N/A values (None) in scoring
        if roe is not None:
            if roe > 15: score += 3
            elif roe > 10: score += 1
            
        if debt_eq is not None:
            if debt_eq < 50: score += 3
            elif debt_eq < 100: score += 1
            
        if margin is not None:
            if margin > 15: score += 2
            elif margin > 10: score += 1
            
        if growth is not None:
            if growth > 10: score += 2
            elif growth > 5: score += 1
            
        return score

    for ticker in tickers:
        symbol = f"{ticker}.NS" if not ticker.endswith('.NS') and not ticker.endswith('.BO') else ticker
        
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # --- DEEP SCRAPE ROE CALCULATION ---
            roe_raw = info.get('returnOnEquity')
            
            if not roe_raw:
                # Try lightweight fallback first
                net_income = info.get('netIncomeToCommon') or info.get('netIncome')
                equity = info.get('totalStockholderEquity') or info.get('totalEquity')
                
                if net_income and equity and equity != 0:
                    roe_raw = net_income / equity
                else:
                    # Deep Fallback: Scrape the actual annual financial statements
                    try:
                        fin = stock.financials
                        bs = stock.balance_sheet
                        if not fin.empty and not bs.empty:
                            # Get latest year Net Income
                            ni = fin.loc['Net Income'].dropna().iloc[0] if 'Net Income' in fin.index else None
                            # Get latest year Equity
                            eq = bs.loc['Stockholders Equity'].dropna().iloc[0] if 'Stockholders Equity' in bs.index else None
                            
                            if ni and eq and eq != 0:
                                roe_raw = ni / eq
                    except Exception as e:
                        pass # Valid for ETFs or missing data
            
            roe = (roe_raw * 100) if roe_raw is not None else None
            
            # --- Additional Metrics (Send None if missing instead of 0) ---
            debt_eq = info.get('debtToEquity')
            op_margin_raw = info.get('operatingMargins')
            op_margin = (op_margin_raw * 100) if op_margin_raw is not None else None
            
            rev_growth_raw = info.get('revenueGrowth')
            rev_growth = (rev_growth_raw * 100) if rev_growth_raw is not None else None
            
            pe = info.get('trailingPE') or info.get('forwardPE')
            eps = info.get('trailingEps') or info.get('forwardEps')
            
            quality_score = score_company(roe, debt_eq, op_margin, rev_growth)
            
            results.append({
                "Ticker": ticker,
                "ROE": float(roe) if roe is not None else None,
                "DebtToEquity": float(debt_eq) if debt_eq is not None else None,
                "OpMargin": float(op_margin) if op_margin is not None else None,
                "RevGrowth": float(rev_growth) if rev_growth is not None else None,
                "PE": float(pe) if pe is not None else None,
                "EPS": float(eps) if eps is not None else None,
                "QualityScore": int(quality_score)
            })
        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")
            
    # Sort by highest Quality Score
    results = sorted(results, key=lambda x: x['QualityScore'], reverse=True)
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=5000)