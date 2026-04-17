import os
import io
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, request, jsonify, send_from_directory
from scipy.optimize import newton

app = Flask(__name__, static_folder='static')

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
    return send_from_directory('static', 'index V3.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_portfolio():
    if 'transactions' not in request.files or 'prices' not in request.files:
        return jsonify({"error": "Missing CSV files"}), 400
    
    # 1. Load Data
    df_tx = pd.read_csv(request.files['transactions'])
    df_px = pd.read_csv(request.files['prices'])

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
    holdings_summary = df_tx.groupby('ScripName').agg(
        TotalShares=('ShareNos', 'sum'),
        NetInvested=('Cashflow', lambda x: -x.sum())
    ).reset_index()
    
    current_holdings = holdings_summary[holdings_summary['TotalShares'] > 0].copy()
    latest_prices = df_px.sort_values('Date').groupby('Stock').last().reset_index()
    current_holdings = current_holdings.merge(latest_prices, left_on='ScripName', right_on='Stock', how='left')
    
    current_holdings['CurrentValue'] = current_holdings['TotalShares'] * current_holdings['Close Price']
    current_holdings['Gain'] = current_holdings['CurrentValue'] - current_holdings['NetInvested']
    current_holdings.fillna(0, inplace=True)

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

    cumulative_invested = daily_inflows.cumsum()

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
        else:
            portfolio_beta = 1.0
    except Exception as e:
        print(f"Beta Calculation Error: {e}")
        portfolio_beta = 1.0

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
            "TotalShares": row['TotalShares'],
            "Close Price": row['Close Price'],
            "NetInvested": row['NetInvested'],
            "CurrentValue": row['CurrentValue'],
            "Gain": row['Gain'],
            "Trend": trend # <-- The new 30-day array
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

    # 8. Package Response (Update Chart variables)
    response = {
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
                "demat": build_report('Demat')
            },
            "chart": {
                "dates": chart_dates,
                "portfolio_abs": chart_portfolio_abs,
                "nifty_abs": chart_nifty_abs,
                "invested_abs": cumulative_invested.fillna(0).tolist()
            }
        }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=5000)