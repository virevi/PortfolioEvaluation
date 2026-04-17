import yfinance as yf
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static')

def format_crores_int(num):
    if num is None or num == 0: return "-"
    crores = int(round(num / 10000000.0))
    # Format with Indian style commas
    return "{:,}".format(crores)

@app.route('/api/prices')
def api_prices():
    symbols = request.args.get("symbols")
    if not symbols:
        return jsonify({"data": []})
    unpacked = [s.strip() for s in symbols.split(",") if s.strip()]
    if not unpacked:
        return jsonify({"data": []})
    data = []
    for sym in unpacked:
        ticker = sym + ".NS"
        try:
            yf_obj = yf.Ticker(ticker)
            price = yf_obj.info.get('regularMarketPrice', None)
            data.append({"Stock": sym, "CurrentPrice": price if price else "-", "Ticker": ticker})
        except Exception:
            data.append({"Stock": sym, "CurrentPrice": "-", "Ticker": ticker})
    return jsonify({"data": data})

@app.route('/')
def root():
    return send_from_directory('static', 'index B1.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
