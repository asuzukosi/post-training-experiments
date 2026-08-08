# filename: stock_price_chart.py
import yfinance as yf
import matplotlib.pyplot as plt

# Define the stock symbols and YTD date range
symbols = ['NVDA', 'TSLA']
start_date = '2022-01-01'
end_date = '2022-12-31'

# Fetch the stock price data using yfinance
df = yf.download(symbols, start=start_date, end=end_date)['Close']

# Plotting the stock price change
plt.figure(figsize=(12, 6))
for symbol in symbols:
    plt.plot(df.index, df[symbol], label=symbol)
plt.title("Stock Price Change YTD")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()