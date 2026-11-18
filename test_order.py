import MetaTrader5 as mt5

# 1. Initialize connection to MT5
if not mt5.initialize():
    print(f"initialize() failed, error code = {mt5.last_error()}")
    quit()

print("Connected to MT5 successfully!")

# 2. Define symbol and lot size for testing
symbol = "EURUSD"
lot_size = 0.01

# Ensure symbol is selected in Market Watch
if not mt5.symbol_select(symbol, True):
    print(f"Failed to select {symbol}")
    mt5.shutdown()
    quit()

# 3. Get current price for a sell market orderpython test_order.py
tick = mt5.symbol_info_tick(symbol)
if tick is None:
    print(f"Failed to get tick info for {symbol}")
    mt5.shutdown()
    quit()

price = tick.bid  # Use Bid price for selling

# 4. Construct the order request
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lot_size,
    "type": mt5.ORDER_TYPE_SELL,
    "price": price,
    "deviation": 20,
    "magic": 234000,
    "comment": "Python API test order",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

# 5. Send the order
result = mt5.order_send(request)
print(f"Order send result: {result}")

if result.retcode != mt5.TRADE_RETCODE_DONE:
    print(f"Order failed, retcode={result.retcode}")
else:
    print("Test order executed successfully!")

# Clean up
mt5.shutdown()