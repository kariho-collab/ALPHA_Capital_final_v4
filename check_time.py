import MetaTrader5 as mt5
from datetime import datetime, timezone

if mt5.initialize():
    tick = mt5.symbol_info_tick("EURUSD")
    if tick:
        server_time = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        local_time = datetime.now(timezone.utc)
        print(f"MT5 Server Time (UTC): {server_time}")
        print(f"Local System Time (UTC): {local_time}")
    mt5.shutdown()
else:
    print("Failed to initialize MT5 connection.")