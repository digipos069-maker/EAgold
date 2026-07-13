import MetaTrader5 as mt5
from datetime import datetime, timedelta

if not mt5.initialize():
    print("initialize() failed")
    quit()

today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
print(f"Fetching from {today_start}")
deals = mt5.history_deals_get(today_start, datetime.now() + timedelta(days=1))

if deals is None:
    print(f"No deals found, error: {mt5.last_error()}")
else:
    print(f"Found {len(deals)} deals today")
    for d in deals:
        if d.entry == 1: # mt5.DEAL_ENTRY_OUT
            print(f"OUT Deal: ticket={d.ticket} pos_id={d.position_id} symbol={d.symbol} profit={d.profit}")
mt5.shutdown()
