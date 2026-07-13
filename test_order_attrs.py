import MetaTrader5 as mt5
from datetime import datetime, timedelta

if not mt5.initialize():
    print("init failed")
    quit()

today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
deals = mt5.history_deals_get(today_start, datetime.now() + timedelta(days=1))

if deals:
    for d in deals:
        if d.entry == 1:
            ticket = d.position_id
            pos_orders = mt5.history_orders_get(position=ticket)
            if pos_orders:
                print("First order attrs:", dir(pos_orders[0]))
                break
mt5.shutdown()
