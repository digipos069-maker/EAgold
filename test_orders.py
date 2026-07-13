import MetaTrader5 as mt5
from datetime import datetime, timedelta

if not mt5.initialize():
    print("initialize() failed")
    quit()

today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
deals = mt5.history_deals_get(today_start, datetime.now() + timedelta(days=1))

for d in deals:
    if d.entry == 1: # mt5.DEAL_ENTRY_OUT
        ticket = d.position_id
        pos_orders = mt5.history_orders_get(position=ticket)
        print(f"pos_id={ticket} has {len(pos_orders) if pos_orders else 0} orders.")
        if pos_orders:
            print(f"Order[0] magic={pos_orders[0].magic} type={pos_orders[0].type} tp={pos_orders[0].tp} sl={pos_orders[0].sl}")
        else:
            print("No pos_orders found!")
mt5.shutdown()
