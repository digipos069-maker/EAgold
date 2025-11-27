import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime, timezone
import MetaTrader5 as mt5
import numpy as np

# --- Indicator Functions ---
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1: return np.array([])
    deltas = np.diff(prices)
    seed = deltas[:period]
    gains = seed[seed >= 0].sum() / period
    losses = -seed[seed < 0].sum() / period
    rs = gains / losses if losses != 0 else np.inf
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0: gain = delta; loss = 0
        else: gain = 0; loss = -delta
        gains = (gains * (period - 1) + gain) / period
        losses = (losses * (period - 1) + loss) / period
        rs = gains / losses if losses != 0 else np.inf
        rsi[i] = 100. - 100. / (1. + rs)
    return rsi

def calculate_ema(prices, period):
    if len(prices) < period: return None
    return np.mean(prices[-period:])

class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gold EA Trading Bot")
        self.root.geometry("800x800")
        self.root.configure(bg="#1e1e1e")

        self.autotrade_enabled = False
        self.strategy_thread = None
        self.last_trade_action = None

        self.timeframe_map = {
            "1 Minute (M1)": mt5.TIMEFRAME_M1, "5 Minutes (M5)": mt5.TIMEFRAME_M5,
            "15 Minutes (M15)": mt5.TIMEFRAME_M15, "1 Hour (H1)": mt5.TIMEFRAME_H1,
            "4 Hours (H4)": mt5.TIMEFRAME_H4,
        }
        self.sleep_intervals = {
            mt5.TIMEFRAME_M1: 60, mt5.TIMEFRAME_M5: 60, mt5.TIMEFRAME_M15: 300,
            mt5.TIMEFRAME_H1: 900, mt5.TIMEFRAME_H4: 1800,
        }

        self.create_widgets()
        self.start_mt5()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10", style="Main.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Main.TFrame", background="#1e1e1e")
        self.style.configure("TLabel", background="#1e1e1e", foreground="white", font=("Arial", 12))
        self.style.configure("TButton", foreground="white", font=("Arial", 12, "bold"))
        self.style.configure("Green.TButton", background="#4CAF50"); self.style.map("Green.TButton", background=[("active", "#45a049")])
        self.style.configure("Red.TButton", background="#f44336"); self.style.map("Red.TButton", background=[("active", "#da190b")])
        self.style.configure("TEntry", fieldbackground="#333333", foreground="white", insertcolor="white")
        self.style.configure("Disabled.TEntry", fieldbackground="#555555", foreground="#aaaaaa")
        self.style.configure("Treeview", background="#333333", foreground="white", fieldbackground="#333333", rowheight=25)
        self.style.map("Treeview", background=[("selected", "#4CAF50")])
        self.style.configure("Treeview.Heading", background="#1e1e1e", foreground="white", font=("Arial", 12, "bold"))
        self.style.configure("TLabelframe", background="#1e1e1e", foreground="white", bordercolor="#4CAF50")
        self.style.configure("TLabelframe.Label", background="#1e1e1e", foreground="white", font=("Arial", 12, "bold"))
        self.style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
        self.style.configure("TNotebook.Tab", background="#333333", foreground="white", padding=[10, 5], font=("Arial", 11, "bold"))
        self.style.map("TNotebook.Tab", background=[("selected", "#4CAF50")])

        self.status_label = ttk.Label(main_frame, text="Connecting to MetaTrader 5...", font=("Arial", 10), foreground="yellow")
        self.status_label.pack(fill=tk.X)

        price_frame = ttk.Frame(main_frame, style="Main.TFrame"); price_frame.pack(pady=10)
        ttk.Label(price_frame, text="Gold Price (XAUUSDm):", font=("Arial", 16, "bold")).pack(side=tk.LEFT, padx=5)
        self.price_label = ttk.Label(price_frame, text="N/A", font=("Arial", 16, "bold"), foreground="#FFD700"); self.price_label.pack(side=tk.LEFT)

        manual_frame = ttk.LabelFrame(main_frame, text="Risk Management (for all trades)", padding="10"); manual_frame.pack(fill=tk.X, pady=10)
        ttk.Label(manual_frame, text="Take Profit ($):").grid(row=0, column=0, padx=5, pady=5)
        self.tp_entry = ttk.Entry(manual_frame, width=10); self.tp_entry.grid(row=0, column=1, padx=5, pady=5); self.tp_entry.insert(0, "3.0")
        ttk.Label(manual_frame, text="Stop Loss ($):").grid(row=1, column=0, padx=5, pady=5)
        self.sl_entry = ttk.Entry(manual_frame, width=10); self.sl_entry.grid(row=1, column=1, padx=5, pady=5); self.sl_entry.insert(0, "2.0")
        buy_button = ttk.Button(manual_frame, text="Manual Buy", command=self.buy, style="Green.TButton"); buy_button.grid(row=0, column=2, padx=20, pady=5)
        sell_button = ttk.Button(manual_frame, text="Manual Sell", command=self.sell, style="Red.TButton"); sell_button.grid(row=1, column=2, padx=20, pady=5)

        strategy_frame = ttk.LabelFrame(main_frame, text="Strategy Settings", padding="10"); strategy_frame.pack(fill=tk.X, pady=10)
        strategy_frame.columnconfigure(2, weight=1)
        # (Strategy widgets setup as before)
        ttk.Label(strategy_frame, text="Strategy:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.strategy_combo = ttk.Combobox(strategy_frame, values=["MA Crossover", "Trend Following", "Gold M5 Scalper"], state="readonly", width=17)
        self.strategy_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w"); self.strategy_combo.set("MA Crossover")
        self.strategy_combo.bind("<<ComboboxSelected>>", self.update_ui_for_strategy)
        ttk.Label(strategy_frame, text="Timeframe:").grid(row=0, column=3, padx=5, pady=5, sticky="e")
        self.timeframe_combo = ttk.Combobox(strategy_frame, values=list(self.timeframe_map.keys()), state="readonly", width=17)
        self.timeframe_combo.grid(row=0, column=4, padx=5, pady=5, sticky="e"); self.timeframe_combo.set("1 Minute (M1)")
        self.param1_label = ttk.Label(strategy_frame, text="Short MA Period:"); self.param1_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.param1_entry = ttk.Entry(strategy_frame, width=10); self.param1_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w"); self.param1_entry.insert(0, "10")
        self.param2_label = ttk.Label(strategy_frame, text="Long MA Period:"); self.param2_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.param2_entry = ttk.Entry(strategy_frame, width=10); self.param2_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w"); self.param2_entry.insert(0, "50")
        self.param3_label = ttk.Label(strategy_frame, text="Max Spread (pips):"); self.param3_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.param3_entry = ttk.Entry(strategy_frame, width=10); self.param3_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w"); self.param3_entry.insert(0, "30")
        self.autotrade_button = ttk.Button(strategy_frame, text="Start Auto Trading", command=self.toggle_autotrade, style="Green.TButton")
        self.autotrade_button.grid(row=1, column=4, rowspan=2, padx=5, pady=5, sticky="nse")

        # --- Tabbed History View ---
        notebook_frame = ttk.Frame(main_frame, style="Main.TFrame")
        notebook_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        notebook = ttk.Notebook(notebook_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        open_tab = ttk.Frame(notebook, style="Main.TFrame"); notebook.add(open_tab, text="Open Positions")
        closed_tab = ttk.Frame(notebook, style="Main.TFrame"); notebook.add(closed_tab, text="Trade History")

        # Open Positions Tree
        self.open_positions_tree = ttk.Treeview(open_tab, columns=("Ticket", "Type", "Price", "TP", "SL"), displaycolumns=("Type", "Price", "TP", "SL"), show="headings")
        self.open_positions_tree.heading("Type", text="Type"); self.open_positions_tree.heading("Price", text="Price"); self.open_positions_tree.heading("TP", text="Take Profit"); self.open_positions_tree.heading("SL", text="Stop Loss")
        self.open_positions_tree.pack(fill=tk.BOTH, expand=True)

        # Closed Trades Tree
        self.closed_trades_tree = ttk.Treeview(closed_tab, columns=("Ticket", "Type", "Price", "TP", "SL", "Status"), displaycolumns=("Type", "Price", "TP", "SL", "Status"), show="headings")
        self.closed_trades_tree.heading("Type", text="Type"); self.closed_trades_tree.heading("Price", text="Price"); self.closed_trades_tree.heading("TP", text="Take Profit"); self.closed_trades_tree.heading("SL", text="Stop Loss"); self.closed_trades_tree.heading("Status", text="Status")
        self.closed_trades_tree.pack(fill=tk.BOTH, expand=True)
        
        self.update_ui_for_strategy()

    def update_ui_for_strategy(self, event=None):
        strategy = self.strategy_combo.get()
        if strategy == "Gold M5 Scalper":
            self.timeframe_combo.set("5 Minutes (M5)")
            self.timeframe_combo.config(state="disabled")
            self.param1_label.config(state="disabled"); self.param1_entry.config(state="disabled")
            self.param2_label.config(state="disabled"); self.param2_entry.config(state="disabled")
            self.param3_label.config(state="normal"); self.param3_entry.config(state="normal")
        else:
            self.timeframe_combo.config(state="readonly")
            self.param1_label.config(state="normal"); self.param1_entry.config(state="normal")
            self.param2_label.config(state="normal"); self.param2_entry.config(state="normal")
            self.param3_label.config(state="disabled"); self.param3_entry.config(state="disabled")
            if strategy == "MA Crossover":
                self.param1_label.config(text="Short MA Period:")
                self.param2_label.config(text="Long MA Period:")
            elif strategy == "Trend Following":
                self.param1_label.config(text="Signal MA Period:")
                self.param2_label.config(text="Trend MA Period:")

    def start_mt5(self):
        def connect():
            if not mt5.initialize(): self.root.after(0, self.update_status, f"MT5 Initialize failed: {mt5.last_error()}", "red"); return
            account_info = mt5.account_info()
            if account_info is None: self.root.after(0, self.update_status, f"MT5: Could not get account info: {mt5.last_error()}", "red"); return
            self.root.after(0, self.update_status, f"Connected to account #{account_info.login}", "green")
            self.update_price()
            # Start the history sync thread
            sync_thread = threading.Thread(target=self.sync_trade_history, daemon=True); sync_thread.start()
        threading.Thread(target=connect, daemon=True).start()

    def sync_trade_history(self):
        while True:
            try:
                open_positions = mt5.positions_get(magic=234000)
                server_open_tickets = {pos.ticket for pos in open_positions} if open_positions else set()
                
                gui_open_tickets = set(int(item_id) for item_id in self.open_positions_tree.get_children())
                
                closed_tickets = gui_open_tickets - server_open_tickets
                
                if closed_tickets:
                    self.root.after(0, self.move_trades_to_history, closed_tickets)
            except Exception as e:
                print(f"Error in history sync: {e}")
            time.sleep(15)

    def move_trades_to_history(self, ticket_ids):
        for ticket in ticket_ids:
            try:
                item_values = self.open_positions_tree.item(ticket, 'values')
                if not item_values: continue
                
                # Values from open_positions_tree: (Ticket, Type, Price, TP, SL)
                # We need to add "Closed" for the closed_trades_tree
                closed_values = item_values + ("Closed",)
                self.closed_trades_tree.insert("", "end", values=closed_values)
                self.open_positions_tree.delete(ticket)
            except Exception as e:
                print(f"Error moving trade {ticket} to history: {e}")

    def update_status(self, text, color): self.status_label.config(text=text, foreground=color)
    def update_price(self):
        def fetch():
            symbol = "XAUUSDm"
            while True:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    self.root.after(0, self.price_label.config, {"text": f"${tick.ask}"})
                time.sleep(1)
        threading.Thread(target=fetch, daemon=True).start()

    def toggle_autotrade(self):
        if not self.autotrade_enabled:
            try:
                strategy = self.strategy_combo.get()
                timeframe = self.timeframe_map[self.timeframe_combo.get()]
                param1 = int(self.param1_entry.get()) if self.param1_entry.cget('state') != 'disabled' else None
                param2 = int(self.param2_entry.get()) if self.param2_entry.cget('state') != 'disabled' else None
                param3 = int(self.param3_entry.get()) if self.param3_entry.cget('state') != 'disabled' else None

                if strategy != "Gold M5 Scalper" and param1 >= param2:
                    messagebox.showerror("Error", "First MA period must be less than second MA period.")
                    return
                
                self.autotrade_enabled = True
                self.autotrade_button.config(text="Stop Auto Trading", style="Red.TButton")
                self.update_status(f"Auto Trading Started: {strategy} on {self.timeframe_combo.get()}", "cyan")
                
                self.strategy_thread = threading.Thread(target=self.run_strategy_loop, args=(strategy, timeframe, param1, param2, param3), daemon=True)
                self.strategy_thread.start()
            except (ValueError, TypeError):
                messagebox.showerror("Error", "Invalid parameter. Please enter valid integers.")
        else:
            self.autotrade_enabled = False
            self.autotrade_button.config(text="Start Auto Trading", style="Green.TButton")
            self.update_status("Auto Trading Stopped.", "orange")
            self.last_trade_action = None

    def run_strategy_loop(self, strategy, timeframe, param1, param2, param3):
        symbol = "XAUUSDm"
        sleep_duration = self.sleep_intervals.get(timeframe, 60)
        while self.autotrade_enabled:
            try:
                if strategy == "MA Crossover": self.run_ma_crossover_logic(strategy, symbol, timeframe, param1, param2)
                elif strategy == "Trend Following": self.run_trend_following_logic(strategy, symbol, timeframe, param1, param2)
                elif strategy == "Gold M5 Scalper": self.run_gold_scalper_logic(strategy, symbol, param3)
                time.sleep(sleep_duration)
            except Exception as e:
                print(f"Error in strategy loop: {e}")
                self.root.after(0, self.update_status, f"Strategy Error: {e}", "red")
                time.sleep(30)

    def run_gold_scalper_logic(self, strategy, symbol, max_spread):
        # Time Filter
        server_time = datetime.now(timezone.utc)
        if not (13 <= server_time.hour < 17):
            self.root.after(0, self.update_status, "Scalper: Outside trading hours (13:00-17:00 UTC). Waiting...", "orange")
            return
        
        # Spread Filter
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            self.root.after(0, self.update_status, "Scalper: Could not get symbol info.", "red")
            return
        spread = symbol_info.spread
        if spread > max_spread:
            self.root.after(0, self.update_status, f"Scalper: Spread too high ({spread} > {max_spread}). Waiting...", "orange")
            return

        # Indicator Logic
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 100)
        if rates is None or len(rates) < 22:
            self.root.after(0, self.update_status, "Scalper: Not enough data for indicators. Waiting...", "orange")
            return
        
        close_prices = np.array([r['close'] for r in rates])
        rsi_values = calculate_rsi(close_prices, 14)
        if rsi_values is None or len(rsi_values) < 2: return
        
        current_rsi, prev_rsi = rsi_values[-1], rsi_values[-2]
        current_price = close_prices[-1]
        ema21 = calculate_ema(close_prices, 21)

        self.root.after(0, self.update_status, f"Scalper: Price={current_price:.2f}, EMA21={ema21:.2f}, RSI={current_rsi:.2f}", "cyan")

        # Buy Signal
        if current_price > ema21 and prev_rsi < 30 and current_rsi >= 30 and self.last_trade_action != "Buy":
            self.root.after(0, self.update_status, "Gold Scalper: Buy signal!", "green")
            self.root.after(0, self.execute_trade, "Buy", is_auto=True)
            self.last_trade_action = "Buy"
        # Sell Signal
        elif current_price < ema21 and prev_rsi > 70 and current_rsi <= 70 and self.last_trade_action != "Sell":
            self.root.after(0, self.update_status, "Gold Scalper: Sell signal!", "red")
            self.root.after(0, self.execute_trade, "Sell", is_auto=True)
            self.last_trade_action = "Sell"

    def run_ma_crossover_logic(self, strategy, symbol, timeframe, short_period, long_period):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, long_period + 5)
        if rates is None or len(rates) < long_period:
            self.root.after(0, self.update_status, f"Not enough data for MA({long_period}). Waiting...", "orange")
            return
        
        close = np.array([rate['close'] for rate in rates])
        short_ma = np.mean(close[-short_period:])
        long_ma = np.mean(close[-long_period:])
        prev_short_ma = np.mean(close[-short_period-1:-1])
        prev_long_ma = np.mean(close[-long_period-1:-1])

        self.root.after(0, self.update_status, f"Checking {strategy}: Short MA={short_ma:.2f}, Long MA={long_ma:.2f}", "cyan")

        if prev_short_ma < prev_long_ma and short_ma > long_ma and self.last_trade_action != "Buy":
            self.root.after(0, self.update_status, "MA Crossover: Buy signal!", "green")
            self.root.after(0, self.execute_trade, "Buy", is_auto=True)
            self.last_trade_action = "Buy"
        elif prev_short_ma > prev_long_ma and short_ma < long_ma and self.last_trade_action != "Sell":
            self.root.after(0, self.update_status, "MA Crossover: Sell signal!", "red")
            self.root.after(0, self.execute_trade, "Sell", is_auto=True)
            self.last_trade_action = "Sell"

    def run_trend_following_logic(self, strategy, symbol, timeframe, signal_period, trend_period):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, trend_period + 5)
        if rates is None or len(rates) < trend_period:
            self.root.after(0, self.update_status, f"Not enough data for MA({trend_period}). Waiting...", "orange")
            return

        close = np.array([rate['close'] for rate in rates])
        signal_ma = np.mean(close[-signal_period:])
        trend_ma = np.mean(close[-trend_period:])
        
        prev_close = close[-2]
        curr_close = close[-1]
        prev_signal_ma = np.mean(close[-signal_period-1:-1])

        self.root.after(0, self.update_status, f"Checking {strategy}: Price={curr_close:.2f}, Trend MA={trend_ma:.2f}", "cyan")

        is_uptrend = curr_close > trend_ma
        if is_uptrend and self.last_trade_action != "Buy":
            if prev_close < prev_signal_ma and curr_close > signal_ma: # Price crosses above signal MA
                self.root.after(0, self.update_status, "Trend Following: Buy signal!", "green")
                self.root.after(0, self.execute_trade, "Buy", is_auto=True)
                self.last_trade_action = "Buy"
        
        elif not is_uptrend and self.last_trade_action != "Sell":
            if prev_close > prev_signal_ma and curr_close < signal_ma: # Price crosses below signal MA
                self.root.after(0, self.update_status, "Trend Following: Sell signal!", "red")
                self.root.after(0, self.execute_trade, "Sell", is_auto=True)
                self.last_trade_action = "Sell"

    def buy(self): self.execute_trade("Buy")
    def sell(self): self.execute_trade("Sell")

    def execute_trade(self, trade_type, is_auto=False):
        try:
            tp_val = float(self.tp_entry.get()); sl_val = float(self.sl_entry.get())
        except ValueError:
            if not is_auto: messagebox.showerror("Error", "Invalid TP/SL values.")
            else: self.root.after(0, self.update_status, "Auto-trade failed: Invalid TP/SL.", "red")
            return

        symbol = "XAUUSDm"; lot_size = 0.01
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            if not is_auto: messagebox.showerror("Error", "Could not fetch price for trade.")
            return
        
        price = tick.ask if trade_type == "Buy" else tick.bid
        tp = price + tp_val if trade_type == "Buy" else price - tp_val
        sl = price - sl_val if trade_type == "Buy" else price + sl_val

        request = { "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot_size, "type": mt5.ORDER_TYPE_BUY if trade_type == "Buy" else mt5.ORDER_TYPE_SELL, "price": price, "sl": sl, "tp": tp, "magic": 234000, "comment": "Python EA", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC }
        result = mt5.order_send(request)
        
        position_id = None
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            deals = mt5.history_deals_get(ticket=result.deal)
            if deals and len(deals) > 0:
                position_id = deals[0].position_id
        
        if not position_id:
            if not is_auto: messagebox.showerror("Error", f"Order failed: {result.comment}")
        else:
            if not is_auto: messagebox.showinfo("Success", f"{trade_type} order placed.")
            trade_source = f"Auto {trade_type}" if is_auto else f"Manual {trade_type}"
            # Add to the "Open Positions" tab
            # Values for open_positions_tree: (Ticket, Type, Price, TP, SL)
            trade_values = (position_id, trade_source, price, tp, sl)
            self.open_positions_tree.insert("", "end", iid=position_id, values=trade_values)

    def on_closing(self):
        self.autotrade_enabled = False
        mt5.shutdown()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
