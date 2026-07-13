import sys
import threading
import time
from datetime import datetime, timezone
import queue
import signal
import json
import os

import MetaTrader5 as mt5
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QLineEdit, QComboBox, QGroupBox, QTabWidget, QTableView,
                             QAbstractItemView, QHeaderView, QMessageBox, QSizeGrip, QFrame, QDialog,
                             QDialogButtonBox, QFormLayout)
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool, Slot, Qt, QPoint, QSize
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor

# Import Strategies
from strategies.ma_crossover import run_ma_crossover_logic
from strategies.trend_following import run_trend_following_logic
from strategies.gold_scalper import run_gold_scalper_logic
from strategies.ict_trader import run_ict_trader_logic
from strategies.ict_gold_scalping import run_ict_gold_scalping_logic
from strategies.gold_scalping_new import run_gold_scalping_new_logic
from strategies.smc import run_smc_logic

# --- Worker Signals ---
class WorkerSignals(QObject):
    update_status = Signal(str, str)
    update_price = Signal(str)
    update_balance = Signal(str)
    add_open_trade = Signal(dict)
    update_pl = Signal(str, float) # Changed ticket to str to handle large 64-bit IDs
    move_to_history = Signal(dict)
    show_message = Signal(str, str)

# --- Backend Logic Worker ---
class BackendWorker(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self.autotrade_enabled = False
        self.last_trade_action = None
        self.strategy_params = {}
        self.symbol = "XAUUSDm"  # Default symbol

        self.timeframe_map = {
            "1 Minute (M1)": mt5.TIMEFRAME_M1, "5 Minutes (M5)": mt5.TIMEFRAME_M5,
            "15 Minutes (M15)": mt5.TIMEFRAME_M15, "1 Hour (H1)": mt5.TIMEFRAME_H1,
            "4 Hours (H4)": mt5.TIMEFRAME_H4,
        }
        self.sleep_intervals = {
            mt5.TIMEFRAME_M1: 60, mt5.TIMEFRAME_M5: 60, mt5.TIMEFRAME_M15: 300,
            mt5.TIMEFRAME_H1: 900, mt5.TIMEFRAME_H4: 1800,
        }

    @Slot()
    def run(self):
        if not self.start_mt5():
            return
        
        threading.Thread(target=self.update_price, daemon=True).start()
        threading.Thread(target=self.sync_trade_history, daemon=True).start()
        threading.Thread(target=self.strategy_main_loop, daemon=True).start()

    def start_mt5(self):
        if not mt5.initialize():
            self.signals.update_status.emit(f"MT5 Initialize failed: {mt5.last_error()}", "red")
            return False
        account_info = mt5.account_info()
        if account_info is None:
            self.signals.update_status.emit(f"MT5: Could not get account info: {mt5.last_error()}", "red")
            return False
        self.signals.update_status.emit(f"Connected to account #{account_info.login}", "green")
        return True

    def ensure_symbol_ready(self):
        symbol = self.symbol.strip()
        if symbol != self.symbol:
            self.symbol = symbol

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.signals.update_status.emit(f"Symbol not found: {symbol}. Check exact broker symbol name.", "red")
            return False

        if not symbol_info.visible and not mt5.symbol_select(symbol, True):
            self.signals.update_status.emit(f"Could not select symbol: {symbol}", "red")
            return False

        return True

    def update_price(self):
        while True:
            tick = mt5.symbol_info_tick(self.symbol) if self.ensure_symbol_ready() else None
            if tick:
                self.signals.update_price.emit(f"${tick.ask:.2f}")
            else:
                self.signals.update_price.emit("N/A")
            
            # Update Balance
            account = mt5.account_info()
            if account:
                self.signals.update_balance.emit(f"${account.balance:.2f}")
            
            time.sleep(1)

    def sync_trade_history(self):
        while True:
            try:
                open_positions = mt5.positions_get(symbol=self.symbol)
                server_positions = {pos.ticket: pos for pos in open_positions} if open_positions else {}
                server_tickets = set(server_positions.keys())
                
                gui_open_tickets = set(self.strategy_params.get('gui_tickets', set()))

                new_tickets = server_tickets - gui_open_tickets
                for ticket in new_tickets:
                    pos = server_positions[ticket]
                    trade_type = "Buy" if pos.type == mt5.ORDER_TYPE_BUY else "Sell"
                    source = "Auto" if getattr(pos, 'magic', 0) == 234000 else "Manual MT5"
                    trade_values = {
                        "ticket": ticket,
                        "type": f"{source} {trade_type}",
                        "price": f"{pos.price_open:.2f}",
                        "tp": f"{pos.tp:.2f}",
                        "sl": f"{pos.sl:.2f}"
                    }
                    self.signals.add_open_trade.emit(trade_values)

                closed_tickets = gui_open_tickets - server_tickets
                still_open_tickets = gui_open_tickets.intersection(server_tickets)

                for ticket in closed_tickets:
                    deals = mt5.history_deals_get(position=ticket)
                    final_pl = sum(d.profit + d.swap for d in deals) if deals else 0.0
                    self.signals.move_to_history.emit({"ticket": ticket, "final_pl": final_pl})

                for ticket in still_open_tickets:
                    profit = server_positions[ticket].profit
                    self.signals.update_pl.emit(str(ticket), profit) # Emit ticket as string

            except Exception as e:
                print(f"Error in history sync: {e}")
            time.sleep(5)

    def execute_trade(self, trade_type, is_auto=False, sl_price=None, tp_price=None, volume=None):
        try:
            max_pos = self.strategy_params['max_pos']
            open_positions = mt5.positions_get(magic=234000)
            if open_positions and len(open_positions) >= max_pos:
                self.signals.update_status.emit(f"Max positions ({max_pos}) reached. Signal ignored.", "orange")
                return False

            tp_val = self.strategy_params['tp']; sl_val = self.strategy_params['sl']
            
            # Determine volume
            if volume is None:
                volume = self.strategy_params.get('lot_size', 0.01)
                
        except (ValueError, TypeError, KeyError):
            self.signals.show_message.emit("Error", "Invalid Risk Management values.")
            return False

        if not self.ensure_symbol_ready():
            self.signals.show_message.emit("Error", f"Symbol is not available or not selected: {self.symbol}")
            return False

        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            self.signals.show_message.emit("Error", f"Could not get live tick price for {self.symbol}")
            return False
        
        price = tick.ask if trade_type == "Buy" else tick.bid
        
        # Use provided sl/tp prices if available, otherwise calculate from UI values
        sl = sl_price if sl_price is not None else (price - sl_val if trade_type == "Buy" else price + sl_val)
        tp = tp_price if tp_price is not None else (price + tp_val if trade_type == "Buy" else price - tp_val)

        request = { "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": float(volume), "type": mt5.ORDER_TYPE_BUY if trade_type == "Buy" else mt5.ORDER_TYPE_SELL, "price": price, "sl": sl, "tp": tp, "magic": 234000, "comment": "Python EA", "type_time": mt5.ORDER_TIME_GTC }
        result = None
        filling_modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
        last_comment = "Unknown"
        for filling_mode in filling_modes:
            request["type_filling"] = filling_mode
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                break
            if result:
                last_comment = result.comment
        
        position_id = None
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            deals = mt5.history_deals_get(ticket=result.deal)
            if deals and len(deals) > 0:
                position_id = deals[0].position_id
        
        if not position_id:
            if not is_auto: self.signals.show_message.emit("Order Failed", f"Order failed: {result.comment if result else last_comment}")
            return False
        else:
            if not is_auto: self.signals.show_message.emit("Success", f"{trade_type} order placed.")
            trade_source = f"Auto {trade_type}" if is_auto else f"Manual {trade_type}"
            trade_values = {"ticket": position_id, "type": trade_source, "price": f"{price:.2f}", "tp": f"{tp:.2f}", "sl": f"{sl:.2f}"}
            self.signals.add_open_trade.emit(trade_values)
            return True

    def strategy_main_loop(self):
        while True:
            if self.autotrade_enabled:
                # --- Global Time Filter ---
                try:
                    start_time = self.strategy_params.get('start_time')
                    end_time = self.strategy_params.get('end_time')
                    
                    if start_time and end_time:
                        current_time = datetime.now(timezone.utc).time()
                        
                        # Handle overnight sessions (e.g., end_time is earlier than start_time)
                        if start_time <= end_time:
                            if not (start_time <= current_time <= end_time):
                                self.signals.update_status.emit(f"Outside trading hours ({start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')} UTC).", "orange")
                                time.sleep(60) # Sleep longer when outside hours
                                continue
                        else: # Overnight session
                            if not (current_time >= start_time or current_time <= end_time):
                                self.signals.update_status.emit(f"Outside trading hours ({start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')} UTC).", "orange")
                                time.sleep(60)
                                continue
                except Exception as e:
                    self.signals.update_status.emit(f"Time filter error: {e}", "red")

                strategy = self.strategy_params.get('strategy')
                timeframe = self.strategy_params.get('timeframe')
                param1 = self.strategy_params.get('param1')
                param2 = self.strategy_params.get('param2')
                param3 = self.strategy_params.get('param3')
                
                try:
                    if strategy == "MA Crossover": run_ma_crossover_logic(self, self.symbol, timeframe, param1, param2)
                    elif strategy == "Trend Following": run_trend_following_logic(self, self.symbol, timeframe, param1, param2)
                    elif strategy == "Gold M5 Scalper": run_gold_scalper_logic(self, self.symbol, param3, self.strategy_params.get('scalper_ema', 21))
                    elif strategy == "ICT Trader": run_ict_trader_logic(self, self.symbol, timeframe)
                    elif strategy == "ICT Gold Scalping": run_ict_gold_scalping_logic(self, self.symbol, timeframe)
                    elif strategy == "Gold Scalping New": run_gold_scalping_new_logic(self, self.symbol)
                    elif strategy == "SMC": run_smc_logic(self, self.symbol, timeframe, param1, param2) # param1=Risk%, param2=RR
                except Exception as e:
                    print(f"Error in strategy run: {e}")
                    self.signals.update_status.emit(f"Strategy Error: {e}", "red")
                
                # Interruptible sleep
                sleep_duration = self.sleep_intervals.get(timeframe, 60)
                for _ in range(sleep_duration):
                    if not self.autotrade_enabled:
                        break # Exit sleep loop immediately if stopped
                    time.sleep(1)
            else:
                time.sleep(1) # Sleep briefly when disabled

# --- Scalper Settings Dialog ---
class ScalperSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gold M5 Scalper Settings")
        self.setModal(True)
        self.setFixedSize(300, 200)
        
        # Make the dialog frameless and translucent
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main layout for the dialog (holds the styled frame)
        dialog_layout = QVBoxLayout(self)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.setSpacing(0)

        # Background Frame (This will be the visible window)
        self.bg_frame = QFrame()
        self.bg_frame.setObjectName("bgFrame")
        self.bg_frame.setStyleSheet("""
            QFrame#bgFrame {
                background-color: #252525;
                border: 1px solid #4CAF50;
                border-radius: 8px;
            }
        """)
        dialog_layout.addWidget(self.bg_frame)

        # Layout inside the frame
        self.main_layout = QVBoxLayout(self.bg_frame)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Custom Title Bar Widget
        self.title_bar_widget = QWidget()
        self.title_bar_widget.setFixedHeight(30)
        self.title_bar_widget.setStyleSheet("""
            background-color: #1e1e1e;
            border-bottom: 1px solid #333;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        """)
        
        title_bar_layout = QHBoxLayout(self.title_bar_widget)
        title_bar_layout.setContentsMargins(10, 0, 5, 0)
        title_bar_layout.setSpacing(5)
        
        self.dialog_title_label = QLabel("Gold M5 Scalper Settings")
        self.dialog_title_label.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 10pt; border: none; background: transparent;")
        title_bar_layout.addWidget(self.dialog_title_label)
        title_bar_layout.addStretch()

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(25, 25)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent; 
                color: #bbbbbb; 
                border: none; 
                font-weight: bold; 
                font-size: 10pt;
            }
            QPushButton:hover { background-color: #d32f2f; color: white; border-radius: 4px; }
        """)
        self.close_btn.clicked.connect(self.reject)
        title_bar_layout.addWidget(self.close_btn)
        
        self.main_layout.addWidget(self.title_bar_widget)

        # Content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 15, 20, 15)
        content_layout.setSpacing(15)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)
        
        self.risk_combo = QComboBox()
        self.risk_combo.addItems(["Low", "Medium", "High"])
        self.risk_combo.setCurrentText("Medium")
        self.risk_combo.setStyleSheet("background-color: #1a1a1a; color: white; padding: 5px; border: 1px solid #444; border-radius: 4px;")
        label_risk = QLabel("Risk Level:")
        label_risk.setStyleSheet("color: #e0e0e0; font-weight: bold; border: none; background: transparent;")
        form_layout.addRow(label_risk, self.risk_combo)
        
        self.ema_combo = QComboBox()
        self.ema_combo.addItems(["21", "50", "100", "200"])
        self.ema_combo.setCurrentText("21")
        self.ema_combo.setStyleSheet("background-color: #1a1a1a; color: white; padding: 5px; border: 1px solid #444; border-radius: 4px;")
        label_ema = QLabel("EMA Trend Filter:")
        label_ema.setStyleSheet("color: #e0e0e0; font-weight: bold; border: none; background: transparent;")
        form_layout.addRow(label_ema, self.ema_combo)
        
        content_layout.addLayout(form_layout)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; padding: 6px 15px; border: none; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #45a049; }
            QPushButton[text="Cancel"] { background-color: #d32f2f; }
            QPushButton[text="Cancel"]:hover { background-color: #b71c1c; }
        """)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        content_layout.addWidget(self.buttons)

        self.main_layout.addWidget(content_widget)

        # Dragging logic
        self.start_pos = QPoint(0, 0)
        self.dragging = False

    def get_settings(self):
        return {
            "risk": self.risk_combo.currentText(),
            "ema": int(self.ema_combo.currentText())
        }

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Check if click is within title bar height (approx 30px + window margins)
            if event.position().y() <= 40:
                self.start_pos = event.globalPosition().toPoint() - self.pos()
                self.dragging = True
                event.accept()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(event.globalPosition().toPoint() - self.start_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            event.accept()

# --- Custom Title Bar ---
class CustomTitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(40)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(10, 0, 10, 0)
        self.layout.setSpacing(10)
        
        # Title
        self.title_label = QLabel("Gold EA Trading Bot")
        self.title_label.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 12pt; border: none;")
        self.layout.addWidget(self.title_label)
        self.layout.addStretch()

        # Buttons
        button_style = """
            QPushButton {
                background-color: transparent; 
                color: #bbbbbb; 
                border: none; 
                font-weight: bold; 
                font-size: 12pt;
                width: 30px;
                height: 30px;
            }
            QPushButton:hover { background-color: #444; color: white; border-radius: 4px; }
        """
        close_style = """
            QPushButton {
                background-color: transparent; 
                color: #bbbbbb; 
                border: none; 
                font-weight: bold; 
                font-size: 12pt;
                width: 30px;
                height: 30px;
            }
            QPushButton:hover { background-color: #d32f2f; color: white; border-radius: 4px; }
        """

        self.minimize_btn = QPushButton("-")
        self.minimize_btn.setStyleSheet(button_style)
        self.minimize_btn.clicked.connect(self.minimize_window)
        self.layout.addWidget(self.minimize_btn)

        self.maximize_btn = QPushButton("□")
        self.maximize_btn.setStyleSheet(button_style)
        self.maximize_btn.clicked.connect(self.maximize_restore_window)
        self.layout.addWidget(self.maximize_btn)

        self.close_btn = QPushButton("✕")
        self.close_btn.setStyleSheet(close_style)
        self.close_btn.clicked.connect(self.close_window)
        self.layout.addWidget(self.close_btn)

        self.start = QPoint(0, 0)
        self.pressing = False

    def minimize_window(self):
        self.parent.showMinimized()

    def maximize_restore_window(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
            self.maximize_btn.setText("□")
        else:
            self.parent.showMaximized()
            self.maximize_btn.setText("❐")

    def close_window(self):
        self.parent.close()

    def mousePressEvent(self, event):
        self.start = self.mapToGlobal(event.position().toPoint())
        self.pressing = True

    def mouseMoveEvent(self, event):
        if self.pressing:
            end = self.mapToGlobal(event.position().toPoint())
            movement = end - self.start
            self.parent.setGeometry(self.parent.x() + movement.x(),
                                  self.parent.y() + movement.y(),
                                  self.parent.width(),
                                  self.parent.height())
            self.start = end

    def mouseReleaseEvent(self, event):
        self.pressing = False

# --- Main Window Class ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gold EA Trading Bot")
        self.setGeometry(100, 100, 900, 800)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 5px;
            }
            QWidget { background-color: #1e1e1e; color: white; font-size: 11pt; }
            
            /* --- Group Box (Sections) --- */
            QGroupBox {
                background-color: #252525;
                border: 1px solid #3d3d3d;
                border-radius: 8px;
                margin-top: 1.2em; /* Leave space for title */
                padding-top: 15px; /* Push content down */
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 15px;
                padding: 0 5px;
                color: #4CAF50; /* Accent color for titles */
                font-size: 11pt;
                font-weight: bold;
                background-color: #1e1e1e; /* Matches main bg to look like it floats */
            }

            /* --- Inputs & Combo Boxes --- */
            QLineEdit, QComboBox {
                background-color: #1a1a1a;
                border: 1px solid #444;
                padding: 8px 10px; /* More padding */
                border-radius: 4px;
                color: #eee;
                font-size: 10pt;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #4CAF50;
                background-color: #151515;
            }
            QComboBox::drop-down {
                border: none;
                width: 25px;
            }
            QComboBox:disabled, QLineEdit:disabled {
                background-color: #2a2a2a;
                color: #777;
                border-color: #444;
            }
            
            /* --- Modern Table Styling --- */
            QTableView {
                background-color: #1e1e1e;
                alternate-background-color: #252525;
                selection-background-color: #3d3d3d;
                selection-color: #ffffff;
                border: 1px solid #333;
                border-radius: 4px;
                gridline-color: #333;
                outline: 0; /* Remove dotted focus line */
            }
            QTableView::item {
                padding: 5px;
                border: none;
            }
            QTableView::item:selected {
                background-color: #3d3d3d;
                color: #ffffff;
            }
            
            /* Header Styling */
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #e0e0e0;
                padding: 8px;
                border: none;
                border-bottom: 2px solid #444;
                font-weight: bold;
                font-size: 10pt;
                text-transform: uppercase;
            }

            /* Scrollbar Styling */
            QScrollBar:vertical {
                border: none;
                background: #1e1e1e;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #555;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #777;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }

            /* Tab Widget Styling */
            QTabWidget::pane {
                border: 1px solid #333;
                border-radius: 4px;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background: #252525;
                color: #888;
                padding: 10px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #4CAF50;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #333;
                color: #aaa;
            }

            QPushButton#buyButton, QPushButton#startButton { background-color: #28a745; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }
            QPushButton#buyButton:hover, QPushButton#startButton:hover { background-color: #218838; }
            QPushButton#sellButton, QPushButton#stopButton { background-color: #dc3545; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }
            QPushButton#sellButton:hover, QPushButton#stopButton:hover { background-color: #c82333; }
            #contentWidget {
                border-top: 1px solid #333;
            }
        """)
        
        self.open_positions_model = QStandardItemModel()
        self.closed_trades_model = QStandardItemModel()
        self.gui_open_tickets = set()

        self._create_ui()
        self._start_backend()
        self.load_settings()

    def _create_ui(self):
        # Main container
        container = QWidget()
        container.setObjectName("MainContainer")
        self.setCentralWidget(container)

        # Main layout (Title Bar + Content)
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Custom Title Bar
        self.title_bar = CustomTitleBar(self)
        main_layout.addWidget(self.title_bar)

        # Content Widget
        content_widget = QWidget()
        content_widget.setObjectName("contentWidget")
        main_layout.addWidget(content_widget)

        # Content Layout
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(15, 15, 15, 15)

        self.status_label = QLabel("Connecting to MetaTrader 5..."); self.status_label.setStyleSheet("color: yellow;")
        layout.addWidget(self.status_label)

        price_layout = QHBoxLayout()
        price_layout.addWidget(QLabel("Symbol:"))
        self.symbol_input = QLineEdit("XAUUSDm")
        self.symbol_input.setFixedWidth(100)
        self.symbol_input.textChanged.connect(self.update_symbol)
        price_layout.addWidget(self.symbol_input)

        price_layout.addWidget(QLabel("Price:"))
        self.price_label = QLabel("N/A"); self.price_label.setStyleSheet("color: #FFD700; font-weight: bold; font-size: 14pt;")
        price_layout.addWidget(self.price_label)
        
        price_layout.addStretch()
        
        price_layout.addWidget(QLabel("Balance:"))
        self.balance_label = QLabel("N/A")
        self.balance_label.setStyleSheet("color: #00FF00; font-weight: bold; font-size: 12pt;")
        price_layout.addWidget(self.balance_label)
        
        layout.addLayout(price_layout)

        risk_box = QGroupBox("Risk Management")
        risk_layout = QGridLayout(risk_box)
        
        risk_layout.addWidget(QLabel("Lot Size:"), 0, 0)
        self.lot_size_input = QLineEdit("0.01")
        risk_layout.addWidget(self.lot_size_input, 0, 1)
        
        risk_layout.addWidget(QLabel("Max Positions:"), 0, 2)
        self.max_pos_input = QLineEdit("5")
        risk_layout.addWidget(self.max_pos_input, 0, 3)

        risk_layout.addWidget(QLabel("Take Profit ($):"), 1, 0)
        self.tp_input = QLineEdit("3.0")
        risk_layout.addWidget(self.tp_input, 1, 1)
        
        risk_layout.addWidget(QLabel("Stop Loss ($):"), 1, 2)
        self.sl_input = QLineEdit("2.0")
        risk_layout.addWidget(self.sl_input, 1, 3)

        self.buy_button = QPushButton("Manual Buy")
        self.buy_button.setObjectName("buyButton")
        self.buy_button.clicked.connect(self.manual_buy)
        risk_layout.addWidget(self.buy_button, 2, 0, 1, 2)

        self.sell_button = QPushButton("Manual Sell")
        self.sell_button.setObjectName("sellButton")
        self.sell_button.clicked.connect(self.manual_sell)
        risk_layout.addWidget(self.sell_button, 2, 2, 1, 2)

        layout.addWidget(risk_box)

        strat_box = QGroupBox("Strategy Settings")
        strat_layout = QGridLayout(strat_box)
        self.strategy_combo = QComboBox(); self.strategy_combo.addItems(["MA Crossover", "Trend Following", "Gold M5 Scalper", "ICT Trader", "ICT Gold Scalping", "Gold Scalping New", "SMC"]); self.strategy_combo.currentTextChanged.connect(self.update_ui_for_strategy)
        strat_layout.addWidget(QLabel("Strategy:"), 0, 0); strat_layout.addWidget(self.strategy_combo, 0, 1)
        self.timeframe_combo = QComboBox(); self.timeframe_combo.addItems(["1 Minute (M1)", "5 Minutes (M5)", "15 Minutes (M15)", "1 Hour (H1)", "4 Hours (H4)"])
        strat_layout.addWidget(QLabel("Timeframe:"), 0, 2); strat_layout.addWidget(self.timeframe_combo, 0, 3)

        self.start_time_label = QLabel("Start Time (UTC):"); self.start_time_input = QLineEdit("08:00")
        strat_layout.addWidget(self.start_time_label, 1, 2); strat_layout.addWidget(self.start_time_input, 1, 3)
        self.end_time_label = QLabel("End Time (UTC):"); self.end_time_input = QLineEdit("17:00")
        strat_layout.addWidget(self.end_time_label, 2, 2); strat_layout.addWidget(self.end_time_input, 2, 3)

        self.param1_label = QLabel("Short MA Period:"); self.param1_input = QLineEdit("10")
        strat_layout.addWidget(self.param1_label, 1, 0); strat_layout.addWidget(self.param1_input, 1, 1)
        self.param2_label = QLabel("Long MA Period:"); self.param2_input = QLineEdit("50")
        strat_layout.addWidget(self.param2_label, 2, 0); strat_layout.addWidget(self.param2_input, 2, 1)
        self.param3_label = QLabel("Max Spread (Points):"); self.param3_input = QLineEdit("300")
        strat_layout.addWidget(self.param3_label, 3, 0); strat_layout.addWidget(self.param3_input, 3, 1)
        
        self.autotrade_button = QPushButton("Start Auto Trading"); self.autotrade_button.setObjectName("startButton"); self.autotrade_button.clicked.connect(self.toggle_autotrade)
        strat_layout.addWidget(self.autotrade_button, 4, 0, 1, 4)
        layout.addWidget(strat_box)
        self.update_ui_for_strategy()

        tabs = QTabWidget()
        open_tab, closed_tab = QWidget(), QWidget()
        tabs.addTab(open_tab, "Open Positions"); tabs.addTab(closed_tab, "Trade History")
        
        self.open_view = QTableView(); self.open_view.setModel(self.open_positions_model)
        self.setup_table_view(self.open_view, self.open_positions_model, ["Ticket", "Type", "Price", "TP", "SL", "P/L ($)"])
        open_layout = QVBoxLayout(open_tab); open_layout.addWidget(self.open_view)

        self.closed_view = QTableView(); self.closed_view.setModel(self.closed_trades_model)
        self.setup_table_view(self.closed_view, self.closed_trades_model, ["Ticket", "Type", "Price", "TP", "SL", "Final P/L", "Status"])
        closed_layout = QVBoxLayout(closed_tab); closed_layout.addWidget(self.closed_view)
        
        layout.addWidget(tabs)

        # Resize Grip
        grip_layout = QHBoxLayout()
        grip_layout.setContentsMargins(0, 0, 0, 0) # Compact
        grip_layout.addStretch()
        self.size_grip = QSizeGrip(self)
        self.size_grip.setStyleSheet("width: 20px; height: 20px; background: transparent;") # Ensure it's visible but subtle
        grip_layout.addWidget(self.size_grip)
        main_layout.addLayout(grip_layout)

    def update_symbol(self, text):
        if hasattr(self, 'worker'):
            self.worker.symbol = text

    def setup_table_view(self, view, model, headers):
        model.setHorizontalHeaderLabels(headers)
        view.verticalHeader().setVisible(False)
        view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        view.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Modern Table Properties
        view.setAlternatingRowColors(True)
        view.setShowGrid(False)  # Cleaner look without grid lines
        view.verticalHeader().setDefaultSectionSize(35) # Taller rows
        view.setFocusPolicy(Qt.NoFocus) # Remove focus dotted line

    def _start_backend(self):
        self.threadpool = QThreadPool()
        self.worker = BackendWorker()
        self.worker.signals.update_status.connect(self.update_status)
        self.worker.signals.update_price.connect(self.update_price)
        self.worker.signals.update_balance.connect(self.update_balance)
        self.worker.signals.add_open_trade.connect(self.add_open_trade)
        self.worker.signals.update_pl.connect(self.update_pl)
        self.worker.signals.move_to_history.connect(self.move_to_history)
        self.worker.signals.show_message.connect(self.show_message)
        self.threadpool.start(self.worker)

    @Slot(str, str)
    def update_status(self, text, color):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")

    @Slot(str)
    def update_price(self, text):
        self.price_label.setText(text)

    @Slot(str)
    def update_balance(self, text):
        self.balance_label.setText(text)

    @Slot(dict)
    def add_open_trade(self, trade_data):
        ticket = trade_data['ticket']
        if ticket in self.gui_open_tickets: return
        self.gui_open_tickets.add(ticket)
        self.worker.strategy_params['gui_tickets'] = self.gui_open_tickets
        row = [QStandardItem(str(trade_data['ticket'])), QStandardItem(trade_data['type']), QStandardItem(trade_data['price']), QStandardItem(trade_data['tp']), QStandardItem(trade_data['sl']), QStandardItem("0.00")]
        self.open_positions_model.appendRow(row)

    @Slot(str, float)
    def update_pl(self, ticket_str, profit):
        ticket = int(ticket_str)
        for row in range(self.open_positions_model.rowCount()):
            if int(self.open_positions_model.item(row, 0).text()) == ticket:
                pl_item = QStandardItem(f"{profit:+.2f}")
                pl_item.setForeground(QColor("lightgreen") if profit >= 0 else QColor("salmon"))
                self.open_positions_model.setItem(row, 5, pl_item)
                break

    @Slot(dict)
    def move_to_history(self, data):
        ticket = data['ticket']
        if ticket not in self.gui_open_tickets: return
        self.gui_open_tickets.remove(ticket)
        self.worker.strategy_params['gui_tickets'] = self.gui_open_tickets
        for row in range(self.open_positions_model.rowCount()):
            if int(self.open_positions_model.item(row, 0).text()) == ticket:
                trade_row = [self.open_positions_model.item(row, col).clone() for col in range(self.open_positions_model.columnCount())]
                
                final_pl = data['final_pl']
                
                # Determine Status and Color
                if final_pl >= 0:
                    color = QColor("#4CAF50") # Green
                    status_text = "WIN"
                else:
                    color = QColor("#d32f2f") # Red
                    status_text = "LOSS"

                # Update P/L Item (Index 5)
                pl_item = QStandardItem(f"{final_pl:+.2f}")
                pl_item.setForeground(color)
                pl_item.setTextAlignment(Qt.AlignCenter)
                trade_row[5] = pl_item
                
                # Create Status Item (Index 6) - Badge Style
                status_item = QStandardItem(status_text)
                status_item.setForeground(QColor("white"))
                status_item.setBackground(color)
                status_item.setTextAlignment(Qt.AlignCenter)
                
                trade_row.append(status_item)
                
                self.closed_trades_model.appendRow(trade_row)
                self.open_positions_model.removeRow(row)
                break

    @Slot(str, str)
    def show_message(self, title, text):
        QMessageBox.information(self, title, text)

    def manual_buy(self):
        self.worker.strategy_params.update(self.get_risk_params())
        self.worker.execute_trade("Buy")

    def manual_sell(self):
        self.worker.strategy_params.update(self.get_risk_params())
        self.worker.execute_trade("Sell")

    def get_risk_params(self):
        try:
            return {
                'tp': float(self.tp_input.text()),
                'sl': float(self.sl_input.text()),
                'max_pos': int(self.max_pos_input.text()),
                'lot_size': float(self.lot_size_input.text())
            }
        except (ValueError, TypeError):
            self.show_message("Error", "Invalid values in Risk Management.")
            return {}

    def toggle_autotrade(self):
        if not self.worker.autotrade_enabled:
            params = self.get_risk_params()
            if not params: return
            try:
                # Validate and parse times
                start_time_str = self.start_time_input.text()
                end_time_str = self.end_time_input.text()
                params['start_time'] = datetime.strptime(start_time_str, "%H:%M").time()
                params['end_time'] = datetime.strptime(end_time_str, "%H:%M").time()

                params['strategy'] = self.strategy_combo.currentText()
                params['timeframe'] = self.worker.timeframe_map[self.timeframe_combo.currentText()]
                if self.param1_input.isEnabled():
                    params['param1'] = float(self.param1_input.text()) if params['strategy'] in ("SMC", "ICT Trader") else int(self.param1_input.text())
                if self.param2_input.isEnabled():
                    params['param2'] = float(self.param2_input.text()) if params['strategy'] in ("SMC", "ICT Trader") else int(self.param2_input.text())
                if self.param3_input.isEnabled(): params['param3'] = int(self.param3_input.text())
                
                # Include Scalper EMA if set, default to 21
                params['scalper_ema'] = getattr(self, 'scalper_ema', 21)

            except (ValueError, TypeError, KeyError):
                self.show_message("Error", "Invalid values in Strategy Settings. Ensure time is HH:MM.")
                return
            self.worker.strategy_params = params
            self.worker.autotrade_enabled = True
            self.autotrade_button.setText("Stop Auto Trading")
            self.autotrade_button.setObjectName("stopButton")
            self.autotrade_button.style().unpolish(self.autotrade_button); self.autotrade_button.style().polish(self.autotrade_button)
        else:
            self.worker.autotrade_enabled = False
            self.worker.last_trade_action = None
            self.autotrade_button.setText("Start Auto Trading")
            self.autotrade_button.setObjectName("startButton")
            self.autotrade_button.style().unpolish(self.autotrade_button); self.autotrade_button.style().polish(self.autotrade_button)

    def update_ui_for_strategy(self):
        strategy = self.strategy_combo.currentText()
        is_scalper = (strategy == "Gold M5 Scalper")
        is_ict = (strategy == "ICT Trader")
        is_ict_scalper = (strategy == "ICT Gold Scalping")
        is_gold_new = (strategy == "Gold Scalping New")
        is_smc = (strategy == "SMC")

        # Strategies that calculate their own Lot Size, TP, and SL internally
        # We disable these inputs to prevent confusion, as user values here are ignored.
        uses_internal_risk = (is_smc or is_ict or is_gold_new)
        
        self.lot_size_input.setEnabled(not uses_internal_risk)
        self.tp_input.setEnabled(not uses_internal_risk)
        self.sl_input.setEnabled(not uses_internal_risk)

        self.timeframe_combo.setEnabled(not is_scalper)
        if is_scalper: 
            self.timeframe_combo.setCurrentText("5 Minutes (M5)")
            # Check if sender is the combobox to avoid popup on startup or programmatic changes not by user
            # But since update_ui_for_strategy is manually called in init, we can just check if window has been shown
            # or simpler, just show it. Users usually won't see it on startup because default is not Scalper.
            if self.sender() == self.strategy_combo:
                dialog = ScalperSettingsDialog(self)
                if dialog.exec():
                    self.apply_scalper_settings(dialog.get_settings())

        # Disable parameter inputs for ICT strategies and Gold Scalping New
        params_enabled = not (is_scalper or is_ict_scalper or is_gold_new)
        # Re-enable for SMC but with specific meanings
        if is_smc: params_enabled = True

        self.param1_label.setEnabled(params_enabled)
        self.param1_input.setEnabled(params_enabled)
        self.param2_label.setEnabled(params_enabled)
        self.param2_input.setEnabled(params_enabled)
        
        # Specific handling for Gold M5 Scalper's param3
        self.param3_label.setEnabled(is_scalper)
        self.param3_input.setEnabled(is_scalper)

        if params_enabled:
            if strategy == "MA Crossover":
                self.param1_label.setText("Short MA Period:")
                self.param2_label.setText("Long MA Period:")
            elif strategy == "Trend Following":
                self.param1_label.setText("Signal MA Period:")
                self.param2_label.setText("Trend MA Period:")
            elif strategy in ("SMC", "ICT Trader"):
                self.param1_label.setText("Risk (%):")
                self.param2_label.setText("RR Ratio:")
                self.param1_input.setText("1")
                self.param2_input.setText("2")

    def apply_scalper_settings(self, settings):
        # Store EMA for the worker (default 21)
        self.scalper_ema = settings['ema']
        
        # Apply Risk to GUI fields
        risk = settings['risk']
        if risk == "Low":
            self.tp_input.setText("1.5")
            self.sl_input.setText("1.0")
        elif risk == "Medium":
            self.tp_input.setText("3.0")
            self.sl_input.setText("2.0")
        elif risk == "High":
            self.tp_input.setText("6.0")
            self.sl_input.setText("4.0")

    def get_config_path(self):
        # If running as exe, look in the same folder as the exe
        if getattr(sys, 'frozen', False):
            application_path = os.path.dirname(sys.executable)
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(application_path, 'settings.json')

    def save_settings(self):
        settings = {
            "tp": self.tp_input.text(),
            "sl": self.sl_input.text(),
            "max_pos": self.max_pos_input.text(),
            "lot_size": self.lot_size_input.text(),
            "strategy": self.strategy_combo.currentText(),
            "timeframe": self.timeframe_combo.currentText(),
            "start_time": self.start_time_input.text(),
            "end_time": self.end_time_input.text(),
            "param1": self.param1_input.text(),
            "param2": self.param2_input.text(),
            "param3": self.param3_input.text(),
            "symbol": self.symbol_input.text(),
            # Save scalper settings if they exist
            "scalper_ema": getattr(self, 'scalper_ema', 21)
        }
        try:
            with open(self.get_config_path(), 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def load_settings(self):
        path = self.get_config_path()
        if not os.path.exists(path):
            return
        
        try:
            with open(path, 'r') as f:
                settings = json.load(f)
                
            if "tp" in settings: self.tp_input.setText(settings["tp"])
            if "sl" in settings: self.sl_input.setText(settings["sl"])
            if "max_pos" in settings: self.max_pos_input.setText(settings["max_pos"])
            if "lot_size" in settings: self.lot_size_input.setText(settings["lot_size"])
            if "timeframe" in settings: self.timeframe_combo.setCurrentText(settings["timeframe"])
            if "start_time" in settings: self.start_time_input.setText(settings["start_time"])
            if "end_time" in settings: self.end_time_input.setText(settings["end_time"])
            if "param1" in settings: self.param1_input.setText(settings["param1"])
            if "param2" in settings: self.param2_input.setText(settings["param2"])
            if "param3" in settings: self.param3_input.setText(settings["param3"])
            if "symbol" in settings: self.symbol_input.setText(settings["symbol"])
            if "scalper_ema" in settings: self.scalper_ema = settings["scalper_ema"]
            
            # Set strategy last to trigger UI updates, but avoid popup during load
            if "strategy" in settings: 
                # Temporarily block signals to prevent popup during loading
                self.strategy_combo.blockSignals(True)
                self.strategy_combo.setCurrentText(settings["strategy"])
                self.strategy_combo.blockSignals(False)
                # Manually update UI without triggering the popup logic (which checks sender)
                self.update_ui_for_strategy()
            
        except Exception as e:
            print(f"Failed to load settings: {e}")

    def closeEvent(self, event):
        self.save_settings()
        self.worker.autotrade_enabled = False
        mt5.shutdown()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Allow Ctrl+C to close the application
    signal.signal(signal.SIGINT, lambda sig, frame: app.quit())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
