from strategies.smc import run_ict_logic


def run_ict_trader_logic(worker, symbol, timeframe):
    params = getattr(worker, "strategy_params", {})
    risk_percent = params.get("param1", 1.0)
    rr_ratio = params.get("param2", 2.0)
    run_ict_logic(worker, symbol, timeframe, risk_percent, rr_ratio)
