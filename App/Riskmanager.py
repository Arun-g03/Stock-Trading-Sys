import traceback



class RiskManager:
    def __init__(self, risk_per_trade: float, max_drawdown: float):
        """Initialize risk parameters such as risk per trade and max portfolio drawdown."""
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
    
    # 🔹 Stop-Loss & Take-Profit Calculation
    def calculate_stop_loss(self, entry_price: float, strategy: str) -> float:
        """Determine stop-loss price based on strategy (ATR, percentage, support levels)."""
        if strategy == "percentage":
            return entry_price * 0.95  # 5% stop loss
        elif strategy == "atr":
            return entry_price - (entry_price * 0.015)  # Using ATR multiplier
        return entry_price * 0.97  # Default 3% stop loss
    
    def calculate_take_profit(self, entry_price: float, strategy: str) -> float:
        """Determine take-profit price based on risk-reward ratio and strategy."""
        risk_reward_ratio = 2.0  # Risk:Reward = 1:2
        stop_loss = self.calculate_stop_loss(entry_price, strategy)
        risk = entry_price - stop_loss
        return entry_price + (risk * risk_reward_ratio)
    
    def apply_trailing_stop(self, entry_price: float, current_price: float) -> float:
        """Dynamically adjust stop-loss to lock in profits as price moves up."""
        trailing_percentage = 0.02  # 2% trailing stop
        if current_price > entry_price:
            return current_price * (1 - trailing_percentage)
        return self.calculate_stop_loss(entry_price, "percentage")
    
    # 🔹 Position Sizing
    def calculate_position_size(self, capital: float, risk_per_trade: float, atr: float) -> int:
        """Determine position size based on capital and volatility-adjusted risk."""
        risk_amount = capital * risk_per_trade
        position_size = int(risk_amount / (atr * 1.5))  # Using ATR for volatility adjustment
        return max(1, min(position_size, int(capital * 0.1)))  # Cap at 10% of capital
    
    # 🔹 Trade Risk Assessment
    def assess_trade_risk(self, ticker: str, liquidity_data: dict) -> bool:
        """Check if the stock is tradeable based on volume, volatility, and portfolio exposure."""
        min_volume = 100000
        max_spread = 0.02
        
        volume = liquidity_data.get('volume', 0)
        spread = liquidity_data.get('spread', 1)
        
        if volume < min_volume or spread > max_spread:
            return False
        return True
    
    def check_max_drawdown(self, portfolio_value: float, max_drawdown: float) -> bool:
        """Prevent trading if portfolio loss exceeds max allowed drawdown."""
        initial_portfolio = 100000  # Example initial portfolio value
        current_drawdown = (initial_portfolio - portfolio_value) / initial_portfolio
        return current_drawdown <= max_drawdown
    
    # 🔹 Risk Monitoring & Auto-Management
    def auto_adjust_stops(self, open_trades: dict) -> dict:
        """Dynamically adjust stop-losses based on market conditions."""
        adjusted_trades = {}
        for ticker, trade in open_trades.items():
            current_price = trade.get('current_price', 0)
            entry_price = trade.get('entry_price', 0)
            adjusted_trades[ticker] = {
                'new_stop_loss': self.apply_trailing_stop(entry_price, current_price)
            }
        return adjusted_trades
    
    def enforce_risk_limits(self, trade_decisions: dict) -> dict:
        """Reject trades if risk rules are violated."""
        filtered_decisions = {}
        for ticker, decision in trade_decisions.items():
            position_size = decision.get('position_size', 0)
            risk_per_trade = decision.get('risk_amount', 0) / decision.get('capital', 1)
            
            if position_size > 0 and risk_per_trade <= self.risk_per_trade:
                filtered_decisions[ticker] = decision
                
        return filtered_decisions    
    def run(self, trade_data: dict, portfolio_value: float):
        """Run all risk checks and return the final risk assessment for a trade."""
        ticker = trade_data['ticker']
        entry_price = trade_data['entry_price']
        strategy = trade_data.get('strategy', 'percentage')
        capital = trade_data.get('capital', 10000)
        atr = trade_data.get('atr', 1.5)
        liquidity_data = trade_data.get('liquidity_data', {})
        
        risk_assessment = {
            "stop_loss": self.calculate_stop_loss(entry_price, strategy),
            "take_profit": self.calculate_take_profit(entry_price, strategy),
            "position_size": self.calculate_position_size(capital, self.risk_per_trade, atr),
            "trade_risk": self.assess_trade_risk(ticker, liquidity_data),
            "max_drawdown_check": self.check_max_drawdown(portfolio_value, self.max_drawdown),
        }
        
        return risk_assessment