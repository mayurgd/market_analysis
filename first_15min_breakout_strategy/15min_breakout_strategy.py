import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple


class FiveMinuteBreakoutStrategy:
    """
    A comprehensive trading strategy class for 5-minute breakout trading.

    This strategy identifies breakouts from the first 15 minutes of trading
    and places trades based on swing point stop losses with configurable
    risk-reward ratios and fixed position size.
    """

    def __init__(
        self,
        initial_capital: float = 50000,
        risk_reward_ratio: float = 2.0,
        swing_window: int = 3,
        fixed_position_size: int = 75,  # Fixed position size in units
        first_15_min_start: time = time(9, 15),
        first_15_min_end: time = time(9, 30),
        trade_start_time: time = time(9, 30),
    ):
        """
        Initialize the trading strategy with configurable parameters.

        Args:
            initial_capital: Starting capital for backtesting
            risk_reward_ratio: Target profit vs stop loss ratio
            swing_window: Number of periods to look for swing points
            fixed_position_size: Fixed number of units to trade per position
            first_15_min_start: Start time for first 15 minutes range
            first_15_min_end: End time for first 15 minutes range
            trade_start_time: Time after which trades can be taken
        """
        self.initial_capital = initial_capital
        self.risk_reward_ratio = risk_reward_ratio
        self.swing_window = swing_window
        self.fixed_position_size = fixed_position_size
        self.first_15_min_start = first_15_min_start
        self.first_15_min_end = first_15_min_end
        self.trade_start_time = trade_start_time

        # Initialize results storage
        self.reset_results()

    def reset_results(self):
        """Reset all internal state for fresh backtesting."""
        self.trades = []
        self.visualization_trades = []
        self.equity_curve = []
        self.current_capital = self.initial_capital
        self.results = {}

    def convert_to_5min(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert 1-minute OHLCV data to 5-minute intervals.

        Args:
            df: DataFrame with 1-minute OHLCV data containing 'date', 'open', 'high', 'low', 'close'

        Returns:
            DataFrame with 5-minute OHLCV data
        """
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["date"])
        df.set_index("datetime", inplace=True)

        # Define OHLC aggregation rules
        ohlc_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
        }

        # Resample to 5-minute intervals
        df_5min = df.resample("5T").agg(ohlc_dict).dropna()

        # Reset index and create date column
        df_5min.reset_index(inplace=True)
        df_5min["date"] = df_5min["datetime"]

        return df_5min

    def find_swing_points(
        self, df: pd.DataFrame, window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Identify swing highs and lows using a rolling window approach.

        Args:
            df: DataFrame with OHLC data
            window: Override default swing window size

        Returns:
            DataFrame with swing_high and swing_low columns added
        """
        if window is None:
            window = self.swing_window

        df = df.copy()
        df["swing_high"] = np.nan
        df["swing_low"] = np.nan

        for i in range(window, len(df) - window):
            # Check for swing high
            current_high = df.iloc[i]["high"]
            is_swing_high = all(
                df.iloc[j]["high"] < current_high
                for j in range(i - window, i + window + 1)
                if j != i
            )

            if is_swing_high:
                df.iloc[i, df.columns.get_loc("swing_high")] = current_high

            # Check for swing low
            current_low = df.iloc[i]["low"]
            is_swing_low = all(
                df.iloc[j]["low"] > current_low
                for j in range(i - window, i + window + 1)
                if j != i
            )

            if is_swing_low:
                df.iloc[i, df.columns.get_loc("swing_low")] = current_low

        return df

    def get_position_size(self) -> int:
        """
        Return the fixed position size.

        Returns:
            Fixed position size in units
        """
        return self.fixed_position_size

    def get_first_15_min_range(
        self, day_data: pd.DataFrame
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Calculate the high and low of the first 15 minutes of trading.

        Args:
            day_data: Single day's trading data

        Returns:
            Tuple of (first_15_high, first_15_low, first_15_data)
        """
        first_15_mask = (day_data["datetime"].dt.time >= self.first_15_min_start) & (
            day_data["datetime"].dt.time <= self.first_15_min_end
        )
        first_15_data = day_data[first_15_mask]

        if len(first_15_data) < 1:
            return None, None, None

        first_15_high = first_15_data["high"].max()
        first_15_low = first_15_data["low"].min()

        return first_15_high, first_15_low, first_15_data

    def identify_trade_signal(
        self,
        current_price: float,
        first_15_high: float,
        first_15_low: float,
        prev_data: pd.DataFrame,
    ) -> Dict:
        """
        Identify if current price represents a valid trade signal.

        Args:
            current_price: Current close price
            first_15_high: High of first 15 minutes
            first_15_low: Low of first 15 minutes
            prev_data: Historical data up to current point

        Returns:
            Dictionary with trade signal information
        """
        trade_info = {
            "signal": None,
            "entry_price": None,
            "stop_loss": None,
            "target": None,
        }

        # Check for long breakout above first 15 min high
        if current_price > first_15_high:
            swing_lows = prev_data.dropna(subset=["swing_low"])
            if len(swing_lows) > 0:
                recent_swing_low = swing_lows.iloc[-1]["swing_low"]
                risk = current_price - recent_swing_low

                if risk > 0:  # Valid stop loss
                    trade_info.update(
                        {
                            "signal": "LONG",
                            "entry_price": current_price,
                            "stop_loss": recent_swing_low,
                            "target": current_price + (risk * self.risk_reward_ratio),
                        }
                    )

        # Check for short breakout below first 15 min low
        elif current_price < first_15_low:
            swing_highs = prev_data.dropna(subset=["swing_high"])
            if len(swing_highs) > 0:
                recent_swing_high = swing_highs.iloc[-1]["swing_high"]
                risk = recent_swing_high - current_price

                if risk > 0:  # Valid stop loss
                    trade_info.update(
                        {
                            "signal": "SHORT",
                            "entry_price": current_price,
                            "stop_loss": recent_swing_high,
                            "target": current_price - (risk * self.risk_reward_ratio),
                        }
                    )

        return trade_info

    def execute_trade(
        self,
        trade_info: Dict,
        entry_time: datetime,
        remaining_data: pd.DataFrame,
        day_data: pd.DataFrame,
        first_15_data: pd.DataFrame,
        swing_data: pd.DataFrame,
        first_15_high: float,
        first_15_low: float,
    ) -> Dict:
        """
        Execute a trade and find its exit point.

        Args:
            trade_info: Trade signal information
            entry_time: Entry timestamp
            remaining_data: Remaining candles in the day
            day_data: Full day's data
            first_15_data: First 15 minutes data
            swing_data: Swing point data
            first_15_high: First 15 min high
            first_15_low: First 15 min low

        Returns:
            Complete trade record
        """
        signal = trade_info["signal"]
        entry_price = trade_info["entry_price"]
        stop_loss = trade_info["stop_loss"]
        target = trade_info["target"]

        # Use fixed position size
        position_size = self.get_position_size()

        # Find exit point
        exit_price = None
        exit_reason = None
        exit_time = None

        # Check each subsequent candle for exit conditions
        for _, candle in remaining_data.iterrows():
            if signal == "LONG":
                if candle["high"] >= target:
                    exit_price = target
                    exit_reason = "TARGET"
                    exit_time = candle["datetime"]
                    break
                elif candle["low"] <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "STOP_LOSS"
                    exit_time = candle["datetime"]
                    break

            elif signal == "SHORT":
                if candle["low"] <= target:
                    exit_price = target
                    exit_reason = "TARGET"
                    exit_time = candle["datetime"]
                    break
                elif candle["high"] >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "STOP_LOSS"
                    exit_time = candle["datetime"]
                    break

        # If no exit found, close at end of day
        if exit_price is None:
            exit_price = remaining_data.iloc[-1]["close"]
            exit_reason = "EOD"
            exit_time = remaining_data.iloc[-1]["datetime"]

        # Calculate P&L
        if signal == "LONG":
            pnl_per_share = exit_price - entry_price
        else:  # SHORT
            pnl_per_share = entry_price - exit_price

        total_pnl = pnl_per_share * position_size
        self.current_capital += total_pnl

        # Create trade record
        trade_record = {
            "date": entry_time.date(),
            "signal": signal,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "stop_loss": stop_loss,
            "target": target,
            "exit_reason": exit_reason,
            "position_size": position_size,
            "pnl_per_share": pnl_per_share,
            "total_pnl": total_pnl,
            "capital_after_trade": self.current_capital,
            "first_15_high": first_15_high,
            "first_15_low": first_15_low,
            "risk_per_share": abs(entry_price - stop_loss),
            "reward_per_share": abs(target - entry_price),
        }

        # Store visualization data for first 5 trades
        if len(self.visualization_trades) < 5:
            viz_data = {
                "day_data": day_data.copy(),
                "trade_record": trade_record.copy(),
                "first_15_data": first_15_data.copy(),
                "swing_data": swing_data.copy(),
            }
            self.visualization_trades.append(viz_data)

        # Record equity curve point
        self.equity_curve.append(
            {
                "date": exit_time,
                "capital": self.current_capital,
                "trade_pnl": total_pnl,
            }
        )

        return trade_record

    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run the complete backtesting process.

        Args:
            df: DataFrame with 1-minute OHLCV data

        Returns:
            Dictionary containing backtest results and statistics
        """
        self.reset_results()

        # Convert to 5-minute data
        df_5min = self.convert_to_5min(df)

        # Ensure proper data types and sorting
        df_5min["datetime"] = pd.to_datetime(df_5min["date"])
        df_5min["date"] = df_5min["datetime"].dt.date
        df_5min = df_5min.sort_values("datetime").reset_index(drop=True)

        # Find swing points
        df_5min = self.find_swing_points(df_5min)

        # Process each trading day
        for date, day_data in df_5min.groupby("date"):
            day_data = day_data.reset_index(drop=True)

            # Get first 15 minutes range
            first_15_high, first_15_low, first_15_data = self.get_first_15_min_range(
                day_data
            )

            if first_15_high is None:
                continue

            # Get data after trade start time
            after_start_mask = day_data["datetime"].dt.time > self.trade_start_time
            after_start_data = day_data[after_start_mask].reset_index(drop=True)

            if len(after_start_data) == 0:
                continue

            # Look for trade signals (only one trade per day)
            for i, row in after_start_data.iterrows():
                current_idx = len(first_15_data) + i
                prev_data = day_data.iloc[: current_idx + 1]

                # Check for trade signal
                trade_info = self.identify_trade_signal(
                    row["close"], first_15_high, first_15_low, prev_data
                )

                if trade_info["signal"] is not None:
                    # Execute the trade
                    remaining_data = after_start_data.iloc[i + 1 :]

                    trade_record = self.execute_trade(
                        trade_info,
                        row["datetime"],
                        remaining_data,
                        day_data,
                        first_15_data,
                        prev_data,
                        first_15_high,
                        first_15_low,
                    )

                    if trade_record is not None:
                        self.trades.append(trade_record)
                        break  # Only one trade per day

        # Calculate and store results
        self.results = self._calculate_results()
        return self.results

    def _calculate_results(self) -> Dict:
        """Calculate comprehensive backtest statistics."""
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        if len(trades_df) == 0:
            return {
                "total_trades": 0,
                "win_ratio": 0,
                "total_return": 0,
                "final_capital": self.initial_capital,
                "trades_df": trades_df,
                "equity_curve": equity_df,
                "visualization_trades": self.visualization_trades,
            }

        # Calculate statistics
        winning_trades = trades_df[trades_df["total_pnl"] > 0]
        losing_trades = trades_df[trades_df["total_pnl"] <= 0]

        win_ratio = len(winning_trades) / len(trades_df) * 100
        total_return = (
            (self.current_capital - self.initial_capital) / self.initial_capital
        ) * 100
        max_drawdown = (
            self._calculate_max_drawdown(equity_df["capital"])
            if len(equity_df) > 0
            else 0
        )

        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.current_capital,
            "total_pnl": self.current_capital - self.initial_capital,
            "total_return": total_return,
            "total_trades": len(trades_df),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_ratio": win_ratio,
            "avg_win": (
                winning_trades["total_pnl"].mean() if len(winning_trades) > 0 else 0
            ),
            "avg_loss": (
                losing_trades["total_pnl"].mean() if len(losing_trades) > 0 else 0
            ),
            "max_drawdown": max_drawdown,
            "target_hits": len(trades_df[trades_df["exit_reason"] == "TARGET"]),
            "stop_loss_hits": len(trades_df[trades_df["exit_reason"] == "STOP_LOSS"]),
            "eod_exits": len(trades_df[trades_df["exit_reason"] == "EOD"]),
            "long_trades": len(trades_df[trades_df["signal"] == "LONG"]),
            "short_trades": len(trades_df[trades_df["signal"] == "SHORT"]),
            "trades_df": trades_df,
            "equity_curve": equity_df,
            "visualization_trades": self.visualization_trades,
            "fixed_position_size": self.fixed_position_size,
        }

    @staticmethod
    def _calculate_max_drawdown(capital_series: pd.Series) -> float:
        """Calculate maximum drawdown from capital series."""
        peak = capital_series.expanding().max()
        drawdown = (capital_series - peak) / peak * 100
        return drawdown.min()

    def plot_trade_setups(self, max_plots: int = 5):
        """Plot trade setups for visualization and verification."""
        if len(self.visualization_trades) == 0:
            print("No trades available for visualization")
            return

        num_plots = min(len(self.visualization_trades), max_plots)
        fig, axes = plt.subplots(num_plots, 1, figsize=(15, 4 * num_plots))

        if num_plots == 1:
            axes = [axes]

        for idx, trade_viz in enumerate(self.visualization_trades[:num_plots]):
            self._plot_single_trade_setup(axes[idx], trade_viz, idx + 1)

        plt.tight_layout()
        plt.show()

    def _plot_single_trade_setup(self, ax, trade_viz: Dict, trade_num: int):
        """Plot a single trade setup on the given axis."""
        day_data = trade_viz["day_data"]
        trade_record = trade_viz["trade_record"]
        first_15_data = trade_viz["first_15_data"]
        swing_data = trade_viz["swing_data"]

        # Plot simplified candlesticks
        for _, row in day_data.iterrows():
            color = "green" if row["close"] >= row["open"] else "red"
            # High-low line
            ax.plot(
                [row["datetime"], row["datetime"]],
                [row["low"], row["high"]],
                color="black",
                linewidth=1,
            )
            # Open-close line
            ax.plot(
                [row["datetime"], row["datetime"]],
                [row["open"], row["close"]],
                color=color,
                linewidth=3,
            )

        # Mark first 15 minutes zone
        first_15_start = first_15_data["datetime"].min()
        first_15_end = first_15_data["datetime"].max()
        ax.axvspan(
            first_15_start,
            first_15_end,
            alpha=0.2,
            color="yellow",
            label="First 15 min",
        )

        # Draw reference lines
        ax.axhline(
            y=trade_record["first_15_high"],
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f'First 15min High: {trade_record["first_15_high"]:.2f}',
        )
        ax.axhline(
            y=trade_record["first_15_low"],
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f'First 15min Low: {trade_record["first_15_low"]:.2f}',
        )

        # Mark swing points
        swing_highs = swing_data.dropna(subset=["swing_high"])
        swing_lows = swing_data.dropna(subset=["swing_low"])

        if len(swing_highs) > 0:
            ax.scatter(
                swing_highs["datetime"],
                swing_highs["swing_high"],
                color="red",
                marker="v",
                s=100,
                label="Swing Highs",
            )

        if len(swing_lows) > 0:
            ax.scatter(
                swing_lows["datetime"],
                swing_lows["swing_low"],
                color="green",
                marker="^",
                s=100,
                label="Swing Lows",
            )

        # Mark entry and exit points
        ax.scatter(
            trade_record["entry_time"],
            trade_record["entry_price"],
            color="purple",
            marker="o",
            s=150,
            label=f'{trade_record["signal"]} Entry: {trade_record["entry_price"]:.2f}',
        )

        exit_color = "green" if trade_record["total_pnl"] > 0 else "red"
        ax.scatter(
            trade_record["exit_time"],
            trade_record["exit_price"],
            color=exit_color,
            marker="x",
            s=150,
            label=f'Exit ({trade_record["exit_reason"]}): {trade_record["exit_price"]:.2f}',
        )

        # Draw stop loss and target lines
        ax.axhline(
            y=trade_record["stop_loss"],
            color="red",
            linestyle=":",
            alpha=0.7,
            label=f'Stop Loss: {trade_record["stop_loss"]:.2f}',
        )
        ax.axhline(
            y=trade_record["target"],
            color="green",
            linestyle=":",
            alpha=0.7,
            label=f'Target: {trade_record["target"]:.2f}',
        )

        # Formatting
        ax.set_title(
            f'Trade #{trade_num} - {trade_record["date"]} - {trade_record["signal"]} - '
            f'P&L: ₹{trade_record["total_pnl"]:.2f} (Position: {trade_record["position_size"]} units)',
            fontweight="bold",
        )
        ax.set_ylabel("Price")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot the equity curve and trade P&L distribution."""
        if len(self.equity_curve) == 0:
            print("No trades to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        equity_df = pd.DataFrame(self.equity_curve)

        # Equity curve
        ax1.plot(equity_df["date"], equity_df["capital"], linewidth=2, color="blue")
        ax1.axhline(
            y=self.initial_capital,
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Initial Capital",
        )
        ax1.set_title(
            f"Equity Curve - 5 Minute Strategy (Fixed Position: {self.fixed_position_size} units)",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("Capital (₹)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))

        # Trade P&L
        colors = ["green" if x > 0 else "red" for x in equity_df["trade_pnl"]]
        ax2.bar(range(len(equity_df)), equity_df["trade_pnl"], color=colors, alpha=0.7)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax2.set_title("Trade P&L", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Trade Number")
        ax2.set_ylabel("P&L (₹)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Equity curve saved to {save_path}")

        plt.show()

    def print_results(self):
        """Print comprehensive backtest results."""
        if not self.results:
            print("No results available. Run backtest first.")
            return

        print("\n=== 5-MINUTE BREAKOUT STRATEGY RESULTS (FIXED POSITION SIZE) ===")
        print(f"Fixed Position Size: {self.fixed_position_size} units")
        print(f"Initial Capital: ₹{self.results['initial_capital']:,.2f}")
        print(f"Final Capital: ₹{self.results['final_capital']:,.2f}")
        print(f"Total P&L: ₹{self.results['total_pnl']:,.2f}")
        print(f"Total Return: {self.results['total_return']:.2f}%")
        print(f"Max Drawdown: {self.results['max_drawdown']:.2f}%")
        print(f"Total Trades: {self.results['total_trades']}")
        print(f"Winning Trades: {self.results['winning_trades']}")
        print(f"Losing Trades: {self.results['losing_trades']}")
        print(f"Win Ratio: {self.results['win_ratio']:.2f}%")
        print(f"Average Win: ₹{self.results['avg_win']:,.2f}")
        print(f"Average Loss: ₹{self.results['avg_loss']:,.2f}")
        print(f"Target Hits: {self.results['target_hits']}")
        print(f"Stop Loss Hits: {self.results['stop_loss_hits']}")
        print(f"End of Day Exits: {self.results['eod_exits']}")
        print(f"Long Trades: {self.results['long_trades']}")
        print(f"Short Trades: {self.results['short_trades']}")

        # Show sample trades
        if len(self.results["trades_df"]) > 0:
            print("\n=== SAMPLE TRADES ===")
            sample_cols = [
                "date",
                "signal",
                "entry_price",
                "exit_price",
                "exit_reason",
                "position_size",
                "total_pnl",
                "capital_after_trade",
            ]
            print(self.results["trades_df"][sample_cols].head(10))

    def run_full_analysis(self, file_path: str, year_filter: int = 2024):
        """
        Complete analysis workflow: load data, backtest, and generate all visualizations.

        Args:
            file_path: Path to CSV file with 1-minute OHLCV data
            year_filter: Filter data to specific year
        """
        print("Loading 1-minute data...")
        df = pd.read_csv(file_path)
        df = df[df["year"] >= year_filter]
        df.rename(columns={"datetime": "date"}, inplace=True)

        print(f"Original 1-min data points: {len(df)}")

        # Run backtest
        print("Running backtest...")
        results = self.backtest(df)

        # Print results
        self.print_results()

        # Generate visualizations
        print("\n=== PLOTTING TRADE SETUPS ===")
        self.plot_trade_setups()

        if len(self.equity_curve) > 0:
            print("\n=== PLOTTING EQUITY CURVE ===")
            self.plot_equity_curve()

        return results


# Example usage
if __name__ == "__main__":
    # Initialize strategy with fixed position size of 75 units
    strategy = FiveMinuteBreakoutStrategy(
        initial_capital=50000,
        risk_reward_ratio=2.0,
        swing_window=3,
        fixed_position_size=75,  # Fixed position size
    )

    # Run complete analysis
    results = strategy.run_full_analysis("datasets/processed_nifty_data.csv")

    # Or run individual components
    # df = pd.read_csv("your_data.csv")
    # results = strategy.backtest(df)
    # strategy.print_results()
    # strategy.plot_trade_setups()
    # strategy.plot_equity_curve()
