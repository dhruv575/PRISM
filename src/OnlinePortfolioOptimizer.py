import torch
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime
import numpy as np
import seaborn as sns
from tqdm import tqdm
import statsmodels.api as sm
from IPython.display import display, clear_output
from matplotlib.animation import FFMpegWriter

class OnlinePortfolioOptimizer:
    def __init__(
        self,
        returns: torch.tensor,
        window_size: int = 21,
        alphas: torch.tensor = torch.tensor([1.0, 1.0, 0.5, 0.25]),
        enp_min: float = 5.0,
        enp_max: float = 20.0,
        lr: float = 0.5,
        eps: float = 1e-8,
        factor_data: pd.DataFrame = None,  # FF3 or CAPM factors
        benchmark_returns: dict = None,
    ):
        self.returns = returns
        self.window_size = window_size
        self.alphas = alphas
        self.enp_min = enp_min
        self.enp_max = enp_max
        self.lr = lr
        self.eps = eps

        self.n_assets = returns.shape[1] - 1
        self.weights = torch.rand(self.n_assets, requires_grad=True)
        self.weights_log = torch.zeros((returns.shape[0], self.n_assets))
        self.return_logs = torch.zeros(returns.shape[0])
        self.rolling_return_list = []

        self.optimizer = None
        self.factor_data = factor_data
        self.benchmark_returns = benchmark_returns or {}
        self.rf = returns["rf"]

    def calculate_hhi(self, weights):
        return torch.sum(weights ** 2)

    def concentration_penalty(self, weights):
        hhi = self.calculate_hhi(weights)
        enp = 1.0 / (hhi + self.eps)
        return torch.relu(self.enp_min - enp) + torch.relu(enp - self.enp_max)

    def calculate_sortino(self, returns, min_acceptable_return):
        excess_returns = returns - min_acceptable_return
        downside = torch.where(excess_returns < 0, excess_returns, torch.tensor(0.0))
        return torch.mean(excess_returns, dim=0) / (torch.std(downside) + self.eps)

    def calculate_max_drawdown(self, returns):
        cum_returns = (returns + 1).cumprod(dim=0)
        return (cum_returns.max() - cum_returns[-1]) / (cum_returns.max() + self.eps)

    def calculate_turnover(self, new_weights, prev_weights):
        return torch.sum(torch.abs(new_weights - prev_weights)) / 2

    def calculate_objective(self, returns, rf, new_weights, prev_weights):
        sortino = self.calculate_sortino(returns, rf)
        max_dd = self.calculate_max_drawdown(returns)
        turnover = self.calculate_turnover(new_weights, prev_weights)
        concentration = self.concentration_penalty(new_weights)
        return (
            self.alphas[0] * sortino
            - self.alphas[1] * max_dd
            - self.alphas[2] * turnover
            - self.alphas[3] * concentration
        )

    def run(self, plot_live_results=True, save_path="optimization.mp4"):
        self.fig = plt.figure(figsize=(20, 18))
        self.loss_history = {k: [] for k in ["sortino", "mdd", "turnover", "enp"]}
        self.optimizer = torch.optim.SGD([self.weights], lr=self.lr)

        metadata = dict(title="Portfolio Optimization", artist="Matplotlib")

        for i, date in enumerate(self.returns.index):
            if not plot_live_results:
                print(f"Processing {date} ({i} / {self.returns.shape[0]})")

            with torch.no_grad():
                self.weights += self.eps * torch.randn_like(self.weights)

            normalized_weights = torch.nn.functional.softmax(self.weights, dim=0)
            daily_returns = torch.tensor(
                self.returns.loc[date].T[:-1], dtype=torch.float32
            )
            ret = torch.dot(normalized_weights, daily_returns)
            self.return_logs[i] = ret.detach()
            self.rolling_return_list.append(ret)

            if len(self.rolling_return_list) > self.window_size:
                self.rolling_return_list.pop(0)
                past_returns = torch.stack(self.rolling_return_list)
                past_rf = torch.tensor(
                    self.rf.iloc[max(0, i - self.window_size):i].values,
                    dtype=torch.float32,
                )

                sortino = self.calculate_sortino(past_returns, past_rf)
                mdd = self.calculate_max_drawdown(past_returns)
                turnover = self.calculate_turnover(normalized_weights, self.weights_log[i - 1])
                concentration = self.concentration_penalty(normalized_weights)

                # because i want to visualize ENP and not concentration
                hhi = torch.sum(normalized_weights**2)
                enp = 1.0/(hhi + self.eps)

                self.loss_history["sortino"].append(sortino.item())
                self.loss_history["mdd"].append(mdd.item())
                self.loss_history["turnover"].append(turnover.item())
                self.loss_history["enp"].append(enp.item())

                total_objective = (
                    self.alphas[0] * sortino
                    - self.alphas[1] * mdd
                    - self.alphas[2] * turnover
                    - self.alphas[3] * concentration
                )

                objective = -total_objective
                self.optimizer.zero_grad()
                objective.backward(retain_graph=True)
                self.optimizer.step()
            else:
                for k in self.loss_history:
                    self.loss_history[k].append(np.nan)

            self.weights_log[i] = normalized_weights

            if plot_live_results and i % 5 == 0:
                clear_output(wait=True)
                self._update_live_plot(i)
                display(self.fig)
                self.fig.canvas.draw()

        print(f"Optimization completed.")
        return self.return_logs, self.weights_log
    

    # def run(
    #         self,
    #         plot_live_results = False,
    #         fig = None,
    #         ax = None,
    #     ):

    #     print("Initializing optimization...")
    #     self.optimizer = torch.optim.SGD([self.weights], lr=self.lr)

    #     for i, date in tqdm(enumerate(self.returns.index)):
    #         # if i % 5 == 0:
    #         #     print(f"Step {i} of {self.returns.shape[0]}", end="\r")

    #         # with torch.no_grad():
    #         #     self.weights += self.eps * torch.randn_like(self.weights)

    #         normalized_weights = torch.nn.functional.softmax(self.weights, dim=0)
    #         daily_returns = torch.tensor(
    #             self.returns.loc[date].T[:-1], dtype=torch.float32
    #         )
    #         ret = torch.dot(normalized_weights, daily_returns)

    #         self.return_logs[i] = ret.detach()
    #         self.rolling_return_list.append(ret)

    #         if len(self.rolling_return_list) > self.window_size:
    #             self.rolling_return_list.pop(0)
    #             past_returns = torch.stack(self.rolling_return_list)
    #             past_rf = torch.tensor(
    #                 self.rf.iloc[max(0, i - self.window_size):i].values,
    #                 dtype=torch.float32,
    #             )
    #             objective = -self.calculate_objective(
    #                 past_returns,
    #                 past_rf,
    #                 normalized_weights,
    #                 self.weights_log[i - 1],
    #             )
    #             self.optimizer.zero_grad()
    #             objective.backward(retain_graph=True)
    #             self.optimizer.step()

    #         self.weights_log[i] = normalized_weights

    #     print("Optimization completed.")
    #     return self.return_logs, self.weights_log


    def evaluate_sharpe(self):
        excess = self.return_logs.numpy() - self.rf.values
        return np.mean(excess) / (np.std(excess) + self.eps)

    def evaluate_sortino(self):
        excess = self.return_logs.numpy() - self.rf.values
        downside_risk = np.std(excess[excess < 0])
        mean_excess = np.mean(excess)
        return mean_excess / downside_risk if downside_risk > 0 else 0.0

    def evaluate_max_drawdown(self):
        cr = np.cumprod(self.return_logs.numpy() + 1)
        peak = np.maximum.accumulate(cr)
        return np.max((peak - cr) / (peak + self.eps))

    def evaluate_alpha(self, model="CAPM"):
        y = np.asarray(
            (self.return_logs.numpy() - self.rf.values)
        )
        if model == "CAPM":
            X = np.asarray(
                (self.factor_data[["mktrf"]])
            )
        elif model == "FF3":
            X = np.asarray(
                (self.factor_data[["mktrf", "smb", "hml"]])
            )
        else:
            raise ValueError("Model must be 'CAPM' or 'FF3'")

        X = sm.add_constant(X)
        result = sm.OLS(y, X).fit()
        return result.params[0], result

    def plot_results(self, top_n=5, title_suffix=""):
        weights_np = self.weights_log.detach().numpy()
        dates = self.returns.index
        opt_returns = self.return_logs.numpy()

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].hist(opt_returns, bins=50, alpha=0.5, label='Optimized', color='red')
        axes[1].plot(dates, np.cumprod(opt_returns + 1), label='Optimized', color='red')

        for name, b_returns in self.benchmark_returns.items():
            axes[0].hist(b_returns, bins=50, alpha=0.5, label=name)
            axes[1].plot(dates, np.cumprod(b_returns + 1), label=name)

        axes[0].set_title('Return Distribution')
        axes[1].set_title('Cumulative Returns')
        for ax in axes: ax.legend()
        fig.suptitle(f"Performance Comparison {title_suffix}", fontsize=16)
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        top_assets = np.argsort(weights_np[-1])[-top_n:]
        for i in range(weights_np.shape[1]):
            label = self.returns.columns[i] if i in top_assets else None
            lw = 2 if i in top_assets else 0.5
            axes[0].plot(dates, weights_np[:, i], label=label, linewidth=lw)

        axes[0].xaxis.set_major_locator(mdates.YearLocator(2))
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[0].set_title("Weights Over Time")
        axes[0].legend()

        axes[1].hist(weights_np[-1], bins=100, log=True, color='blue', alpha=0.7)
        axes[1].set_title("Final Day Weight Distribution")
        plt.tight_layout()
        plt.show()

    def _update_live_plot(self, i):
        self.fig.clf()
        gs = self.fig.add_gridspec(6, 2)

        # Title + Hyperparams Text
        ax_title = self.fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        ax_title.set_title("Optimization Results", fontsize=25, weight='bold')
        summary = (
            f"Alphas: {self.alphas.tolist()} | "
            f"Window Size: {self.window_size} | "
            f"ENP Range: [{self.enp_min}, {self.enp_max}] | "
            f"LR: {self.lr} | # Assets: {self.n_assets} | "
            f"Date: {self.returns.index[i].strftime('%Y-%m-%d')}"
        )
        ax_title.text(0.5, 0.2, summary, ha='center', fontsize=20)

        # Cumulative Return Comparison
        ax_cumret = self.fig.add_subplot(gs[1, :])
        cum_returns = (self.return_logs[:i+1].numpy() + 1).cumprod()
        ax_cumret.plot(self.returns.index[:i+1], cum_returns, label="Optimized", color="red")
        if "S&P500" in self.benchmark_returns:
            spx = (self.benchmark_returns["S&P500"][:i+1] + 1).cumprod()
            ax_cumret.plot(self.returns.index[:i+1], spx, label="S&P 500", color="C0")
        ax_cumret.set_title("Cumulative Portfolio Value vs Benchmark")
        ax_cumret.legend()

        # Loss Components Grid (2x2)
        keys = list(self.loss_history.keys())
        for idx, key in enumerate(keys):
            ax = self.fig.add_subplot(gs[2 + idx // 2, idx % 2])
            ax.plot(self.loss_history[key][:i+1], label=key, color = f"C{idx}", marker = 'o')
            ax.set_title(f"{key.capitalize()} over Time")
            if key == "enp":
                if self.loss_history[key][i] <= self.enp_max * 2:
                    ax.set_ylim(0, self.enp_max * 2)

                ax.axhline(self.enp_min, linestyle='--', color='gray', alpha=0.5, label="ENP Min")
                ax.axhline(self.enp_max, linestyle='--', color='gray', alpha=0.5, label="ENP Max")
                ax.legend()

        # Weights Distribution (left)
        ax_weights_dist = self.fig.add_subplot(gs[4, 0])
        ax_weights_dist.hist(
            self.weights_log[i].detach().numpy(), 
            bins=50, 
            color='orange', 
            alpha=0.7,
            log=(self.n_assets > 99)
        )
        ax_weights_dist.set_title("Weight Distribution")

        # Weights Evolution (right)
        ax_weights = self.fig.add_subplot(gs[4, 1])
        weights_np = self.weights_log[:i+1].detach().numpy()
        final_weights = weights_np[-1]
        top_50 = np.argsort(final_weights)[-50:]
        top_5 = top_50[-5:]
        for idx in top_50:
            label = self.returns.columns[idx] if idx in top_5 else None
            lw = 2 if idx in top_5 else 0.5
            ax_weights.plot(self.returns.index[:i+1], weights_np[:, idx], label=label, linewidth=lw)
        if len(top_5) > 0:
            ax_weights.legend(fontsize=6)
        ax_weights.set_title("Weights Evolution (Top 50)")
        ax_weights.set_xticklabels([])

        # Return Distribution (bottom full-width)
        ax_ret_dist = self.fig.add_subplot(gs[5, :])
        rets = self.return_logs[:i+1].numpy()
        rf_today = self.rf.iloc[i]
        bins = 50
        hist, bin_edges = np.histogram(rets, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        bar_colors = ['red' if center < rf_today else 'green' for center in bin_centers]

        ax_ret_dist.bar(
            bin_centers,
            hist,
            width=(bin_edges[1] - bin_edges[0]),
            color=bar_colors,
            edgecolor='black',
            alpha=0.3
        )

        ax_ret_dist.axvline(
            rf_today,
            color='red',
            linestyle='--',
            label=f'RF Rate: {rf_today:.4f}'
        )
        ax_ret_dist.set_title("Return Distribution with Daily RF Threshold")
        ax_ret_dist.set_xlim(-0.1, 0.1)
        ax_ret_dist.legend()
        plt.tight_layout()