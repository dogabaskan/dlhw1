from typing import List
import numpy as np
from dataclasses import dataclass, field
import plotly.graph_objects as go
import ipywidgets as widgets


@dataclass
class Logger():
    """Logger for rendering and logging metrics"""

    epoch_train_accuracy: List[float] = field(default_factory=list)
    epoch_eval_accuracy: List[float] = field(default_factory=list)
    iter_train_accuracy: List[float] = field(default_factory=list)
    iter_train_loss: List[float] = field(default_factory=list)
    smooth_window_len: int = 50
    verbose: bool = True
    live_figure_update: bool = False

    def __post_init__(self):
        self.iteration_figure = self._make_figure(
            "iteration", "Train Iterations", ["accuracy", "nll loss"])
        self.epoch_figure = self._make_figure("epoch", "Per Epoch Accuracies", ["train", "eval"])

    def render(self) -> widgets.HBox:
        """ Display iteration and epoch figures

        Returns:
            widgets.HBox: Horizontal Box of figures
        """
        return widgets.HBox(
            [self.iteration_figure, self.epoch_figure]
        )

    @staticmethod
    def _make_figure(x_axis_name: str, title: str, trace_names: List[str]) -> go.FigureWidget:
        """ Generate scatter plot

        Args:
            x_axis_name (str): Name of the x axis
            title (str): Title of the plot
            trace_names (List[str]): Legend names

        Returns:
            go.FigureWidget: Scatter figure
        """
        fig = go.FigureWidget()
        fig.update_layout(dict(
            template="none",
            width=500,
            xaxis=dict(title=x_axis_name),
            title=title
        ))
        for name in trace_names:
            fig.add_trace(go.Scatter(x=[], y=[], name=name, mode="markers+lines"))
        return fig

    def log_iteration(self, epoch: int, iteration: int) -> None:
        """ Write or render iteration specific metrics

        Args:
            epoch (int): Epoch number
            iteration (int): Iteration number
        """
        if iteration % self.smooth_window_len == 0:
            if self.verbose:
                print("Epoch: {}, Iteration: {}, Train loss: {:.4f}, Train acc: {:.4f}".format(
                    epoch,
                    iteration,
                    np.mean(self.iter_train_loss[-self.smooth_window_len:]),
                    np.mean(self.iter_train_accuracy[-self.smooth_window_len:])))
            if self.live_figure_update:
                x_axis = np.arange(len(self.iter_train_accuracy),
                                   step=self.smooth_window_len) * self.smooth_window_len
                self.iteration_figure.data[0].x = x_axis
                self.iteration_figure.data[0].y = self.smooth(self.iter_train_accuracy)
                self.iteration_figure.data[1].x = x_axis
                self.iteration_figure.data[1].y = self.smooth(self.iter_train_loss)

    def smooth(self, array: np.ndarray) -> np.ndarray:
        """ Smoothing function with stride

        Args:
            array (np.ndarray): 1D array of metric to smooth

        Returns:
            np.ndarray: Smoothed array
        """
        return np.convolve(array, np.ones(self.smooth_window_len)/self.smooth_window_len, mode="valid")[::self.smooth_window_len]

    def log_epoch(self, epoch: int) -> None:
        """ Write or render epoch specific metrics

        Args:
            epoch (int): Epoch number
        """
        if self.verbose:
            print("- " * 20)
            print("Epoch: {}, Train acc: {}, Eval acc: {}".format(
                epoch,
                self.epoch_train_accuracy[-1],
                self.epoch_eval_accuracy[-1]))

        if self.live_figure_update:
            x_axis = np.arange(len(self.epoch_train_accuracy))
            self.epoch_figure.data[0].x = x_axis
            self.epoch_figure.data[0].y = self.epoch_train_accuracy
            self.epoch_figure.data[1].x = x_axis
            self.epoch_figure.data[1].y = self.epoch_eval_accuracy

    @staticmethod
    def render_confusion_matrix(confusion_matrix: np.ndarray) -> go.FigureWidget:
        """ Plot confusion matrix

        Args:
            confusion_matrix (np.ndarray): 2D Confusion matrix array

        Returns:
            go.FigureWidget: Heatmap figure
        """
        fig = go.FigureWidget()
        fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            hoverongaps=False))
        fig.update_layout(dict(
            template="none",
            width=500,
            xaxis=dict(title="True Labels"),
            yaxis=dict(title="Predicted Labels", autorange="reversed"),
            title="Confusion Matrix"
        ))
        return fig
