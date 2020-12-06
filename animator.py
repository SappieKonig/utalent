import tensorflow as tf
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation as fani
import numpy as np

# the animator class keeps a list of losses from different epochs,
# and displays this as a graph in the browser.
# this helps to decide whether model training should be halted
class animator:

    def __init__(self, max_history=5, max_loss=None):
        self.max_history = max_history
        self.fig = go.FigureWidget()
        self.fig.add_scatter()
        self.history = list(np.zeros(max_history))
        self.max_loss = max_loss
        self.has_run = False

    def push(self, loss):
        if self.max_loss == None:
            if self.has_run == False:
                self.max_loss = loss
            else:
                Exception("Why the fuck has it not run yet?")
        self.history = self.history[1:]
        self.history += [loss]
        with self.fig.batch_update():
            self.fig.data[0].y = self.history
            self.fig.show()
