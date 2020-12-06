import tensorflow as tf
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation as fani
import numpy as np

# the animator class keeps a list of losses from different epochs,
# and displays this as a graph in the browser.
# this helps to decide whether model training should be halted
class animator:

    def __init__(self, max_history=5, max_loss=None):
        # we keep a history of all the losses of the last 5 epochs
        self.max_history = max_history
        
        # creating the graph, which gets displayed in the push function
        self.fig = go.FigureWidget()
        self.fig.add_scatter()
        
        # we start with an empty history
        self.history = list(np.zeros(max_history))

    def push(self, loss):
        # we add our new loss to the history, whilst removing the last
        self.history = self.history[1:]
        self.history += [loss]
        
        # here we update the graph and display it in the browser
        with self.fig.batch_update():
            self.fig.data[0].y = self.history
            self.fig.show()
