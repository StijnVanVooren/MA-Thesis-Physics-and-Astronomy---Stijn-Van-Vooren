import sys
from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6 import QtGui

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.colors as colors

from os.path import dirname, abspath




loader = QUiLoader();

app = QtWidgets.QApplication(sys.argv)
window = loader.load("network_triplets.ui", None) # Loads UI, but at RUNTIME!!
window2 = loader.load("plot_settings.ui", None) # Loads UI, but at RUNTIME!!

window.show()
window2.show()
app.exec()