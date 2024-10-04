"""
WiN_GUI.py

This GUI is designed to visualize the encoding from sample-based data into event-/spike-based data. 
It supports datasets with a specific structure, as detailed in the README.md. The GUI allows users 
to change the neuron model and its parameters on the fly and provides auditory feedback on the 
encoding results.

Features:
- Visualization of spike patterns
- Real-time adjustment of neuron model parameters
- Auditory feedback on encoding results
- Detailed description available at: https://www.sciencedirect.com/science/article/pii/S2352711024001304
- And [fill in the link to the v2 update publication]

Authors:
- Simon F. Muller-Cleve (v1, v2)
- Fernando M. Quintana (v1)
- Vittorio Fra (v1, v2)

Dependencies:
- numpy
- pandas
- torch
- matplotlib
- pydub
- PyQt6

Usage:
Run this script to launch the GUI. Ensure that the required dependencies are installed and the dataset 
is structured as specified in the README.md.

License:
This project is licensed under the GPL-3.0 License. See the LICENSE file for more details.

"""

import logging
import sys
from decimal import Decimal
from random import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm
from pydub import AudioSegment
from pydub.generators import Sawtooth
from PyQt6 import QtCore
from PyQt6.QtCore import QObject, Qt, QThread, QUrl
from PyQt6.QtGui import QColor
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDial,
                             QFileDialog, QGridLayout, QLabel, QMainWindow,
                             QPushButton, QSlider, QSplitter, QTableWidget,
                             QTableWidgetItem, QTabWidget, QWidget)

from utils.data_management import (create_directory, load_data,
                                   preprocess_data, split_data)
from utils.neuron_models import IZ_neuron, LIF_neuron, MN_neuron, RLIF_neuron
from utils.spike_pattern_classifier import classifySpikes, prepareDataset

WINDOW_WIDTH, WINDOW_HEIGTH = 1500, 750
DISPLAY_HEIGHT = 35
MIDPOINT_LIGHTNESS = 200
EXTREME_LIGHTNESS = 150


class EncodingCalc(QObject):
    """EncodingGUI's controller class."""

    signalData = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, parent=None):
        super(self.__class__, self).__init__()
        self.main_gui = parent
        self.main_gui.simulate_event.connect(self.simulate)

    @torch.no_grad()
    @QtCore.pyqtSlot()
    def simulate(self):
        if self.main_gui.active_class == -1:
            return

        params = {}
        for counter, (param_key, _) in enumerate(self.main_gui.parameter.items()):
            params[param_key] = self.main_gui.sliderValues[counter] / \
                float(self.main_gui.factor[counter])

        if self.main_gui.neuron_model_name == "Mihalas-Niebur":
            self.neurons = MN_neuron(
                len(self.main_gui.channels),
                params,
                dt=self.main_gui.dt / self.main_gui.dt_slider.value(),
            )

        elif self.main_gui.neuron_model_name == "Izhikevich":
            self.neurons = IZ_neuron(
                len(self.main_gui.channels),
                params,
                dt=self.main_gui.dt / self.main_gui.dt_slider.value(),
            )

        elif self.main_gui.neuron_model_name == "Leaky integrate-and-fire":
            self.neurons = LIF_neuron(
                len(self.main_gui.channels),
                params,
                dt=self.main_gui.dt / self.main_gui.dt_slider.value(),
            )

        elif self.main_gui.neuron_model_name == "Recurrent leaky integrate-and-fire":
            self.neurons = RLIF_neuron(
                len(self.main_gui.channels),
                params,
                dt=self.main_gui.dt / self.main_gui.dt_slider.value(),
            )

        else:
            raise ValueError("Select a valid neuron model.")

        sample = torch.where(self.main_gui.labels == self.main_gui.active_class)[
            0][self.main_gui.selectedRepetition]

        if self.main_gui.enable_data_splitting:
            input_data = self.main_gui.data_split[sample].unsqueeze(1)
        else:
            input_data = self.main_gui.data[sample].unsqueeze(1)

        # neuron ouput
        output = []
        for t in range(input_data.shape[0]):
            out = self.neurons(input_data[t])  # get output spikes
            local_output = []
            local_output.append(out.cpu().numpy())
            for state in self.neurons.state:
                # TODO make it indepenent of position in state tuple
                # first entry here is membrane potential and last spikes
                # rest are remaining state variables
                local_output.append(state.cpu().numpy())
            output.append(local_output)
        output = np.stack(output)

        if self.main_gui.enable_data_splitting:
            if self.main_gui.data_split is None:
                self.signalData.emit(
                    self.main_gui.data[sample].unsqueeze(1).cpu().numpy(), output)
            else:
                self.signalData.emit(
                    self.main_gui.data_split[sample].unsqueeze(1).cpu().numpy(), output)
        else:
            self.signalData.emit(
                self.main_gui.data[sample].unsqueeze(1).cpu().numpy(), output)


class WiN_GUI_Window(QMainWindow):

    """EncodingGUI's main window."""

    draw_event = QtCore.pyqtSignal()  # Signal used to update the plots
    simulate_event = QtCore.pyqtSignal()  # Signal used to trigger a new simulation
    # Signal used to trigger generation of new audio
    audio_event = QtCore.pyqtSignal()
    write_event = QtCore.pyqtSignal()

    def __init__(self):

        super().__init__()

        # Window creation
        self.setWindowTitle("WiN-GUI")
        self.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGTH)

        self.setWindowFlags(
            QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.CustomizeWindowHint
            | QtCore.Qt.WindowType.WindowTitleHint
            | QtCore.Qt.WindowType.WindowCloseButtonHint
            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
            | QtCore.Qt.WindowType.WindowMaximizeButtonHint
            # | QtCore.Qt.WindowType.WindowStaysOnTopHint  # enforce window in forground
        )

        # TODO inlcude removing tmp folder when closing GUI
        # create tmp path to store audio file
        self.tmp_path = "./tmp"
        # used to store tmp data (audio file, etc.)
        create_directory(self.tmp_path)

        # setting defaults
        self.upsample_fac = 1
        self.scale = 1

        self.dt = 1E-3  # 100Hz
        self.data_dict = None
        self.active_class = -1  # Not initialized

        self.enable_data_splitting = False
        self.normalizeData = False
        self.filterSignal = False
        self.startTrialAtNull = False
        self.neuron_model_name = "Mihalas-Niebur"
        self.dataFilename = None
        self.neuronStateVariables = None

        # Init the main layout
        self.generalLayout = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.generalLayout)

        # Add tabs
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.tabs.addTab(self.tab1, "Data Visualization")
        self.tabs.addTab(self.tab2, "Spike-Pattern Visualization")

        # Add tabs to the main layout
        self.generalLayout.addWidget(self.tabs)

        # Canvas pane in the first tab
        self.canvasLayout = QGridLayout(self.tab1)
        self.tab1.setLayout(self.canvasLayout)

        # Spike Pattern Visualizer in the second tab
        self.createSpikePatternVisualizer()

        # Parameters pane (always visible on the right side)
        self.parametersLayout = QGridLayout()
        parametersWidget = QWidget(self)
        parametersWidget.setLayout(self.parametersLayout)
        self.generalLayout.addWidget(parametersWidget)

        # Initialize GUI elements
        self.createCanvas()  # Now in the first tab
        self.loadParameter()
        self.createModelSelection()
        self.createDataSelection()
        self.createPreprocessingSelection()
        self.createAudioPushButtons()
        self.createParamSlider()

        # Encoding simulator creation and threading
        self.encoding_calc = EncodingCalc(self)
        self.thread = QThread(parent=self)
        self.encoding_calc.moveToThread(self.thread)

        # Connect signals and start the thread
        self.draw_event.connect(self.drawCanvas)
        self.audio_event.connect(self.spikeToAudio)
        self.write_event.connect(self.writeTable)
        self.encoding_calc.signalData.connect(self._updateCanvas)
        self.encoding_calc.signalData.connect(self._updateSpikesToAudio)
        self.encoding_calc.signalData.connect(
            self._updateSpikePatternClassification)

        self.thread.start()

    ######################
    # DATA VISUALIZATION #
    ######################

    def createAudioPushButtons(self):
        filename = f"{self.tmp_path}/spikeToAudio.wav"
        # self.eventsAudioStream = []  # TODO use this variable for the event audio stream
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        # TODO use local variable instead
        self.player.setSource(QUrl.fromLocalFile(filename))
        self.audio_output.setVolume(10)
        # player.play()

        self.play_button = QPushButton('Play')
        self.play_endlessly_button = QPushButton('Play Endlessly')
        self.playingAudio = False
        self.playingAudioInLoop = False

        pushButtonLayout = QGridLayout()
        title = QLabel("Audio output")
        pushButtonLayout.addWidget(title, 0, 0)
        pushButtonLayout.addWidget(self.play_button, 1, 0)
        pushButtonLayout.addWidget(self.play_endlessly_button, 1, 1)
        self.setLayout(pushButtonLayout)

        self.play_button.clicked.connect(self._playOnce)
        self.play_endlessly_button.clicked.connect(self._playEndlessly)

        self.parametersLayout.addLayout(
            pushButtonLayout, 5, 0, Qt.AlignmentFlag.AlignBottom)

    def createCanvas(self):
        """
        Visualization of the sample-based signal,
        the neuron membrane potential, the neuron internal currents
        and the threshold potential.
        Can show raster spike plot instead of membrane potential.
        """
        self.plotLayout = QGridLayout()

        # dynamically creating the plots for state variables despite V or spk plus raw input
        if self.neuron_model_name == "Mihalas-Niebur":
            num_figures = len(MN_neuron.NeuronState._fields)
        elif self.neuron_model_name == "Izhikevich":
            num_figures = len(IZ_neuron.NeuronState._fields)
        elif self.neuron_model_name == "Leaky integrate-and-fire":
            num_figures = len(LIF_neuron.NeuronState._fields)
        elif self.neuron_model_name == "Recurrent leaky integrate-and-fire":
            num_figures = len(RLIF_neuron.NeuronState._fields)
        else:
            ValueError("No neuron model selected.")
        num_figures += 1  # add raster plot

        # Create a figure and GridSpec
        self.figure = plt.figure(figsize=(10, 6))
        self.gs = GridSpec((num_figures // 2) + 1, 2, figure=self.figure)

        # Create lists to store the axes
        self.axes = {}
        self.axis_names = []
        for i in range(num_figures):
            self.axis_names.append(f"_dynamic_ax{i}")

        # Create the axes dynamically using GridSpec
        for i, axis_name in enumerate(self.axis_names):
            if i < len(self.axis_names) - 1:
                ax = self.figure.add_subplot(self.gs[i // 2, i % 2])
            else:
                ax = self.figure.add_subplot(self.gs[-1, :])
            self.axes[axis_name] = ax

        # Assign the axes to the corresponding variables dynamically
        for ax_variable_name, ax in self.axes.items():
            setattr(self, ax_variable_name, ax)

        # Add the figure to the layout
        self.canvas = FigureCanvas(self.figure)
        self.plotLayout.addWidget(self.canvas, 0, 0)
        self.canvasLayout.addLayout(self.plotLayout, 0, 0)

        # Adjust layout to reduce whitespace
        self.figure.tight_layout()
        # Alternatively, you can use subplots_adjust for more control:
        # self.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.4)

    def createChannelSelection(self):
        # create the channel selection
        self.channel_grid = QGridLayout()
        self.channel_box = []
        # TODO set first layout as default showing only text 'sensor selection'
        if self.dataFilename == None:
            # TODO create textbox
            print('Show only text.')
        else:
            position_list = []
            # creating the default layout as a grid (pref. height over width)
            sqrt = np.sqrt(len(self.channels))
            if sqrt.is_integer():
                for i in range(int(sqrt)):
                    for j in range(int(sqrt)):
                        position_list.append([i, j])
            else:
                for i in range(int(sqrt)+1):
                    for j in range(int(sqrt) + 1):
                        if len(position_list) == len(self.channels):
                            break
                        else:
                            position_list.append([i, j])
            for i in range(len(self.channels)):
                checkbox = QPushButton(str(i))
                checkbox.setCheckable(True)
                checkbox.setChecked(True)
                checkbox.setStyleSheet(
                    "background-color : lightgreen;"
                    "border-top-left-radius : 25px;"
                    "border-top-right-radius : 25px;"
                    "border-bottom-left-radius : 25px;"
                    "border-bottom-right-radius : 25px"
                )
                checkbox.clicked.connect(
                    lambda value, id=i: self._updateChannelCheckbox(value, id))
                self.channel_box.append(checkbox)
                checkbox.setFixedSize(50, 50)
                self.channel_grid.addWidget(
                    checkbox, position_list[i][0], position_list[i][1], alignment=Qt.AlignmentFlag.AlignCenter)

        self.parametersLayout.addLayout(self.channel_grid, 2, 0)

    def createDataSelection(self):
        """
        Used to select the class.
        """
        dataSelectionLayout = QGridLayout()
        title = QLabel("Data management")
        self.loadButton = QPushButton("Load data")

        self.loadButton.clicked.connect(self.openData)
        dataSelectionLayout.addWidget(title, 0, 0, 1, 0)
        dataSelectionLayout.addWidget(self.loadButton, 1, 0, 1, 0)

        # TODO only needed when multiple trials with same label given (read from data)
        # TODO if no label given and multiple trials create dial
        # create a dial to select the repetition
        self.selectedRepetition = 0
        self.dialRepetition = QDial(self)
        self.dialRepetition.setMinimum(0)
        self.dialRepetition.setMaximum(0)
        self.dialRepetition.setValue(self.selectedRepetition)
        self.dialRepetition.sliderReleased.connect(self._updateDialRepetition)
        dataSelectionLayout.addWidget(self.dialRepetition, 2, 0)

        # TODO only needed if differnet labels given (read from data)
        # create a combo box with all letters
        self.comboBoxLetters = QComboBox()
        self.comboBoxLetters.currentTextChanged.connect(
            self._updateComboBoxLettersText)
        dataSelectionLayout.addWidget(self.comboBoxLetters, 2, 1)

        self.parametersLayout.addLayout(
            dataSelectionLayout, 0, 0, Qt.AlignmentFlag.AlignTop
        )
        self.createChannelSelection()

    def createModelSelection(self):
        """
        Select the neuron model to use.
        """
        modelSelectionLayout = QGridLayout()
        title = QLabel("Neuron model and parameters")
        self.combo_box_neuron_model = QComboBox(self)
        neuron_neuron_model_names = [
            "Mihalas-Niebur",
            "Izhikevich",
            "Leaky integrate-and-fire",
            "Recurrent leaky integrate-and-fire",
        ]
        self.combo_box_neuron_model.addItems(neuron_neuron_model_names)
        self.combo_box_neuron_model.currentTextChanged.connect(
            self.changeModel)
        self.loadParameter()
        modelSelectionLayout.addWidget(title, 0, 0)
        modelSelectionLayout.addWidget(self.combo_box_neuron_model, 1, 0)
        self.parametersLayout.addLayout(
            modelSelectionLayout, 3, 0, Qt.AlignmentFlag.AlignBottom)

    def createParamSlider(self):
        """
        Used to select the neuron parameter to change.
        """
        self.sliderLayout = QGridLayout()
        self.sliderParamLabel = {}
        self.sliders = []
        self.sliderValues = [0] * len(self.parameter)
        self.factor = np.ones(len(self.parameter))
        self.steps_size = [1] * len(self.parameter)

        # slider can only handle int values, for float we need to scale the step size
        for id, (param_key, param_values) in enumerate(self.parameter.items()):
            if param_values[2] < 1:
                # find the factor to scale step size to int
                decimals = Decimal(str(param_values[2]))
                self.factor[id] = 10 ** (abs(decimals.as_tuple().exponent))
            self.steps_size[id] = int(param_values[2] * self.factor[id])
            self.sliderValues[id] = int(
                param_values[-1] * self.factor[id])  # read start value

            # create a slider for every param
            slider = QSlider(Qt.Orientation.Horizontal, self)
            slider.setMinimum(int(param_values[0] * self.factor[id]))
            slider.setMaximum(int(param_values[1] * self.factor[id]))
            slider.setValue(self.sliderValues[id])  # set start value
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(
                int(abs(np.diff(param_values[:2])[0]) * self.factor[id] / 20)
            )  # display n steps
            slider.setSingleStep(self.steps_size[id])  # read step value

            # Connect the sliderReleased signal to the event handler
            slider.sliderReleased.connect(
                lambda id=id, slider=slider: self._updateParamSlider(
                    slider.value(), id)
            )

            self.sliders.append(slider)
            self.sliderLayout.addWidget(slider, id + 2, 1)

            # create label for each slider
            # # TODO here we assume that the parameter name is always 'x_n' or does not contain '_' at all
            # # TODO this is not ideal, but in most cases variables have only a single digit subscript
            # # Use regular expression to find the pattern 'x_n' and format it as 'x<sub>n</sub>'
            # import re
            # formatted_string = re.sub(r'(\w)_(\w)', r'\1<sub>\2</sub>', param_key)
            # # Creating a QLabel
            # sliderLabel = QLabel()
            # # Setting HTML content with the formatted string
            # sliderLabel.setText(f'<html><head/><body><p>{formatted_string}</p></body></html>')

            sliderLabel = QLabel(param_key, self)  # write parameter name
            sliderLabel.setAlignment(Qt.AlignmentFlag.AlignLeft)
            self.sliderLayout.addWidget(sliderLabel, id + 2, 0)
            # link the value readout to each slider
            self.sliderParamLabel[id] = QLabel(
                str(np.mean(param_values[-1])), self
            )  # set start value of label
            self.sliderParamLabel[id].setAlignment(Qt.AlignmentFlag.AlignRight)
            self.sliderParamLabel[id].setMinimumWidth(80)
            self.sliderLayout.addWidget(self.sliderParamLabel[id], id + 2, 2)

        self.parametersLayout.addLayout(
            self.sliderLayout, 4, 0)

    def createPreprocessingSelection(self):
        """
        Creates the preprocessing section.
        """
        # init layout
        self.preprocessingLayout = QGridLayout()
        title = QLabel("Preprocessing")
        self.preprocessingLayout.addWidget(title, 0, 0)

        # TODO create a 2x2 grid for checkboxes
        # checkboxes for: normalize, filter, startTrialAtNull, split_data
        self.normalizeDataCheckbox = QCheckBox("Normalize data")
        self.normalizeDataCheckbox.setChecked(self.normalizeData)
        self.normalizeDataCheckbox.stateChanged.connect(
            self._updateNormalizeData)
        self.preprocessingLayout.addWidget(
            self.normalizeDataCheckbox, 1, 0, Qt.AlignmentFlag.AlignLeft)

        self.filterSignalCheckbox = QCheckBox("Filter signal")
        self.filterSignalCheckbox.setChecked(self.filterSignal)
        self.filterSignalCheckbox.stateChanged.connect(
            self._updateFilterSignal)
        self.preprocessingLayout.addWidget(
            self.filterSignalCheckbox, 1, 1, Qt.AlignmentFlag.AlignLeft)

        self.startTrialAtNullCheckbox = QCheckBox("Start trial at null")
        self.startTrialAtNullCheckbox.setChecked(self.startTrialAtNull)
        self.startTrialAtNullCheckbox.stateChanged.connect(
            self._updateStartTrialAtNull)
        self.preprocessingLayout.addWidget(
            self.startTrialAtNullCheckbox, 2, 0, Qt.AlignmentFlag.AlignCenter)

        self.splitDataCheckbox = QCheckBox("Split data")
        self.splitDataCheckbox.setChecked(self.enable_data_splitting)
        self.splitDataCheckbox.stateChanged.connect(
            self._updateSplitData)
        self.preprocessingLayout.addWidget(
            self.splitDataCheckbox, 2, 1, Qt.AlignmentFlag.AlignLeft)

        # slider to set the upsampling factor and scale
        # TODO inlcude downsampling (1/4, 1/2)
        dt_label_text = QLabel("upsample", self)
        dt_label_text.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.preprocessingLayout.addWidget(dt_label_text, 3, 0)
        self.dt_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.dt_slider.setMinimum(1)
        self.dt_slider.setMaximum(10)
        self.dt_slider.setValue(1)
        self.dt_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.dt_slider.setTickInterval(1)
        self.dt_slider.setSingleStep(1)
        self.dt_slider.sliderReleased.connect(self._updateDt)
        self.preprocessingLayout.addWidget(self.dt_slider, 3, 1)
        self.dt_label = QLabel(str(1), self)  # define start value
        self.dt_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.dt_label.setMinimumWidth(80)
        self.preprocessingLayout.addWidget(self.dt_label, 3, 2)

        scale_label_text = QLabel("scale", self)
        scale_label_text.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.preprocessingLayout.addWidget(scale_label_text, 4, 0)
        self.scale_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(10)
        self.scale_slider.setValue(1)
        self.scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.scale_slider.setTickInterval(1)
        self.scale_slider.setSingleStep(1)
        self.scale_slider.sliderReleased.connect(self._updateScale)
        self.preprocessingLayout.addWidget(self.scale_slider, 4, 1)
        self.scale_label = QLabel(str(1), self)  # define start value
        self.scale_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.scale_label.setMinimumWidth(80)
        self.preprocessingLayout.addWidget(self.scale_label, 4, 2)

        # add to overall layout
        self.parametersLayout.addLayout(
            self.preprocessingLayout, 1, 0, Qt.AlignmentFlag.AlignTop
        )

    def createSpikePatternVisualizer(self):
        """Create a table for spike pattern visualization in the second tab."""
        self.spikePatternTable = QTableWidget()
        # Example rows, adjust as needed
        self.spikePatternTable.setRowCount(1)
        # Two columns: ID and Spike Pattern
        self.spikePatternTable.setColumnCount(22)
        self.spikePatternTable.setHorizontalHeaderLabels(
            ["ID",
             "Spike Pattern",
             "Tonic spiking",
             "Class 1",
             "Spike frequency\nadaptation",
             "Phasic spiking",
             "Accommodation",
             "Threshold\nvariability",
             "Rebound spike",
             "Class 2",
             "Integrator",
             "Input\nbistability",
             "Hyperpolarizing\nspiking",
             "Hyperpolarizing\nbursting",
             "Tonic bursting",
             "Phasic bursting",
             "Rebound burst",
             "Mixed mode",
             "Afterpotentials",
             "Basal\nbistability",
             "Preferred\nfrequency",
             "Spike latency"
             ])

        layout = QGridLayout()
        layout.addWidget(self.spikePatternTable, 0, 0)
        self.tab2.setLayout(layout)

    @QtCore.pyqtSlot()
    def drawCanvas(self):
        """
        Update the plots.
        """
        channels = self.channels
        upsample_fac = self.upsample_fac
        input = self.input_data
        output = self.output_data
        axes = self.axes
        axis_names = self.axis_names
        dt = self.dt

        # some color schemes can be found here
        # (https://matplotlib.org/stable/tutorials/colors/colormaps.html)

        colors = cm.hsv(
            np.linspace(0, 1, (len(channels) * 2))
        )  # len(channels)*2 to increase visibility

        # TODO check if this is wanted (fast plotting vs accuracy)
        idx = np.arange(0, input.shape[0], upsample_fac)
        time = idx * dt / upsample_fac

        # plot the input data
        axes[axis_names[0]].clear()
        axes[axis_names[0]].set_prop_cycle("color", colors[::2][channels])
        axes[axis_names[0]].plot(time, input[idx, 0][:, channels])
        ymin = np.min(input[:, 0, channels]) - 0.1 * abs(np.min(input[:, 0, channels]))
        ymax = np.max(input[:, 0, channels]) + 0.1 * abs(np.max(input[:, 0, channels]))
        if ymin != ymax:
            axes[axis_names[0]].set_ylim(ymin, ymax)
        axes[axis_names[0]].set_title("Input")
        self.canvas.draw()

        axes[axis_names[1]].clear()

        # dynamically creating the plots for state variables
        if self.neuron_model_name == "Mihalas-Niebur":
            self.neuronStateVariables = MN_neuron.NeuronState._fields
        elif self.neuron_model_name == "Izhikevich":
            self.neuronStateVariables = IZ_neuron.NeuronState._fields
        elif self.neuron_model_name == "Leaky integrate-and-fire":
            self.neuronStateVariables = LIF_neuron.NeuronState._fields
        elif self.neuron_model_name == "Recurrent leaky integrate-and-fire":
            self.neuronStateVariables = RLIF_neuron.NeuronState._fields
        else:
            ValueError("No neuron model selected.")

        variable_info = []
        for i, name in enumerate(self.neuronStateVariables):
            variable_info.append({"index": str(i + 1), "title": name})

        variables = []
        for i, single_variable_info in enumerate(variable_info):
            var_info = single_variable_info
            index = var_info["index"]
            title = var_info["title"]

            var_name = f"_dynamic_ax{index}"
            var = getattr(self, var_name, None)

            if var is not None:
                variables.append({"ax": var, "index": index, "title": title})

        for i, var in enumerate(variables):
            ax = var["ax"]
            index = int(var["index"])
            title = var["title"]

            ax.clear()
            if i < len(variables) - 1:
                ax.set_prop_cycle("color", colors[::2][channels])
                ax.plot(time, output[idx, index, 0, :][:, channels])

                ymin = np.min(output[idx, index, 0, :][:, channels]) - 0.1 * abs(np.min(output[idx, index, 0, :][:, channels]))
                ymax = np.max(output[idx, index, 0, :][:, channels]) + 0.1 * abs(np.max(output[idx, index, 0, :][:, channels]))
                if ymin != ymax:
                    ax.set_ylim(ymin, ymax)
                ax.set_title(title)
            else:
                # create raster plot
                t, neuron_idx = np.where(output[:, 0, 0, :])
                i = np.where(np.in1d(neuron_idx, np.where(channels)[0]))
                t = t[i] * (dt / upsample_fac)
                neuron_idx = neuron_idx[i]
                # remove color argument to get monochrom
                ax.scatter(x=t, y=neuron_idx, c=colors[::2][neuron_idx], s=5)
                ax.set_ylim(-1, output.shape[-1])
                ax.set_xlim(0, output.shape[0] / (1 / self.dt) / upsample_fac)

        self.canvas.draw()

    def _updateCanvas(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.draw_event.emit()

    def _updateChannelCheckbox(self, value, id):
        checkbox = self.sender()
        if value:
            checkbox.setStyleSheet(
                "background-color : lightgreen;"
                "border-top-left-radius : 25px;"
                "border-top-right-radius : 25px;"
                "border-bottom-left-radius : 25px;"
                "border-bottom-right-radius : 25px"
            )
            self.channels[id] = value
            self.simulate_event.emit()
        else:
            if self.channels.sum() == 1:
                checkbox.setChecked(True)
            else:
                checkbox.setStyleSheet(
                    "background-color : red;"
                    "border-top-left-radius : 25px;"
                    "border-top-right-radius : 25px;"
                    "border-bottom-left-radius : 25px;"
                    "border-bottom-right-radius : 25px"
                )
                self.channels[id] = value
                self.simulate_event.emit()

    def _updateComboBoxLettersText(self, s):
        self.active_class = self.le.transform([s])[0]
        self.simulate_event.emit()

    def _resetLayout(self, layout, sublayout):
        """Remove all the sliders from the interface."""
        def deleteItems(layout):
            if layout is not None:
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()
                    else:
                        deleteItems(item.layout())
        # Delete the layout that contains the sliders
        deleteItems(sublayout)
        if layout != None:
            layout.removeItem(sublayout)

    @QtCore.pyqtSlot()
    def writeTable(self):
        """Update the spike pattern visualizer."""
        # TODO label table and raster plot do not match!!!
        # Clear the table
        self.spikePatternTable.setRowCount(self.output_data.shape[-1])
        # Add new rows
        for i in range(self.output_data.shape[-1]):
            self.spikePatternTable.setItem(
                i, 0, QTableWidgetItem(str(i)))  # ID
            self.spikePatternTable.setItem(i, 1, QTableWidgetItem(
                self.finalPredictionList[i]))  # predicted spike pattern
            for pattern_label_counter in range(20):
                if self.finalPredictionList[i] == 'No spikes':
                    item = QTableWidgetItem("0 %")
                    # color the cell white
                    item.setBackground(QColor(255, 255, 255))
                    self.spikePatternTable.setItem(
                        i, pattern_label_counter + 2, item)
                else:
                    probability = self.normalized_softmax[i, pattern_label_counter]
                    percentage = int((probability * 100) + 0.5)
                    item = QTableWidgetItem(str(percentage) + " %")

                    # Calculate color based on probability
                    red = int(probability * 255)
                    blue = int((1 - probability) * 255)
                    green = 5
                    color = QColor(red, green, blue)

                    # Adjust color lightness based on distance from 0.5
                    distance_from_mid = abs(probability - 0.5)
                    lightness_factor = EXTREME_LIGHTNESS + \
                        int((1 - distance_from_mid * 2) *
                            (MIDPOINT_LIGHTNESS - EXTREME_LIGHTNESS))
                    adjusted_color = color.lighter(lightness_factor)

                    item.setBackground(adjusted_color)
                    # probability of each pattern
                    self.spikePatternTable.setItem(
                        i, pattern_label_counter + 2, item)

    # END DATA VISUALIZATION

    #################
    # MODEL HANDLER #
    #################

    def changeModel(self, neuron_model_name):
        # Create widgets for changing neuron model
        self.neuron_model_name = neuron_model_name
        self.loadParameter()  # load parameters from file
        # set parameters to initial values
        self._resetLayout(self.parametersLayout, self.sliderLayout)
        self.createParamSlider()  # create widgets for parameters
        # set parameters to initial values
        self._resetLayout(None, self.canvasLayout)
        self.createCanvas()  # create widgets for parameters
        # Emit a signal to update the GUI
        self.simulate_event.emit()
        logging.info(f"Neuron model changed to {neuron_model_name} neuron.")

    def loadParameter(self):
        # Load the parameters for the selected neuron model
        if self.neuron_model_name == "Mihalas-Niebur":
            from utils.neuron_parameters import mn_parameter
            self.parameter = mn_parameter
        elif self.neuron_model_name == "Izhikevich":
            from utils.neuron_parameters import iz_parameter
            self.parameter = iz_parameter
        elif self.neuron_model_name == "Leaky integrate-and-fire":
            from utils.neuron_parameters import lif_parameter
            self.parameter = lif_parameter
        elif self.neuron_model_name == "Recurrent leaky integrate-and-fire":
            from utils.neuron_parameters import rlif_parameter
            self.parameter = rlif_parameter
        else:
            raise ValueError("Select a valid neuron model.")

    # END MODEL HANDLER

    ##################
    # DATAMANAGEMENT #
    ##################

    def openData(self):
        """Load data and set up the data management interface."""
        self.dataFilename = QFileDialog.getOpenFileName(
            self, "Open file", "./", "Pickle file (*.pkl)")[0]
        self.data_dict = pd.read_pickle(self.dataFilename)
        # check windows vs unix  # TODO check if this is actually needed
        if len(self.dataFilename.split('/')) > len(self.dataFilename.split('\\')):
            newFilename = self.dataFilename.split('/')
        else:
            newFilename = self.dataFilename.split('\\')
        self.loadButton.setText(newFilename[-1])

        self._loadData()
        self.createChannelSelection()

        # Set the class selection
        self.comboBoxLetters.clear()
        if 'letter' in list(self.data_dict.keys()):
            logging.info('Found special naming. Gonna use it')
            logging.info('Setting up box for class selection.')
            self.comboBoxLetters.addItems(
                list(np.unique(self.data_dict["letter"])))
            #  set first class as default
            self.active_class = self.le.transform(
                [list(np.unique(self.data_dict["letter"]))[0]])[0]
        else:
            if 'class' in list(self.data_dict.keys()):
                logging.info('Found standart naming. Gonna use it.')
                logging.info('Setting up box for class selection.')
                self.comboBoxLetters.addItems(
                    list(np.unique(self.data_dict["class"])))
                #  set first class as default
                self.active_class = self.le.transform(
                    [list(np.unique(self.data_dict["class"]))[0]])[0]
            else:
                logging.warning('No classes found. (Remove box?)')

        # only create wheel if multiple repetitions are given
        if 'repetition' in list(self.data_dict.keys()) and len(np.unique(self.data_dict["repetition"])) > 1:
            logging.info('Setting up dial to select the repetition.')
            # Modify the repetition selection
            self.selectedRepetition = int(
                random() * len(np.unique(self.data_dict["repetition"]))
            )
            self.dialRepetition.setMaximum(
                len(np.unique(self.data_dict["repetition"])) - 1)
        else:
            # remove dial
            logging.warning('Only single trial per class. (Removing dial?)')
            # Remove the dial widget from the layout
            self.dialRepetition.setParent(None)
            self.dialRepetition.deleteLater()

    def _loadData(self):
        """Load the data from the file."""
        self.data_split, self.labels, self.timestamps, self.le, self.data = load_data(
            self.dataFilename,
            upsample_fac=self.upsample_fac,
            normalize_data=self.normalizeData,
            scale_val=self.scale,
            filter_data=self.filterSignal,
            startTrialAtNull=self.startTrialAtNull,
        )
        if 'example_braille_data' in self.dataFilename:
            self.dt = 1E-2  # 1/40  # 25Hz in sec
        self.channels = np.ones(self.data.shape[-1], dtype=bool)
        self.data_default = self.data.numpy()
        self.timestamps_default = self.timestamps.copy()

        self.simulate_event.emit()

    def _updateDialRepetition(self):
        """Update the repetition according to the dial."""
        self.dialRepetition.sliderReleased.connect(self._updateDialRepetition)
        self.simulate_event.emit()

    def _updateDt(self):
        """Recalculate the input data and neuron output with new dt."""
        # TODO this approach is not very memory friendly especially for big datasets.
        # Think about working with single class/trial only?
        value = self.sender().value()
        self.dt_label.setText(str(value))
        self.upsample_fac = value
        data_steps = len(self.data_default[0])
        self.data_steps = data_steps * self.upsample_fac
        # here we change the number of computed time steps according to the upsample factor
        self._updateData()
        self.simulate_event.emit()

    def _updateFilterSignal(self):
        """Update data according to filter signal checkbox."""
        self.filterSignal = self.sender().isChecked()
        self._updateData()
        self.simulate_event.emit()

    def _updateNormalizeData(self):
        """Update data according to normalize data checkbox."""
        self.normalizeData = self.sender().isChecked()
        self._updateData()
        self.simulate_event.emit()

    def _updateParamSlider(self, value, id):
        """Update the parameter according to the slider."""
        value = self.sliders[id].value(
        ) // self.steps_size[id] * self.steps_size[id]
        self.sliders[id].setValue(value)

        # here we convert int back to float
        if self.sliderValues[id] != value:
            self.sliderValues[id] = value
            self.sliderParamLabel[id].setText(
                str(value / int(self.factor[id])))
        self.simulate_event.emit()

    def _updateScale(self):
        """Update the data scaling."""
        value = self.sender().value()
        self.scale_label.setText(str(value))
        self.scale = value
        self._updateData()
        self.simulate_event.emit()

    def _updateSplitData(self):
        """
        Used to enforce or disable splitting of data into positive and negative values.
        If no negative values given, channel will contain zeros only.
        """
        self.enable_data_splitting = self.sender().isChecked()
        self._updateData()
        if self.enable_data_splitting:
            self.channels = np.ones(self.data_split.shape[-1], dtype=bool)
        else:
            self.channels = np.ones(self.data.shape[-1], dtype=bool)
        self._resetLayout(self.parametersLayout, self.channel_grid)
        self.createChannelSelection()
        self.simulate_event.emit()

    def _updateData(self):
        timestamps, data = preprocess_data(
            data=self.data_default,
            timestamps=self.timestamps_default,
            upsample_fac=self.upsample_fac,
            normalize_data=self.normalizeData,
            scale_val=self.scale,
            filter_data=self.filterSignal,
            startTrialAtNull=self.startTrialAtNull
        )
        self.data = torch.tensor(data)
        if self.enable_data_splitting:
            self.data_split = torch.tensor(split_data(data))
        self.timestamps = timestamps

    def _updateStartTrialAtNull(self):
        """
        Checkbox to set each sensor to start at zero.
        Will redraw the plots.
        """
        self.startTrialAtNull = self.sender().isChecked()
        self._updateData()
        self.simulate_event.emit()

    # END DATAMANAGEMENT

    ###################
    # SPIKES TO AUDIO #
    ###################

    @QtCore.pyqtSlot()
    def spikeToAudio(self, out_path: str, neuron_spike_times: np.array, audio_duration: float):
        """
        Create the audio file from the spike times.
        """
        neuron_spike_times = neuron_spike_times * 1000  # ms
        audio_duration = audio_duration * 1000  # ms

        # Set the audio properties
        tick_duration = 10.0  # ms
        frequency = 150  # Hz

        tick = Sawtooth(freq=frequency).to_audio_segment(
            duration=tick_duration)

        # Create a silent audio segment to hold the combined ticks
        audio = AudioSegment.silent(duration=audio_duration)

        # Place ticks for each neuron's spike times
        for spike_time in neuron_spike_times:
            audio = audio.overlay(tick, position=int(spike_time))

        # Export the audio to a WAV file
        output_file = f"{out_path}/spikeToAudio.wav"
        audio.export(output_file, format="wav")

        return audio

    def _playEndlessly(self):
        if self.playingAudioInLoop:
            self.player.setLoops(QMediaPlayer.Loops.Once)
            self.playingAudioInLoop = False
            self.play_endlessly_button.setText('Play Endlessly')
        else:
            self.player.setLoops(QMediaPlayer.Loops.Infinite)
            self.playingAudioInLoop = True
            self.play_endlessly_button.setText('Play Once')

    def _playOnce(self):
        # self.player.playbackState().name: StoppedState, PlayingState, PausedState
        # TODO include resetting of button when reached end of audio
        if self.playingAudio:
            # audio is playing
            # self.player.stop()
            self.player.pause()
            self.playingAudio = False
            self.play_button.setText('Play')
        else:
            self.player.play()
            self.playingAudio = True
            self.play_button.setText('Pause')

    def _updateSpikesToAudio(self):
        '''
        Triggers the calculation of the spike to audio conversion.
        '''
        neuronSpikeTimesDense = np.reshape(self.output_data[:, 0, :, :], (
            self.output_data.shape[0], self.output_data.shape[-1]))  # TODO call spikes by key?
        # convert sparse representation to spike times
        neuronSpikeTimes = np.where(neuronSpikeTimesDense == 1)[0] * self.dt
        audio_duration = len(self.input_data) * self.dt

        audio = self.spikeToAudio(
            out_path='./tmp', neuron_spike_times=neuronSpikeTimes, audio_duration=audio_duration)

    # END SPIKE TO AUDIO

    ############################
    # SPIKE PATTERN CLASSIFIER #
    ############################
    def spikePatternClassification(self):
        generator = prepareDataset(self.output_data)
        predictions, softmax = classifySpikes(generator)
        # let us get the most frequent predicted class over all batches
        self.finalPredictionList = []
        for sensorId in range(self.output_data.shape[-1]):
            # nothing to do if no spikes given
            uniquePredictions, count = np.unique(
                predictions[sensorId, :], return_counts=True)
            
            # Sort predictions by count in descending order
            sorted_indices = np.argsort(count)[::-1]
            sorted_predictions = uniquePredictions[sorted_indices]
            sorted_counts = count[sorted_indices]
            
            # Check if 'No spikes' has the highest count
            if len(sorted_predictions) > 1 and sorted_predictions[0] == 'No spikes':
                # Select the second highest count
                self.finalPredictionList.append(sorted_predictions[1])
            else:
                # Select the highest count
                self.finalPredictionList.append(sorted_predictions[0])
                
        mean_softmax = np.mean(softmax, axis=1)
        # Calculate the sum along axis 1, keeping the dimensions
        sum_mean_softmax = np.sum(mean_softmax, axis=1, keepdims=True)

        # Normalize mean_softmax, avoiding division by zero
        self.normalized_softmax = np.divide(
            mean_softmax, 
            sum_mean_softmax, 
            where=sum_mean_softmax != 0
        )

    def _updateSpikePatternClassification(self):
        self.spikePatternClassification()
        self.write_event.emit()


def main():
    """EncodingGUI's main function."""
    WiN_GUI = QApplication([])
    winGUIwindow = WiN_GUI_Window()
    winGUIwindow.show()

    sys.exit(WiN_GUI.exec())  # TODO inlcude removing tmp folder


if __name__ == "__main__":
    main()
