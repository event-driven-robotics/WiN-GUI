# WiN-GUI
 
Have you ever wondered how your data would look like in *neuromorphic*? Have you ever wondered what is actually going on *in* a neuron and how does a bunch of them *sound* like? Just run the WiN_GUI.py file and you will find out!

Here we present the **WiN-GUI** (Watch inside Neurons-GUI) to interactively change neuron models and parameters. It allows to load any sample-based time-series data (for more details see [Data structure](#data-structure)) and converts it into membrane voltage traces and spike-trains. Playing with the parameter sliders helps to understand their interaction, the neuron mechanics and how your encoded data changes accordingly. The neuron model can be changed with a single click and the GUI will update the visualization. If you have your own neuron model and want to tune tha parameters check out the How-to: [include custom neuron models](#include-custom-neuron-models) chapter. 

Available neuron models are:
- [Mihalas-Niebur neuron](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/utils/neuron_models.py#L50).
- [Izhikevich neuron](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/utils/neuron_models.py#L149).
- [Leaky-integrate and firing neuron](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/utils/neuron_models.py#L222).
- [Recurrent leaky-integrate and firing neuron](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/utils/neuron_models.py#L274).

![WiN_gui](https://github.com/2103simon/encoding_gui/blob/main/assets/WiN_gui.png)

## Installation
The WiN-GUI is based on [PyQT6](https://www.riverbankcomputing.com/software/pyqt/) and [PyTorch](https://pytorch.org/). Some other packages are requiered, too. Use the [requirements](https://github.com/2103simon/encoding_gui/blob/main/requirements.txt) file provided to set everything up Further libraries we face missing often are listed in [Packages you might need to install](#packages-you-might-need-to-install).


## Data structure
The WiN-GUI is intentionally designed to visualize data from robotic experiments different classes and multiple recordings per class. Nonetheless does the GUI support single class and single recording. In that case you only have to make sure, that the data is structured like: `nb_time_steps x nb_sensors`. If you provide the data as python dictionary with an entry 'label' the dataselection over the drop down menu is availabale, otherwise over the dial. If you want to visualize multi-class and multi recordings you have to provide each record with the according label (e.g. 'label': class_name). 

Below you can find some example data structures:    
1. single trial: `nb_time_steps x nb_sensors`
2. single class, multiple repetitions: `nb_trial x nb_time_steps x nb_sensors` (do not forget to provide the label 'repetition' as dictionary key)
3. multi-class, no repetition: `nb_classes x nb_time_steps x nb_sensors` (do not forget to provide the label 'class' as dictionary key)
4. multi-class, multiple repetitions: `(nb_classes x nb_repetitions) x nb_time_steps x nb_sensors`

Please have a look at the example data provided for further details.

## Parameter settings
The parameter to be manipulated in the WiN-GUI are defined in the [neuron model](https://github.com/2103simon/encoding_gui/blob/main/utils/neuron_models.py) and seperatly in a [python file](https://github.com/2103simon/encoding_gui/blob/main/utils/neuron_parameters.py). The parameter listed in the python file must not contain all parameters defined in the neuron model, the default will be used for the missing one. The python file allows setting the min and max value for each parameter, the precision (step) for the slider, and the intial value.

## Preprocessing
We use the Multidimensional image processing (scipy.ndimage) for filtering, and Signal processing (:mod:`scipy.signal`) for resampling.

## Channel selection

## Neuron model and parameter

# How to
## Change the filter
If you want to change the filter properties you can do this [here](https://github.com/2103simon/encoding_gui/blob/d07c60c680ace8ccb1121eeaa21acb9480533ef1/utils/data_management.py#L34). To change the filter replace the default filter [here](https://github.com/2103simon/encoding_gui/blob/d07c60c680ace8ccb1121eeaa21acb9480533ef1/utils/data_management.py#L57).

## Include custom neuron models
The WiN-GUI can be extended with further neuron models! To use your custom neuron model, the model has to be added [here](https://github.com/2103simon/encoding_gui/blob/main/utils/models.py) and the according parameter [here](https://github.com/2103simon/encoding_gui/blob/main/parameters/encoding_parameter.py). Finally, you need to load your model to be used in the GUI. Brief overview showing a custom neuron model in PyTroch you can find [here](https://pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html).

Here comes a step by step how to guide based on the LIF neuron:

1. Add your neuron model:
   1. Add a new class as `NAME(nn.Module)`, [ref.](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/utils/neuron_models.py#L222).
   2. List the `NeuronState` variables, [ref.]([TODO ref](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/utils/neuron_models.py#L223)).
   3. Define the model (`init`, `forward`, `reset`) and pay attention to set the default values, [ref.]([TODO ref](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/utils/neuron_models.py#L225)).
2. Add the neuron parameter:
   1. Add a new dictionary in the [neuron_parameters file](https://github.com/2103simon/encoding_gui/blob/main/utils/neuron_parameters.py) as `NAME={}`, [ref](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/utils/neuron_parameters.py#L34).
   2. Each entry has to match a single member in the `NeuronState` list (see 1.2), but not all `NeuronState` members have to be listed here. For not listed the default values from the `init` (see 1.3) are used.
3. Load the neuron model and parameter:
    1. Include the model [here](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/WiN_GUI.py#L37), [here](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/WiN_GUI.py#L64), [here](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/WiN_GUI.py#L262), [here](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/WiN_GUI.py#L609), and [here](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/WiN_GUI.py#L742).
    2. The default neuron model can be set [here](https://github.com/2103simon/encoding_gui/blob/398aa68263e1a07fee5272eccd69fc206003d92b/WiN_GUI.py#L177).
  
If you implement a new model, and want to make it available for the community, let us know!

## Write your own template for the sensor visualization
For those who want to give there GUI the perfect personal glance can change the sensor visualization to reflect the physical setup. The default will be a grid, shown in the example is the iCub fingertip map.
TODO here will be the description of default settings and possibility to include a personal visualization.


### Packages you might need to install
apt-get packages:
apt-get install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev

apt packages:
apt install ffmpeg, portaudio19-dev

