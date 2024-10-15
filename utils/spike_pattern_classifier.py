"""
spike_pattern_classifier.py

This module implements functions and utilities for classifying spike patterns using spiking neural networks (SNNs). 
It includes functions for checking GPU availability, preparing datasets, loading pre-trained weights, and running 
simulations of spiking neural networks. The module also defines custom autograd functions for surrogate gradient 
computation, which are used to train the SNN.

Main Components:
- checkCuda: A function to check for available GPUs and set up the device for computation.
- classifySpikes: A function to classify spike patterns from a generator of spike data.
- prepareDataset: A function to prepare a dataset for spike pattern classification.
- getFiringPatternLabels: A function to retrieve a mapping of firing pattern labels.
- loadWeights: A function to load pre-trained weights for the spiking neural network.
- computeActivity: A function to compute the activity of a feedforward spiking neural network layer.
- computeRecurrentActivity: A function to compute the activity of a recurrent spiking neural network layer.
- runSNN: A function to run a spiking neural network simulation.

Dependencies:
- numpy
- torch
- torch.nn
- torch.utils.data
- tqdm

Usage:
This module is intended to be used as part of the WiN-GUI project for spike pattern classification. 
To use this module, ensure that the required dependencies are installed and the dataset is structured 
as specified in the README.md.

Example:
    import torch
    from spike_pattern_classifier import classifySpikes, prepareDataset, loadWeights, runSNN

    # Prepare dataset
    dataset = prepareDataset(data_path)

    # Load pre-trained weights
    model = loadWeights(model_path)

    # Run SNN simulation
    results = runSNN(model, dataset)

"""

import numpy as np
import torch
import torch.nn as nn
from utils.surrogate_gradient import activation
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def checkCuda(share_GPU=False, gpu_sel=0, gpu_mem_frac=None):
    """
    Check for available GPU and distribute work (if needed/wanted).

    This function checks for available GPUs and sets up the device for computation. 
    It can distribute the load across multiple GPUs if specified, or use a single GPU or CPU.

    Args:
        share_GPU (bool, optional): If True, the load will be shared across multiple GPUs. Defaults to False.
        gpu_sel (int, optional): The index of the GPU to use if not sharing the load. Defaults to 0.
        gpu_mem_frac (float, optional): The fraction of GPU memory to allocate for the process. 
                                        If None, the default allocation is used. Defaults to None.

    Returns:
        torch.device: The device to be used for computation (either a specific GPU or the CPU).
    """

    if (torch.cuda.device_count() > 1) & (share_GPU):
        gpu_av = [torch.cuda.is_available()
                  for ii in range(torch.cuda.device_count())]
        # print("Detected {} GPUs. The load will be shared.".format(
        #     torch.cuda.device_count()))
        for gpu in range(len(gpu_av)):
            if True in gpu_av:
                if gpu_av[gpu_sel]:
                    device = torch.device("cuda:"+str(gpu))
                    # print("Selected GPUs: {}" .format("cuda:"+str(gpu)))
                else:
                    device = torch.device("cuda:"+str(gpu_av.index(True)))
            else:
                device = torch.device("cpu")
                # print("No available GPU detected. Running on CPU.")
    elif (torch.cuda.device_count() > 1) & (not share_GPU):
        # print("Multiple GPUs detected but single GPU selected. Setting up the simulation on {}".format(
        #     "cuda:"+str(gpu_sel)))
        device = torch.device("cuda:"+str(gpu_sel))
        if gpu_mem_frac is not None:
            # decrese or comment out memory fraction if more is available (the smaller the better)
            torch.cuda.set_per_process_memory_fraction(
                gpu_mem_frac, device=device)
    else:
        if torch.cuda.is_available():
            # print("Single GPU detected. Setting up the simulation there.")
            device = torch.device("cuda:"+str(torch.cuda.current_device()))
            # thr 1: None, thr 2: 0.8, thr 5: 0.5, thr 10: None
            if gpu_mem_frac is not None:
                # decrese or comment out memory fraction if more is available (the smaller the better)
                torch.cuda.set_per_process_memory_fraction(
                    gpu_mem_frac, device=device)
        else:
            # print("No GPU detected. Running on CPU.")
            device = torch.device("cpu")

    return device


def prepareDataset(data):
    """
    Prepares a dataset for spike pattern classification.

    This function processes the input spike data, ensuring that each sensor has exactly 1000 time steps. 
    If the data has fewer than 1000 time steps, it repeats the data. If the data has more than 1000 time steps, 
    it uses a sliding window approach to create multiple samples. The function returns a DataLoader for the processed dataset.

    Args:
        data (np.ndarray): A 4D NumPy array of shape [timesteps, internal variables, 1, sensors] containing the spike data.

    Returns:
        DataLoader: A DataLoader object for the processed dataset, with each batch containing the spike data and corresponding sensor indices.
    """

    neuronSpikeTimesDense = np.reshape(
        data[:, 0, :, :], (data.shape[0], data.shape[-1]))
    neuronSpikeTimesDense = torch.as_tensor(
        neuronSpikeTimesDense, dtype=torch.float32)
    target_nb_samples = 1000
    stride = 100

    # if neuronSpikeTimesDense.shape[0] < target_nb_samples:
    #     # we need the data to be repeated
    #     sensor_idc = torch.arange(neuronSpikeTimesDense.shape[-1])
    #     repeats = 1000 // neuronSpikeTimesDense.shape[0]
    #     remainder = 1000 % neuronSpikeTimesDense.shape[0]

    #     # Create an array of zeros
    #     neuronSpikeTimesDenseRepeted = torch.zeros(
    #         (target_nb_samples, neuronSpikeTimesDense.shape[1]), dtype=neuronSpikeTimesDense.dtype)

    #     for sensor_idx in range(neuronSpikeTimesDense.shape[1]):
    #         # Repeat and concatenate the array to get exactly 1000 entries
    #         neuronSpikeTimesDenseRepeted[:, sensor_idx] = torch.cat([
    #             neuronSpikeTimesDense[:, sensor_idx].repeat(repeats),
    #             neuronSpikeTimesDense[:remainder, sensor_idx]
    #         ])

    #     neuronSpikeTimesDense = neuronSpikeTimesDenseRepeted.unsqueeze(0)
    #     print(neuronSpikeTimesDense.shape)
    #     # Add extra dimension to match the shape
    #     sensor_idc = sensor_idc.unsqueeze(0)
    #     print(sensor_idc.shape)
    #     batch_size = 1

    if neuronSpikeTimesDense.shape[0] > target_nb_samples:
        nb_splits = (
            neuronSpikeTimesDense.shape[0] - target_nb_samples) // stride + 1

        # Create sensor ID list
        sensor_idc_init = torch.arange(neuronSpikeTimesDense.shape[-1])
        sensor_idc = sensor_idc_init.unsqueeze(0).repeat(nb_splits, 1)

        start_points = range(
            0, neuronSpikeTimesDense.shape[0] - target_nb_samples + 1, stride)

        # Pre-allocate array for the sliced data
        neuronSpikeTimesDenseRepeted = torch.zeros(
            (nb_splits, target_nb_samples, neuronSpikeTimesDense.shape[1]), dtype=neuronSpikeTimesDense.dtype)

        # Fill the new array using sliding window
        for sensor_idx in range(neuronSpikeTimesDense.shape[1]):
            for split_idx, start in enumerate(start_points):
                neuronSpikeTimesDenseRepeted[split_idx, :,
                                             sensor_idx] = neuronSpikeTimesDense[start:start + target_nb_samples, sensor_idx]

        neuronSpikeTimesDense = neuronSpikeTimesDenseRepeted

        batch_size = min(nb_splits, 128)

    else:
        batch_size = 1
        neuronSpikeTimesDense = neuronSpikeTimesDense.unsqueeze(0)
        sensor_idc = torch.arange(neuronSpikeTimesDense.shape[-1])
        sensor_idc = sensor_idc.unsqueeze(0)

    # data has always 1000 entries in the first dimension and the second dimension is the number of sensors each sensor is repeated for data longer then 1000 entries
    ds_test = TensorDataset(neuronSpikeTimesDense, sensor_idc)
    generator = DataLoader(ds_test, batch_size=batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    return generator


def getFiringPatternLabels():
    """
    Retrieves a mapping of firing pattern labels.

    This function returns a dictionary that maps single-character keys to descriptive labels of various firing patterns. 
    These labels are used to classify different types of neuronal firing behaviors.

    Returns:
        dict: A dictionary where keys are single-character strings and values are descriptive labels of firing patterns.
    """

    labels_mapping = {
        'A': "Tonic spiking",
        'B': "Class 1",
        'C': "Spike frequency adaptation",
        'D': "Phasic spiking",
        'E': "Accommodation",
        'F': "Threshold variability",
        'G': "Rebound spike",
        'H': "Class 2",
        'I': "Integrator",
        'J': "Input bistability",
        'K': "Hyperpolarizing spiking",
        'L': "Hyperpolarizing bursting",
        'M': "Tonic bursting",
        'N': "Phasic bursting",
        'O': "Rebound burst",
        'P': "Mixed mode",
        'Q': "Afterpotentials",
        'R': "Basal bistability",
        'S': "Preferred frequency",
        'T': "Spike latency",
    }
    return labels_mapping


def loadWeights(map_location):
    """
    Loads pre-trained weights for the spiking neural network.

    This function loads the pre-trained weights from a specified file and maps them to the given device location.

    Args:
        map_location (str or torch.device): The device location to map the loaded weights to (e.g., 'cpu', 'cuda').

    Returns:
        dict: A dictionary containing the loaded weights for the spiking neural network.
    """

    lays = torch.load("./utils/weights.pt",
                      map_location=map_location)
    return lays


def computeActivity(nb_input, nb_neurons, input_activity, nb_steps, device):
    """
    Computes the activity of a feedforward spiking neural network layer.

    This function simulates the activity of a feedforward layer in a spiking neural network over a specified number of time steps. 
    It records the membrane potential and spike activity for each input and neuron.

    Args:
        nb_input (int): The number trials within a batch.
        nb_neurons (int): The number of neurons in the layer.
        input_activity (torch.Tensor): A tensor of shape [nb_input, nb_steps] representing the input activity over time.
        nb_steps (int): The number of time steps to simulate.
        device (torch.device): The device to perform the computation on (e.g., 'cpu', 'cuda').

    Returns:
        torch.Tensor: A tensor of shape [nb_input, nb_steps, nb_neurons] representing the spike activity of the neurons over time.
    """

    syn = torch.zeros((nb_input, nb_neurons), device=device, dtype=torch.float)
    mem = torch.zeros((nb_input, nb_neurons), device=device, dtype=torch.float)

    # Preallocate memory for recording
    mem_rec = torch.zeros((nb_steps, nb_input, nb_neurons),
                          device=device, dtype=torch.float)
    spk_rec = torch.zeros((nb_steps, nb_input, nb_neurons),
                          device=device, dtype=torch.float)

    # Compute feedforward layer activity
    for t in range(nb_steps):
        mthr = mem - 1.0
        out = activation(mthr)
        rst_out = out.detach()

        new_syn = 0.8187 * syn + input_activity[:, t]
        new_mem = (0.9048 * mem + syn) * (1.0 - rst_out)

        mem_rec[t] = mem
        spk_rec[t] = out

        mem = new_mem
        syn = new_syn

    # Transpose spk_rec to match the original output shape
    spk_rec = spk_rec.transpose(0, 1)
    return spk_rec


def computeRecurrentActivity(nb_input, nb_neurons, input_activity, layer, nb_steps, device):
    """
    Computes the activity of a recurrent spiking neural network layer.

    This function simulates the activity of a recurrent layer in a spiking neural network over a specified number of time steps. 
    It records the membrane potential and spike activity for each input and neuron.

    Args:
        nb_input (int): The number trials within a batch.
        nb_neurons (int): The number of neurons in the recurrent layer.
        input_activity (torch.Tensor): A tensor of shape [nb_input, nb_steps] representing the input activity over time.
        layer (torch.Tensor): A tensor representing the recurrent weights of the layer.
        nb_steps (int): The number of time steps to simulate.
        device (torch.device): The device to perform the computation on (e.g., 'cpu', 'cuda').

    Returns:
        torch.Tensor: A tensor of shape [nb_input, nb_steps, nb_neurons] representing the spike activity of the neurons over time.
    """

    out = torch.zeros((nb_input, nb_neurons), device=device, dtype=torch.float)
    syn = torch.zeros((nb_input, nb_neurons), device=device, dtype=torch.float)
    mem = torch.zeros((nb_input, nb_neurons), device=device, dtype=torch.float)

    # Preallocate memory for recording
    mem_rec = torch.zeros((nb_steps, nb_input, nb_neurons),
                          device=device, dtype=torch.float)
    spk_rec = torch.zeros((nb_steps, nb_input, nb_neurons),
                          device=device, dtype=torch.float)

    # Compute recurrent layer activity
    for t in range(nb_steps):
        # input activity plus last step output activity
        h1 = input_activity[:, t] + torch.einsum("ab,bc->ac", (out, layer))
        mthr = mem - 1.0
        out = activation(mthr)
        rst = out.detach()  # We do not want to backprop through the reset

        new_syn = 0.8187 * syn + h1
        new_mem = (0.9048 * mem + syn) * (1.0 - rst)

        mem_rec[t] = mem
        spk_rec[t] = out

        mem = new_mem
        syn = new_syn

    # Transpose spk_rec to match the original output shape
    spk_rec = spk_rec.transpose(0, 1)
    return spk_rec


def runSNN(inputs, nb_steps, layers, device):
    """
    Runs a spiking neural network (SNN) simulation.

    This function simulates the activity of a spiking neural network over a specified number of time steps. 
    It processes the input through a recurrent layer and a readout layer, and returns the spike activity of the output layer.

    Args:
        inputs (torch.Tensor): A tensor of shape [batch_size, timesteps, input_dim] representing the input spike data.
        nb_steps (int): The number of time steps to simulate.
        layers (tuple): A tuple containing the weight matrices (w1, w2, v1) for the network layers.
        device (torch.device): The device to perform the computation on (e.g., 'cpu', 'cuda').

    Returns:
        torch.Tensor: A tensor of shape [batch_size, timesteps, nb_outputs] representing the spike activity of the output layer.
    """

    w1, w2, v1 = layers
    bs = inputs.shape[0]
    nb_outputs = 20  # number of spiking behaviours from MN paper
    nb_hidden = 250

    h1 = torch.einsum("abc,cd->abd", (inputs, w1))
    spk_rec = computeRecurrentActivity(bs, nb_hidden, h1, v1, nb_steps, device)

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))
    s_out_rec = computeActivity(bs, nb_outputs, h2, nb_steps, device)

    return s_out_rec


def classifySpikes(generator):
    """
    Classifies spike patterns from a generator of spike data.

    This function processes spike data from multiple sensors, runs a spiking neural network (SNN) to classify the spike patterns, 
    and returns the predicted labels and softmax probabilities for each sensor.

    Args:
        generator (iterable): An iterable that yields tuples of (spikes, id), where:
            - spikes (torch.Tensor): A tensor of shape [batch_size, timesteps, sensors] containing the spike data.
            - id: An identifier for the batch (not used in this function).

    Returns:
        tuple: A tuple containing:
            - predictions_out_list (np.ndarray): A 2D array of shape [sensors, total_batches] containing the predicted labels for each sensor.
            - softmax_out_list (np.ndarray): A 3D array of shape [sensors, total_batches, num_classes] containing the softmax probabilities for each sensor.
    """

    labels_mapping = getFiringPatternLabels()
    device = checkCuda()

    # Load the pre-trained weights
    layers = loadWeights(map_location=device)

    # The log softmax function across output units
    log_softmax_fn = nn.LogSoftmax(dim=1)

    predictions_out_list = []
    softmax_out_list = []
    
    generatorObject = tqdm(generator, desc="Classifying spikes", total=len(
        generator), position=0, leave=True)
    for spikes, _ in generatorObject:
        predictions_list = []
        softmax_list = []

        channelObject = tqdm(
            range(spikes.shape[-1]), desc="Processing channels", position=1, leave=False)
        for sensorId in channelObject:
            nb_spikes = torch.sum(spikes[:, :, sensorId], axis=1)
            sensor_spikes = spikes[:, :, sensorId].to(device).unsqueeze(2)

            # Identify trials with 0 spikes
            zero_spike_trials = (nb_spikes == 0)

            # Filter out trials with 0 spikes
            non_zero_spike_trials = ~zero_spike_trials
            filtered_sensor_spikes = sensor_spikes[non_zero_spike_trials]

            # Run SNN only on trials with non-zero spikes
            spks_out = runSNN(inputs=filtered_sensor_spikes, nb_steps=spikes.shape[1], layers=layers, device=device)
            spks_sum = torch.sum(spks_out, 1)  # sum over time
            max_activity_idc = torch.argmax(spks_sum, 1)  # argmax over output units

            # MN-defined label of the spiking behaviour
            prediction = [labels_mapping[list(labels_mapping.keys())[idx.item()]] for idx in max_activity_idc]
            softmax = torch.exp(log_softmax_fn(spks_sum))

            # Initialize prediction_list and softmax_list with default values for zero spike trials
            predictions = ["No spikes"] * len(nb_spikes)
            softmaxs = [np.zeros(20)] * len(nb_spikes)

            # Fill in the values for non-zero spike trials
            non_zero_indices = torch.flatten(torch.nonzero(non_zero_spike_trials)).tolist()
            for i, idx in enumerate(non_zero_indices):
                predictions[idx] = prediction[i]
                softmaxs[idx] = softmax[i].cpu().detach().numpy()

            # Append the results to the final lists
            predictions_list.append(predictions)
            softmax_list.append(softmaxs)

        # Convert lists to NumPy arrays
        predictions_list = np.array(predictions_list)
        softmax_list = np.array(softmax_list)

        # Concatenate the results for the current batch
        if len(predictions_out_list) == 0:
            predictions_out_list = predictions_list
            softmax_out_list = softmax_list
        else:
            predictions_out_list = np.concatenate(
                (predictions_out_list, predictions_list), axis=1)
            softmax_out_list = np.concatenate(
                (softmax_out_list, softmax_list), axis=1)

    return predictions_out_list, softmax_out_list
