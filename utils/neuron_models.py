"""
neuron_models.py

This module implements various neuron models for use in spiking neural network (SNN) simulations. 
The primary focus is on biologically plausible neuron models, such as the Mihalas-Niebur (MN) neuron model.

Main Components:
- MN_neuron: A class implementing the Mihalas-Niebur neuron model. This model includes parameters for 
  synaptic weights, membrane potential, and other neuron-specific properties.
- IZ_neuron: A class implementing the Izhikevich neuron model. This model simulates spiking and bursting 
  behavior of neurons.
- LIF_neuron: A class implementing the Leaky Integrate-and-Fire neuron model. This model is a simple 
  representation of neuronal activity.
- CuBaLIF_neuron: A class implementing the Current-Based Leaky Integrate-and-Fire neuron model. This model 
  includes a synaptic current to simulate more complex neuronal dynamics.

Classes and Functions:
- MN_neuron: A PyTorch module representing the Mihalas-Niebur neuron model.
  - NeuronState: A named tuple to store the state variables of the neuron (V, i1, i2, Thr, spk).
  - __init__(self, nb_inputs, parameters_combination, dt=1/1000, ...): Initializes the neuron with the given parameters.
  - forward(self, input): Defines the forward pass of the neuron model.
- IZ_neuron: A PyTorch module representing the Izhikevich neuron model.
  - __init__(self, nb_inputs, parameters_combination, dt=1/1000, ...): Initializes the neuron with the given parameters.
  - forward(self, input): Defines the forward pass of the neuron model.
- LIF_neuron: A PyTorch module representing the Leaky Integrate-and-Fire neuron model.
  - __init__(self, nb_inputs, parameters_combination, dt=1/1000, ...): Initializes the neuron with the given parameters.
  - forward(self, input): Defines the forward pass of the neuron model.
- CuBaLIF_neuron: A PyTorch module representing the Current-Based Leaky Integrate-and-Fire neuron model.
  - __init__(self, nb_inputs, parameters_combination, dt=1/1000, ...): Initializes the neuron with the given parameters.
  - forward(self, input): Defines the forward pass of the neuron model.

Dependencies:
- torch: PyTorch library for tensor computations and neural network operations.
- surrogate_gradient: Custom autograd function for spiking nonlinearity with a surrogate gradient.

Usage:
This module is intended to be used as part of the WiN-GUI project for simulating spiking neural networks. 
To use this module, ensure that the required dependencies are installed.

Example:
    import torch
    from neuron_models import MN_neuron, IZ_neuron, LIF_neuron, CuBaLIF_neuron

    # Define neuron parameters
    nb_inputs = 10
    parameters_combination = {...}

    # Initialize the MN neuron
    mn_neuron = MN_neuron(nb_inputs, parameters_combination)

    # Define input tensor
    input_tensor = torch.randn(nb_inputs)

    # Run the forward pass
    output = mn_neuron(input_tensor)

License:
This project is licensed under the GPL-3.0 License. See the LICENSE file for more details.

"""

from collections import namedtuple

import torch
import torch.nn as nn
from utils.surrogate_gradient import activation


# MN neuron
class MN_neuron(nn.Module):
    NeuronState = namedtuple("NeuronState", ["V", "i1", "i2", "Thr", "spk"])

    def __init__(
        self,
        nb_inputs,
        parameters_combination,
        dt=1 / 1000,
        a=5,
        A1=10,
        A2=-0.6,
        b=10,
        G=50,
        k1=200,
        k2=20,
        R1=0,
        R2=1,
    ):  # default combination: M2O of the original paper
        super(MN_neuron, self).__init__()

        # One-to-one synapse
        self.linear = nn.Parameter(torch.ones(
            1, nb_inputs))
        self.N = nb_inputs
        one2N_matrix = torch.ones(1, nb_inputs)
        # define some constants
        self.C = 1
        self.EL = -0.07  # V
        self.Vr = -0.07  # V
        self.Tr = -0.06  # V
        self.Tinf = -0.05  # V

        # define parameters
        self.a = a
        self.A1 = A1
        self.A2 = A2
        self.b = b  # 1/s
        self.G = G * self.C  # 1/s
        self.k1 = k1  # 1/s
        self.k2 = k2  # 1/s
        self.R1 = R1  # not Ohm?
        self.R2 = R2  # not Ohm?
        self.dt = dt  # get dt from sample rate!

        # update parameters from GUI
        for param_name in list(parameters_combination.keys()):
            eval_string = (
                "self.{}".format(param_name) + " = " +
                str(parameters_combination[param_name])
            )
            exec(eval_string)

        # set up missing parameters
        self.a = nn.Parameter(one2N_matrix * self.a)
        self.A1 = nn.Parameter(one2N_matrix * self.A1 *
                               self.C)
        self.A2 = nn.Parameter(one2N_matrix * self.A2 *
                               self.C)

        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(
                V=torch.ones(x.shape[0], self.N, device=x.device) * self.EL,
                i1=torch.zeros(x.shape[0], self.N, device=x.device),
                i2=torch.zeros(x.shape[0], self.N, device=x.device),
                Thr=torch.ones(x.shape[0], self.N,
                               device=x.device) * self.Tinf,
                spk=torch.zeros(x.shape[0], self.N, device=x.device),
            )

        V = self.state.V
        i1 = self.state.i1
        i2 = self.state.i2
        Thr = self.state.Thr

        i1 += -self.k1 * i1 * self.dt
        i2 += -self.k2 * i2 * self.dt
        V += self.dt * (self.linear * x + i1 + i2 -
                        self.G * (V - self.EL)) / self.C
        Thr += self.dt * (self.a * (V - self.EL) - self.b * (Thr - self.Tinf))

        spk = activation(V - Thr)

        i1 = (1 - spk) * i1 + (spk) * (self.R1 * i1 + self.A1)
        i2 = (1 - spk) * i2 + (spk) * (self.R2 * i2 + self.A2)
        Thr = ((1 - spk) * Thr) + \
            ((spk) * torch.max(Thr, torch.tensor(self.Tr)))
        V = ((1 - spk) * V) + ((spk) * self.Vr)

        self.state = self.NeuronState(V=V, i1=i1, i2=i2, Thr=Thr, spk=spk)

        return spk

    def reset(self):
        self.state = None


class IZ_neuron(nn.Module):
    # u = membrane recovery variable
    NeuronState = namedtuple('NeuronState', ['V', 'u', 'spk'])

    def __init__(
        self,
        nb_inputs,
        parameters_combination,
        dt=1/1000,
        a=0.02,
        b=0.2,
        c=-65,
        d=8,
    ):
        super(IZ_neuron, self).__init__()

        # One-to-one synapse
        self.linear = nn.Parameter(torch.ones(
            1, nb_inputs))
        self.N = nb_inputs
        # define some constants
        self.spike_value = 35  # spike threshold

        # define parameters
        self.a = a
        self.b = b
        self.c = c  # reset potential
        self.d = d
        self.dt = dt*1E3  # convert from sec to ms

        # update parameters from GUI
        for param_name in list(parameters_combination.keys()):
            eval_string = (
                "self.{}".format(param_name) + " = " +
                str(parameters_combination[param_name])
            )
            exec(eval_string)

        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(V=torch.ones(x.shape[0], self.N, device=x.device) * self.c,
                                          u=torch.zeros(
                                              x.shape[0], self.N, device=x.device) * self.b*self.c,
                                          spk=torch.zeros(x.shape[0], self.N, device=x.device))

        V = self.state.V
        u = self.state.u

        numerical_res = round(self.dt)
        if self.dt > 1:
            output_spike = torch.zeros_like(self.state.spk)
            for i in range(numerical_res):
                V = V + (((0.04 * V + 5) * V) + 140 - u + x)
                u = u + self.a * (self.b * V - u)

                # create spike when threshold reached
                spk = activation(V - self.spike_value)
                output_spike = output_spike + spk

                # (reset membrane voltage) or (only update)
                V = (spk * self.c) + ((1 - spk) * V)
                # (reset recovery) or (update currents)
                u = (spk * (u + self.d)) + ((1 - spk) * u)
        else:
            V = V + self.dt*(((0.04 * V + 5) * V) + 140 - u + x)
            u = u + self.dt*self.a * (self.b * V - u)

            # create spike when threshold reached
            spk = activation(V - self.spike_value)
            output_spike = spk

            # (reset membrane voltage) or (only update)
            V = (spk * self.c) + ((1 - spk) * V)
            # (reset recovery) or (update currents)
            u = (spk * (u + self.d)) + ((1 - spk) * u)

        self.state = self.NeuronState(V=V, u=u, spk=spk)

        return spk

    def reset(self):
        self.state = None


class LIF_neuron(nn.Module):
    NeuronState = namedtuple("NeuronState", ["V", "spk"])

    def __init__(
            self,
            nb_inputs,
            parameters_combination,
            dt=1/1000,
            beta=1.0,
            thr=1.0,
            R=1.0,
    ):
        super(LIF_neuron, self).__init__()

        self.nb_inputs = nb_inputs
        self.beta = beta
        self.threshold = thr
        # self.V_rest = -0.04  # -40mV
        self.R = R
        self.dt = dt

        # update parameters from GUI
        for param_name in list(parameters_combination.keys()):
            eval_string = (
                "self.{}".format(param_name) + " = " +
                str(parameters_combination[param_name])
            )
            exec(eval_string)

        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(
                V=torch.zeros(x.shape[0], self.nb_inputs,
                              device=x.device),
                spk=torch.zeros(x.shape[0], self.nb_inputs, device=x.device),
            )
        V = self.state.V
        spk = self.state.spk

        # V = (self.beta * V + (1.0-self.beta) * x * self.R) * (1.0 - spk)
        V = (self.beta * V + x * self.R) * (1.0 - spk) # reset mechanism: zero
        spk = activation(V-self.threshold)

        self.state = self.NeuronState(V=V, spk=spk)

        return spk

    def reset(self):
        self.state = None


class CuBaLIF_neuron(nn.Module):
    NeuronState = namedtuple("NeuronState", ["V", "syn", "spk"])

    def __init__(
            self,
            nb_inputs,
            parameters_combination,
            dt=1/1000,
            alpha=1.0,
            beta=1.0,
            thr=1.0,
            R=1.0,
    ):
        super(CuBaLIF_neuron, self).__init__()

        self.nb_inputs = nb_inputs
        self.alpha = alpha
        self.beta = beta
        self.threshold = thr
        # self.V_rest = -0.04  # -40mV
        self.R = R
        self.dt = dt

        # update parameters from GUI
        for param_name in list(parameters_combination.keys()):
            eval_string = (
                "self.{}".format(param_name) + " = " +
                str(parameters_combination[param_name])
            )
            exec(eval_string)

        self.state = None

    def forward(self, x):
        if self.state is None:
            self.state = self.NeuronState(
                V=torch.zeros(x.shape[0], self.nb_inputs,
                              device=x.device),
                syn=torch.zeros(x.shape[0], self.nb_inputs,
                                device=x.device),
                spk=torch.zeros(x.shape[0], self.nb_inputs, device=x.device),
            )
        V = self.state.V
        spk = self.state.spk
        syn = self.state.syn

        # syn = self.alpha*syn + spk
        # V = (self.beta * V + (1.0-self.beta) * x *
        #      self.R + (1.0-self.beta)*syn) * (1.0 - spk)
        syn = self.alpha*syn + x*self.R
        V = (self.beta * V + syn) * (1.0 - spk) # reset mechanism: zero
        spk = activation(V-self.threshold)

        self.state = self.NeuronState(V=V, syn=syn, spk=spk)

        return spk

    def reset(self):
        self.state = None
