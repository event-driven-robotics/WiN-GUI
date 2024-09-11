from io import StringIO
import os
import numpy as np
import pandas as pd
import random
from subprocess import check_output
import torch


def create_directory(
    directory_path
    ):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile! :(
            return None
        return directory_path


def gpu_usage_df(visible=None):
    """
    Create a pandas dataframe with index, occupied memory and occupied percentage of the available GPUs from the nvidia-smi command.
    Columns: [gpu_index, gpu_mem, gpu_perc]

    Fra, Vittorio; Politecnico di Torino; EDA Group; Torino, Italy.
    """

    gpu_query_usage_df = pd.read_csv(StringIO(str(check_output(["nvidia-smi", "pmon", "-s", "m", "-c", "1"]), 'utf-8')), header=[0,1])
    
    row_read = []
    gpu_idx = []
    gpu_mem = []
    for ii in range(len(gpu_query_usage_df)):
        row_read.append([jj for jj in gpu_query_usage_df.iloc[ii].item().split(" ") if jj != ''])
        if row_read[ii][0].isdigit():
            gpu_idx.append(int(row_read[ii][0]))
        else:
            gpu_idx.append(0)
        if row_read[ii][3].isdigit():
            gpu_mem.append(int(row_read[ii][3]))
        else:
            gpu_mem.append(0)
    
    gpu_usage_df = pd.DataFrame()
    gpu_usage_df["gpu_index"] = gpu_idx
    gpu_usage_df["gpu_mem"] = gpu_mem
    
    gpu_usage_df_sum = gpu_usage_df.groupby("gpu_index").sum().reset_index()

    if visible != None:
        gpu_usage_df_sum = gpu_usage_df_sum.loc[gpu_usage_df_sum.gpu_index.isin(visible)].reset_index()
    
    gpu_perc = []
    for num,el in enumerate(gpu_usage_df_sum["gpu_index"]):
        if visible != None:
            gpu_perc.append(gpu_usage_df_sum["gpu_mem"].iloc[num]/int(np.round(torch.cuda.get_device_properties(device=visible.index(el)).total_memory/1e6,0))*100)
        else:
            gpu_perc.append(gpu_usage_df_sum["gpu_mem"].iloc[num]/int(np.round(torch.cuda.get_device_properties(device=el).total_memory/1e6,0))*100)
        #gpu_perc.append(gpu_usage_df_sum["gpu_mem"].iloc[num]/int(np.round(get_gpu_memory(el),0))*100)
    gpu_usage_df_sum["gpu_perc"] = gpu_perc

    return gpu_usage_df_sum


def check_gpu_memory_constraint(
    gpu_usage_df,
    visible,
    gpu_mem_frac
    ):
    """
    Returns a boolean value after checking if (at least one of) the available GPU(s) satisfies the constraint based on the required gpu_mem_frac to be allocated.

    Fra, Vittorio; Politecnico di Torino; EDA Group; Torino, Italy.
    """

    if visible != None:
        gpu_usage_df = gpu_usage_df.loc[gpu_usage_df.gpu_index.isin(visible)].copy()
    
    flag_available = False
    for num,el in enumerate(gpu_usage_df["gpu_perc"]):
        if 100 - el > gpu_mem_frac*100:
            flag_available = True
            break
    
    return flag_available


def set_device(
    gpu_sel=None,
    random_sel=False,
    auto_sel=False,
    visible=None,
    gpu_mem_frac=0.3
    ):
    """
    Check for available GPU and select which to use (manually, randomly or automatically).

    Fra, Vittorio; Politecnico di Torino; EDA Group; Torino, Italy.
    """

    if (gpu_sel == None) & (random_sel == False) & (auto_sel == False):

        device = torch.device("cpu")
        print("No GPU-related setting specified. Running on CPU.")

    else:

        if torch.cuda.is_available():

            if torch.cuda.device_count() > 1:
                
                gpu_df = gpu_usage_df(visible)
                
                if random_sel:
                    gpu_query_index = str(check_output(["nvidia-smi", "--format=csv", "--query-gpu=index"]), 'utf-8').splitlines()
                    gpu_devices = [int(ii) for ii in gpu_query_index if ii != 'index']
                    gpu_devices_checked = []
                    for el in gpu_devices:
                        if 100 - gpu_df[gpu_df["gpu_index"]==el]["gpu_perc"].item() > gpu_mem_frac*100:
                            gpu_devices_checked.append(el)
                    gpu_sel = random.choice(gpu_devices_checked)
                
                elif auto_sel:
                    less_occupied = gpu_df[gpu_df["gpu_mem"]==np.nanmin(gpu_df["gpu_mem"])]["gpu_index"].to_list()
                    if len(less_occupied) == 1:
                        gpu_sel = less_occupied[0]
                    else:
                        gpu_sel = random.choice(less_occupied)
                
                print("Multiple GPUs detected but single GPU selected. Setting up the simulation on {}".format("cuda:"+str(gpu_sel)))
                if visible != None:
                    device = torch.device("cuda:"+str(visible.index(gpu_sel)))
                else:
                    device = torch.device("cuda:"+str(gpu_sel))
            
            elif torch.cuda.device_count() == 1:
                print("Single GPU detected. Setting up the simulation there.")
                device = torch.device("cuda:0")
            
            torch.cuda.set_per_process_memory_fraction(gpu_mem_frac, device=device) # decrese or comment out memory fraction if more is available (the smaller the better)
        
        else:
            
            device = torch.device("cpu")
            print("GPU was asked for but not detected. Running on CPU.")
    
    return device


def load_weights(
    layers,
    map_location,
    variable=False,
    requires_grad=True
    ):
    
    if variable: # meaning that the weights are not to be loaded <-- layers is a variable name
        
        lays = layers
        
        for ii in lays:
            ii.to(map_location)
            ii.requires_grad = requires_grad
    
    else: # meaning that weights are to be loaded from a file <-- layers is a path
        
        lays = torch.load(layers, map_location=map_location)
        
        for ii in lays:
            ii.requires_grad = requires_grad
        
    return lays

