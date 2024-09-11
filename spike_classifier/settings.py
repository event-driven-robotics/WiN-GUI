import argparse


parser = argparse.ArgumentParser()

# Auto-selection of GPU
parser.add_argument('-auto_gpu',
                    type=bool,
                    default=False,
                    help='Enable or not auto-selection of GPU to use.')
# Manual selection of GPU
parser.add_argument('-manual_gpu_idx',
                    type=int,
                    default=0,
                    help='Set which GPU to use.')
# (maximum) GPU memory fraction to be allocated
parser.add_argument('-gpu_mem_frac',
                    type=float,
                    default=0.3,
                    help='The maximum GPU memory fraction to be used by this experiment.')


args = parser.parse_args()

settings = vars(args)