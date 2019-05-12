#!/bin/bash

echo ""
echo "To run in interact mode, execute the following:"
echo ""
echo "    interact -p GPU-shared -N 1 --gres=gpu:k80:1 --ntasks-per-node=1"
echo ""
echo "Then load the following modules:"
echo ""
echo "    module load cmake/3.11.4 gcc/5.3.0 cuda/10.1 anaconda/5.2.0"
echo ""
#really not sure about what exactly to add for nodes etc.
