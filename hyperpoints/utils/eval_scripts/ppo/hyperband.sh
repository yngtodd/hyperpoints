#!/bin/bash

BATCH_SIZE=16
COUNTER=0
SEED=1000
CUDA=9

for i in {1..100}
do
    if [[ $COUNTER > $BATCH_SIZE ]]; then CUDA=$((CUDA+1)); COUNTER=$((0)); fi
    vel /Users/yngtodd/src/develop/vel/examples-configs/rl/atari/journal/ppo/hyperband_qbert_ppo.yaml train -r $i -s $SEED -d cuda:$CUDA &
    COUNTER=$((COUNTER+1))
    SEED=$((SEED+1))
    echo $CUDA
done
