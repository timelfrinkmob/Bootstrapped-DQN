#!/bin/bash
for seed in 1 2 3
do
    for env in Pong Breakout;
    do
	    for exp in "--no-bootstrap --noisy" "--bootstrap --no-noisy" "--no-bootstrap --no-noisy";
	    do
            python train.py --num-steps=1000 --env="$env" $exp --seed=$seed
        done
    done
done

