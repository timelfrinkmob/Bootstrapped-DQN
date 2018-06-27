#!/bin/bash
for seed in 1 2 3
do
    for env in Pong Breakout;
    do
	    for exp in "--bootstrap --no-noisy --no-greedy" "--no-bootstrap --noisy --no-greedy" "--no-bootstrap --no-noisy --no-greedy"  "--no-bootstrap --no-noisy --greedy" ;
	    do
            python train.py --num-steps=5000000 --env="$env" $exp --seed=$seed
        done
    done
done

aws s3 cp ./models/ s3://thesis-tim-files/baseline/atari/ --recursive
sudo shutdown -P now
