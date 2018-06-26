#!/bin/bash
for seed in 1 2 3
do
    for i in 10 20 30 40 50 60 70 80 90 100;
    do
	    for exp in "--bootstrap --no-noisy --no-greedy" "--no-bootstrap --noisy --no-greedy" "--no-bootstrap --no-noisy --no-greedy"  "--no-bootstrap --no-noisy --greedy" ;
	    do
            python train_chain.py --n=$i $exp --seed=$seed
        done
    done
done

aws s3 cp ./models/ s3://thesis-tim-files/baseline/chain/ --recursive
sudo shutdown -P now
