#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
PYTHONPATH=/u/goyalani/atari-py:$PYTHONPATH python3.6 baselines/atari_ddpg/main.py --env-id BeamRider-v0 --entropy_weight 0.01 --actor-lr 0.00001 --critic-lr 0.00005 --clip-norm 50 --layer-norm
#0.00005 0.0001
#0.0001 0.0005
#0.0001 0.0001
#0.00001 0.0001

#PYTHONPATH=/u/goyalani/atari-py:$PYTHONPATH python3.6 baselines/atari_ddpg/main.py --batch-size 128 --entropy_weight 0.01 --actor-lr 0.00005 --critic-lr 0.00001
