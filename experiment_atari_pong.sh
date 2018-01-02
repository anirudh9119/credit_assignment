#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
PYTHONPATH=/u/goyalani/atari-py:$PYTHONPATH python3.6 baselines/atari_ddpg/main.py --entropy_weight 0.1 --actor-lr 0.00005 --critic-lr 0.00001 --env-id Pong-v0


#PYTHONPATH=/u/goyalani/atari-py:$PYTHONPATH python3.6 baselines/atari_ddpg/main.py --entropy_weight 0 --actor-lr 0.00005 --critic-lr 0.00001 --env-id Breakout-v0

#PYTHONPATH=/u/goyalani/atari-py:$PYTHONPATH python3.6 baselines/atari_ddpg/main.py --entropy_weight 0.01 --actor-lr 0.00005 --critic-lr 0.00001
#PYTHONPATH=/u/goyalani/atari-py:$PYTHONPATH python3.6 baselines/atari_ddpg/main.py --batch-size 128 --entropy_weight 0.01 --actor-lr 0.00005 --critic-lr 0.00001
