#!/usr/bin/env bash

for combo in {"soft ","gru "};
    do gnome-terminal --tab -e "bash -c \"export PYTHONPATH=/home/victor/ece547:\$PYTHONPATH; /home/victor/anaconda3/bin/python main.py $combo gru true _ifo 500; exec bash\"" &
    sleep 15
done