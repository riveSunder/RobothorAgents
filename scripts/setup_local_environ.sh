#!/bin/bash
source ../robothor/bin/activate
export CHALLENGE_CONFIG=`pwd`/dataset/challenge_config.yaml
export CHALLENGE_SPLIT=train

python3 robothor_challenge/env.py
