#!/bin/bash
export CHALLENGE_CONFIG=`pwd`/dataset/challenge_config.yaml
export CHALLENGE_SPLIT=train

python3 robothor_agents/off_policy_dqn.py
