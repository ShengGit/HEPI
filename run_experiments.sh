#!/bin/bash

python main.py \
--policy "HEPI" \
--env "HalfCheetah-v3" \

python main.py \
--policy "HEPI" \
--env "Hopper-v3" \

python main.py \
--policy "HEPI" \
--env "Walker2d-v3" \

python main.py \
--policy "HEPI" \
--env "Ant-v3" \

python main.py \
--policy "HEPI" \
--env "InvertedPendulum-v2" \
--start_timesteps 1000

python main.py \
--policy "HEPI" \
--env "InvertedDoublePendulum-v2" \
--start_timesteps 1000

python main.py \
--policy "HEPI" \
--env "Reacher-v2" \
--start_timesteps 1000

