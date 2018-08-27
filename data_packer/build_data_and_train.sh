#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python build_vqa_data.py

python train_question_generator.py