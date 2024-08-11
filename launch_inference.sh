#!/bin/bash

MODEL_PATH="./paligemma-3b-pt-224"
PROMPT=""
IMAGE_FILE_PATH="test_images/image2.jpg"
MAX_TOKENS_TO_GENERATE=300
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="True"

/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \
