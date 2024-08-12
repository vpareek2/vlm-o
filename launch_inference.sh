#!/bin/bash

MODEL_PATH="./paligemma_vqav2"
PROMPT="What is happening in the photo?"
IMAGE_FILE_PATH="test_images/image1.jpg"
MAX_TOKENS_TO_GENERATE=300
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"

python3 inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \
