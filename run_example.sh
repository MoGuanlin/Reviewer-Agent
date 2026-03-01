#!/bin/bash
# 激活环境 (根据实际情况修改)
# source activate reviewer

# 设置代理 (根据实际情况修改)
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

PYTHON="/remote-home/MoGuanlin/anaconda3/envs/reviewer/bin/python3"
echo "Running review on sample paper..."
$PYTHON main.py --pdf "agent_old/paper_review_agent/socg26-paper389.pdf" --model "gemini-2.5-pro" --thinking_level "low" --outdir "outputs/example_run_new_prompt"
