source scripts/env.sh

python3 generative_agents/generate_conversations.py \
    --out-dir ./data/multimodal_dialog/example/ \
    --prompt-dir ./prompt_examples \
    --events --session --summary --num-sessions 2 \
    --persona --blip-caption \
    --num-days 10 --num-events 2 --max-turns-per-session 5 --num-events-per-session 1
