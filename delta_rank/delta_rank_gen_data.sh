
python delta_rank_gen.py \
    --base_model_path /dssg/home/acct-aemzl/aemzl-user1/modelscope_models/llama3-8b \
    --target_model_path /dssg/home/acct-aemzl/aemzl-user1/qbadam/inner_saves/alpaca_inner_K50_alpaca_gpt4_4epoch_batch24_1e5_oldquant/block_32_step_1599 \
    --save_path /dssg/home/acct-aemzl/aemzl-user1/qbadam/inner_exp/alpaca_inner_K50_alpaca_gpt4_4epoch_batch24_1e5_oldquant/block_32_step_1599/ranks_model_diff.json
