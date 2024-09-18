python sigular_ratio.py \
    --base_model_path /dssg/home/acct-aemzl/aemzl-user1/modelscope_models/llama3-8b \
    --target_model_path /dssg/home/acct-aemzl/aemzl-user1/qbadam/inner_saves/alpaca_inner_K50_alpaca_gpt4_4epoch_batch24_1e5_oldquant/block_64_step_3199 \
    --save_path /dssg/home/acct-aemzl/aemzl-user1/qbadam/inner_exp/saves/alpaca_inner_K50_alpaca_gpt4_4epoch_batch24_1e5_oldquant/block_64_step_3199/ratio_qproj_layer0.json \
    --num_layer 0 \
    --target_module q_proj