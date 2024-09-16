import json
import matplotlib.pyplot as plt

model1_path = "/dssg/home/acct-aemzl/aemzl-user1/qbadam/inner_exp/saves/alpaca_inner_K50_alpaca_gpt4_4epoch_batch24_1e5_oldquant/block_32_step_1599/ranks_model_diff.json"

model2_path = "/dssg/home/acct-aemzl/aemzl-user1/qbadam/inner_exp/saves/alpaca_inner_K50_alpaca_gpt4_4epoch_batch24_1e5_oldquant/block_64_step_3199/ranks_model_diff.json"


with open(model1_path, "r") as f:
    ranks_model1_diff = json.load(f)

with open(model2_path, "r") as f:
    ranks_model2_diff = json.load(f)

num_layers = len(ranks_model1_diff["q_proj"])  # 假设所有列表长度相同
layers = list(range(1, num_layers + 1))

plt.figure(figsize=(12, 8))

# 定义投影名称和对应的颜色
projections = {
    "q_proj": "blue",
    "k_proj": "orange",
    "v_proj": "green",
    "o_proj": "black",
    "gate_proj": "red",
    "down_proj": "purple",
    "up_proj": "brown",
}

# 绘制每个投影的有效秩
for proj_name, color in projections.items():
    ranks1 = ranks_model1_diff.get(proj_name)
    ranks2 = ranks_model2_diff.get(proj_name)

    if ranks1 is not None and ranks2 is not None:
        plt.plot(layers, ranks1, label=f"{proj_name} 1-epoch - Base", color=color)
        plt.plot(
            layers,
            ranks2,
            label=f"{proj_name} 2-epoch - Base",
            color=color,
            linestyle="--",
        )

plt.legend()
plt.title("Effective Rank of Weight Differences between Models and Base Model")
plt.xlabel("Layer")
plt.ylabel("Effective Rank of Weight Difference")
plt.savefig("effective_rank_difference_comparison.pdf", format="pdf")
plt.show()
