import json
import matplotlib.pyplot as plt

prefix = "/dssg/home/acct-aemzl/aemzl-user1/qbadam/inner_exp/saves/alpaca_inner_K50_alpaca_gpt4_4epoch_batch24_1e5_oldquant/block_64_step_3199"
layer_0_up_path = f"{prefix}/ratio_upproj_layer0.json"
layer_0_q_path = f"{prefix}/ratio_qproj_layer0.json"
layer_31_up_path = f"{prefix}/ratio_upproj_layer31.json"
layer_31_q_path = f"{prefix}/ratio_qproj_layer31.json"


with open(layer_0_up_path, "r") as f:
    layer_0_up = json.load(f)[0]

with open(layer_0_q_path, "r") as f:
    layer_0_q = json.load(f)[0]

with open(layer_31_up_path, "r") as f:
    layer_31_up = json.load(f)[0]

with open(layer_31_q_path, "r") as f:
    layer_31_q = json.load(f)[0]

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)  # (rows, columns, panel number)
plt.plot(layer_0_up, "r", label="layer 0")
plt.plot(layer_31_up, "b", label="layer 31", linestyle="--")
plt.title("up projection")
plt.xlabel("k")
plt.ylabel("singluar ratio")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(layer_0_q, "r", label="layer 0")
plt.plot(layer_31_q, "b", label="layer 31", linestyle="--")
plt.title("q projection")
plt.xlabel("top k")
plt.legend()

plt.tight_layout()
plt.savefig("ratio_comparison.pdf", format="pdf")
