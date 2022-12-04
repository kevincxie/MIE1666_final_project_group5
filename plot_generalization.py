import os
import pickle
import matplotlib.pyplot as plt
import json

results_path = "gen_sgld"

metrics_path = os.path.join(results_path, "metrics.pkl")
with open(metrics_path, 'rb') as f:
    perf_dict = pickle.load(f)

oracle_cost = perf_dict["oracle_cost"]
perf_per_model = perf_dict["perf_per_model"]

fig, ax = plt.subplots()
ax.set_title('Generalization Performance Ablations')
ax.set_xlabel('Number of Training Samples')
ax.set_ylabel('Cost')
for model_name, perfd in perf_per_model.items():
    ax.plot(perfd['train_sizes'], perfd['avg_test_costs'], label=model_name)
# ax.axhline(oracle_cost, label='oracle', c='red')
ax.legend()
fig.savefig(os.path.join(results_path, "gen_perf.png"))


# results_turbo = "results/lamcts/turbo900"
results_turbo = "results/turbo/500"
offset = 2.5
with open(results_turbo, 'rt') as f:
    turbo_perf = json.load(f)

fig, ax = plt.subplots()
ax.plot([x+offset for x in turbo_perf], label='turbo')
ax.set_yscale('log')
for model_name, perfd in perf_per_model.items():
    if model_name == "full":
        colors = ['pink','coral', 'crimson']
    else:
        colors = ['lightgreen', 'lime', 'green']

    # if model_name == ''
    for i, avg_cost in enumerate(perfd['avg_test_costs']):
        train_size = perfd['train_sizes'][i]
        ax.axhline(avg_cost + offset, linestyle='--', c = colors[i], label=f'{model_name}: {train_size}')
        # ax.plot(perfd['train_sizes'], perfd['avg_test_costs'], label=model_name)
ax.legend()
ax.set_xlabel('Number of function evaluations')
ax.set_ylabel('Adjusted log cost')
ax.set_title('Performance Comparison to Black Box Optimizers')
fig.savefig(os.path.join(results_path, "gen_perf_comp.png"))
