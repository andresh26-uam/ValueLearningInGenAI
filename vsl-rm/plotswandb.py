import wandb
import pandas as pd 
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
api = wandb.Api()
from wandb.apis.public import Runs
# Project is specified by <entity/project-name>

GROUP = "VSLRMVSBTRM_ULTRA"
runs: Runs = api.runs("andres-hsn/huggingface", include_sweeps=False, filters={"state": "finished", "group": GROUP})
hist_list = [] 

from pprint import pprint

for run in runs: 
    # require at least one of the target metrics in the run summary/history
    if not ('eval/representativeness' in run.summary or 'eval/avg_coherence' in run.summary):
        continue
    # capture loss type, defaulting to UNKNOWN
    loss_type = run.config.get('loss_func_type', 'UNKNOWN')

    # fetch history for both metrics (may contain NaNs)
    hist = run.history(keys=['train/epoch', 'eval/representativeness', 'eval/avg_coherence'])
    hist['name'] = run.config.get("run_name", str(run.id))
    hist['loss_func_type'] = loss_type
    hist_list.append(hist)

df = pd.concat(hist_list, ignore_index=True)

# metrics to plot
metrics = [
    ('eval/representativeness', 'Representativeness'),
    ('eval/avg_coherence', 'Avg. Coherence'),
]

# keep numeric rows only
df = df.copy()
df['train/epoch'] = pd.to_numeric(df['train/epoch'], errors='coerce')

out_path = Path(f'results/plots/representativeness_vs_coherence_{GROUP}.pdf')
out_path.parent.mkdir(parents=True, exist_ok=True)

if df.dropna(subset=['train/epoch'] + [m[0] for m in metrics]).empty:
    print('No data to plot after filtering; exiting.')
else:
    fig, ax = plt.subplots(figsize=(12, 8))

    groups = {
        'VSL-RM': df[df['loss_func_type'] == 'DEFAULT'],
        'BT-RM': df[df['loss_func_type'] == 'ONLY_VALUE_SYSTEM'],
    }

    colors = {'VSL-RM': 'tab:blue', 'BT-RM': 'tab:red'}
    line_styles = {
        'eval/representativeness': '-',
        'eval/avg_coherence': '--',
    }

    for group_name, group_df in groups.items():
        if group_df.empty:
            continue

        for metric_key, metric_label in metrics:
            # drop rows without metric values
            g = group_df.dropna(subset=[metric_key, 'train/epoch'])
            if g.empty:
                continue

            # aggregate per epoch: compute mean and one-sided std devs across runs
            rows = []
            for epoch, series in g.groupby('train/epoch')[metric_key]:
                vals = series.dropna().to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                m = float(vals.mean())
                above = vals[vals > m] - m
                below = m - vals[vals < m]
                std_pos = float(above.std(ddof=1)) if above.size > 0 else 0.0
                std_neg = float(below.std(ddof=1)) if below.size > 0 else 0.0
                rows.append({'train/epoch': float(epoch), 'mean': m, 'std_pos': std_pos, 'std_neg': std_neg})

            agg = pd.DataFrame(rows)
            if agg.empty:
                continue
            agg = agg.sort_values('train/epoch')

            x = agg['train/epoch'].to_numpy()
            y = agg['mean'].to_numpy()
            ystd_pos = agg['std_pos'].to_numpy()
            ystd_neg = agg['std_neg'].to_numpy()

            linestyle = line_styles.get(metric_key, '-')
            ax.plot(x, y, label=f'{group_name} {metric_label}', color=colors.get(group_name), linestyle=linestyle, linewidth=2)
            # shade negative deviation (mean - std_neg -> mean)
            ax.fill_between(x, y - ystd_neg, y, alpha=0.35, color=colors.get(group_name))
            # shade positive deviation (mean -> mean + std_pos)
            ax.fill_between(x, y, y + ystd_pos, alpha=0.25, color=colors.get(group_name))

    ax.set_xlabel('Training Epoch', fontsize=26)
    #ax.set_title('Representativeness and Average Coherence', fontsize=26)
    ax.legend(loc='best', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f'Saved plot to {out_path}')