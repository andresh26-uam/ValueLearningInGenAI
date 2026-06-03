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

GROUP = "VSLRMVSBTRM_PKUlong"
runs: Runs = api.runs("<blinded>", include_sweeps=False, filters={"state": "finished", "group": GROUP})
hist_list = [] 


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
    ('eval/representativeness', 'Value System Accuracy (VSA)'),
    ('eval/avg_coherence', 'Avg. Grounding Accuracy (AGA)'),
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
        'SEQ-RM': df[df['loss_func_type'] == 'FIRST_GROUNDING_THEN_VALUE_SYSTEM'],
    }

    colors = {'VSL-RM': 'tab:blue', 'BT-RM': 'tab:red', 'SEQ-RM': 'tab:green'}
    line_styles = {
        'eval/representativeness': '-',
        'eval/avg_coherence': ':',
    }

    # First pass: collect all aggregated data and find max epoch
    all_agg_data = {}
    max_epoch = 0

    for group_name, group_df in groups.items():
        if group_df.empty:
            continue
        all_agg_data[group_name] = {}

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
            agg = agg.sort_values('train/epoch').reset_index(drop=True)
            all_agg_data[group_name][metric_key] = agg

            # Track max epoch
            max_epoch = max(max_epoch, agg['train/epoch'].max())

    # Second pass: pad shorter series to max_epoch using last values
    padding_positions = set()  # Track positions where padding starts
    for group_name in all_agg_data:
        for metric_key in all_agg_data[group_name]:
            agg = all_agg_data[group_name][metric_key]
            if len(agg) > 0:
                last_row = agg.iloc[-1]
                if last_row['train/epoch'] < max_epoch:
                    # Record padding start position
                    padding_positions.add(last_row['train/epoch'])
                    # Append a row at max_epoch with the last metric values
                    new_row = pd.DataFrame({
                        'train/epoch': [max_epoch],
                        'mean': [last_row['mean']],
                        'std_pos': [last_row['std_pos']],
                        'std_neg': [last_row['std_neg']]
                    })
                    agg = pd.concat([agg, new_row], ignore_index=True)
                    all_agg_data[group_name][metric_key] = agg

    # Third pass: plot the padded data
    for group_name in all_agg_data:
        for metric_key, metric_label in metrics:
            if metric_key not in all_agg_data[group_name]:
                continue

            agg = all_agg_data[group_name][metric_key]
            if agg.empty:
                continue

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

    # Add vertical lines at padding positions
    for pos in sorted(padding_positions):
        ax.axvline(x=pos, color='grey', linestyle='--', alpha=0.6, linewidth=1.5)

    ax.set_xlabel('Training Epoch', fontsize=26)
    #ax.set_title('Representativeness and Average Coherence', fontsize=26)
    ax.legend(loc='best', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f'Saved plot to {out_path}')