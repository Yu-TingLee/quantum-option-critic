import os
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator

PLOT_ENVS = [
    "Acrobot-v1",
]

PLOT_GROUPS = {
    "Group_1": [
        "Random", "Classical-re", "Hybrid_FOTP-re", "Hybrid_FO-re", "Hybrid_FT-re", "Hybrid_FP-re"
    ],
    "Group_2": [
        "Random", "Classical-re", "Hybrid_F-re", "Hybrid_O-re", "Hybrid_T-re", "Hybrid_P-re"
    ],
    # "Options_all": [
    #     "Classical-re", "Classical-re-3options", "Classical-re-4options", "Hybrid_P-re", "Hybrid_P-re-3options", "Hybrid_P-re-4options"
    # ]
}

PLOT_MODELS = {model for group in PLOT_GROUPS.values() for model in group}

def get_data(runs_dir):
    all_data = []
    size_guidance = {event_accumulator.SCALARS: 0}
    
    folders = glob.glob(os.path.join(runs_dir, "*_*_*"))
    run_counts = defaultdict(int)
    print(f"Found {len(folders)} folders in {runs_dir}.")
    for folder in folders:
        folder_name = os.path.basename(folder)
        parts = folder_name.split("_")
        
        env_name = parts[1]
        model_name = "_".join(parts[2:])
        
        # Filter envs and models early
        if env_name not in PLOT_ENVS:
            continue
        if model_name not in PLOT_MODELS:
            continue
        run_counts[(env_name, model_name)] += 1

        event_files = glob.glob(os.path.join(folder, "events.out.tfevents.*"))
        print(f"Processing ({model_name}, {env_name})")
        
        for f in event_files:
            ea = event_accumulator.EventAccumulator(f, size_guidance=size_guidance)
            ea.Reload()
            reward_tag = "episodic_rewards_total_steps"

            events = ea.Scalars(reward_tag)
            for i, e in enumerate(events):
                all_data.append({
                    'total_steps': e.step,
                    'episode': i,
                    'episodic_reward': e.value,
                    'model_name': model_name,
                    'env_name': env_name,
                    'run_id': folder_name
                })
    print("\n--- Run Counts ---")
    for (env, model), count in run_counts.items():
        print(f"Env: {env} | Model: {model} | Runs detected: {count}")
    return pd.DataFrame(all_data)

# Load Data
df = get_data('runs')

df['total_steps_binned'] = (df['total_steps'] // 1000) * 1000

# Group by Bin -> Calculate Mean/Std of raw data
df_step_stats = df.groupby(['total_steps_binned', 'model_name', 'env_name'])['episodic_reward'].agg(['mean', 'std']).reset_index()
df_step_stats['std'] = df_step_stats['std'].fillna(0)
df_step_stats = df_step_stats.sort_values(by=['model_name', 'env_name', 'total_steps_binned'])

# Smoothing window
df_step_stats['mean_smooth'] = df_step_stats.groupby(['model_name', 'env_name'])['mean'].transform(
    lambda x: x.rolling(window=20, min_periods=1).mean()
)
df_step_stats['std_smooth'] = df_step_stats.groupby(['model_name', 'env_name'])['std'].transform(
    lambda x: x.rolling(window=20, min_periods=1).mean()
)

# Group by Episode -> Calculate Mean/Std of raw data
df_ep_stats = df.groupby(['episode', 'model_name', 'env_name'])['episodic_reward'].agg(['mean', 'std']).reset_index()
df_ep_stats['std'] = df_ep_stats['std'].fillna(0)
df_ep_stats = df_ep_stats.sort_values(by=['model_name', 'env_name', 'episode'])

# Smoothing window
df_ep_stats['mean_smooth'] = df_ep_stats.groupby(['model_name', 'env_name'])['mean'].transform(
    lambda x: x.rolling(window=20, min_periods=1).mean()
)
df_ep_stats['std_smooth'] = df_ep_stats.groupby(['model_name', 'env_name'])['std'].transform(
    lambda x: x.rolling(window=20, min_periods=1).mean()
)


os.makedirs('plots', exist_ok=True)
sns.set_theme(style="whitegrid")


palette = sns.color_palette("Set1")
unique_envs = df['env_name'].unique()

for env_name in unique_envs:
    env_step_data = df_step_stats[df_step_stats['env_name'] == env_name]
    env_ep_data = df_ep_stats[df_ep_stats['env_name'] == env_name]
    
    # Filter raw data for table stats
    env_raw_data = df[df['env_name'] == env_name]

    for group_name, model_list in PLOT_GROUPS.items():
        group_raw_data = env_raw_data[env_raw_data['model_name'].isin(model_list)]
        if group_raw_data.empty:
            continue

        print(f"Average Reward per Episode: {env_name} [{group_name}]")
        print("="*50)
        
        episodes_per_run = group_raw_data.groupby(['model_name', 'run_id']).size()
        ep_stats = episodes_per_run.groupby('model_name').agg(['mean', 'std'])
        
        stats = group_raw_data.groupby('model_name')['episodic_reward'].agg(['sum', 'count'])
        num_unique_runs = group_raw_data.groupby('model_name')['run_id'].nunique()
        
        stats['avg_episodes'] = ep_stats['mean']
        stats['episodes_std'] = ep_stats['std']
        stats['avg_reward'] = stats['sum'] / stats['count']
        
        # Calculate performance relative to Classical if it exists
        if 'Classical-re' in stats.index:
            baseline_reward = stats.loc['Classical-re', 'avg_reward']
            stats['rel_reward'] = abs(stats['avg_reward']) / abs(baseline_reward)
            stats['rel_reward'] = stats['rel_reward'].map(lambda x: f"{x:.2f}x")
        else:
            stats['rel_reward'] = "N/A"

        # Print formatted results
        print(stats[['avg_episodes', 'episodes_std', 'avg_reward', 'rel_reward']].to_string(float_format="{:.2f}".format))
        print("="*50 + "\n")
            
        def plot_group(data, x_col, y_col, err_col, filename, x_label, x_limit=None, x_formatter=None):
            fig, ax = plt.subplots(figsize=(3.5, 2))
            
            for model in model_list:
                subset = data[data['model_name'] == model]
                if subset.empty:
                    continue
                display_label = model.replace("-re", "")
                color = palette[model_list.index(model) % len(palette)]
                
                ax.plot(
                    subset[x_col], 
                    subset[y_col], 
                    label=display_label, 
                    color=color, 
                    linewidth=1.0
                )
                
                ax.fill_between(
                    subset[x_col],
                    subset[y_col] - subset[err_col],
                    subset[y_col] + subset[err_col],
                    color=color,
                    alpha=0.25,
                    linewidth=0
                )

            # Formatting
            ax.set_title(f"{env_name}", fontsize=10)
            ax.set_xlabel(x_label, fontsize=8)
            ax.set_ylabel('Episodic Reward', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8, pad=0)
            if x_limit:
                ax.set_xlim(0, x_limit)
            if env_name == "CartPole-v1":
                ax.set_ylim(0, 500)
            elif env_name == "Acrobot-v1":
                ax.set_ylim(-500, 0)
            
            if x_formatter:
                ax.xaxis.set_major_locator(ticker.FixedLocator([0, 200000, 400000, 600000, 800000, 1000000]))
                ax.xaxis.set_major_formatter(x_formatter)
                ax.xaxis.get_offset_text().set_fontsize(8)
            
            plt.legend(loc='best', fontsize=5, framealpha=0.8)
            plt.tight_layout(pad=0.1)
            plt.savefig(os.path.join('plots', filename), dpi=600, bbox_inches='tight', pad_inches=0.02)
            plt.close()

        # Reward vs Total Steps
        formatter = ticker.ScalarFormatter(useMathText=False)
        formatter.set_scientific(True)
        formatter.set_powerlimits((6, 6))
        
        plot_group(
            data=env_step_data,
            x_col='total_steps_binned',
            y_col='mean_smooth',
            err_col='std_smooth',
            x_label='Total Steps',
            filename=f"{env_name}_{group_name}_reward_vs_steps.png",
            x_limit=1000000,
            x_formatter=formatter
        )

        # Reward vs Episode (4000)
        # plot_group(
        #     data=env_ep_data,
        #     x_col='episode',
        #     y_col='mean_smooth',
        #     err_col='std_smooth',
        #     filename=f"{env_name}_{group_name}_reward_vs_episode_4000.png",
        #     x_label='Episode',
        #     x_limit=4000
        # )

        # Reward vs Episode (2000)
        # plot_group(
        #     data=env_ep_data,
        #     x_col='episode',
        #     y_col='mean_smooth',
        #     err_col='std_smooth',
        #     filename=f"{env_name}_{group_name}_reward_vs_episode_2000.png",
        #     x_label='Episode',
        #     x_limit=2000
        # )

print("\nPlots saved to ./plots.")