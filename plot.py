import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing import event_accumulator


def get_data(runs_dir):
    all_data = []
    size_guidance = {event_accumulator.SCALARS: 0}
    
    folders = glob.glob(os.path.join(runs_dir, "*_*_*"))
    
    for folder in folders:
        folder_name = os.path.basename(folder)
        parts = folder_name.split("_")
        env_name = parts[1]
        if len(parts) < 3: continue
            
        model_name = "_".join(parts[2:])
        event_files = glob.glob(os.path.join(folder, "events.out.tfevents.*"))
        
        for f in event_files:
            try:
                ea = event_accumulator.EventAccumulator(f, size_guidance=size_guidance)
                ea.Reload()
                tags = ea.Tags()['scalars']
            except:
                continue

            if "episodic_rewards_total_steps" in tags:
                reward_tag = "episodic_rewards_total_steps"
            else:
                reward_tag = next((t for t in tags if 'episodic_return' in t or 'reward' in t.lower()), None)
            
            print(f"Found tag: '{reward_tag}' in {model_name}")
            events = ea.Scalars(reward_tag)
            for i, e in enumerate(events):
                all_data.append({
                    'total_steps': e.step,
                    'episode': i,
                    'episodic_reward': e.value,
                    'model_name': model_name
                })
    return pd.DataFrame(all_data), env_name

df, env_name = get_data('runs')


print("\n")
print("Average Reward per Episode:")
print("="*50)

# Calculate sum and count
stats = df.groupby('model_name')['episodic_reward'].agg(['sum', 'count'])
stats['average_reward'] = stats['sum'] / stats['count']
stats = stats.rename(columns={'count': '#episode'})

baseline_reward = stats.loc['Classical', 'average_reward']
stats['%-growth_val'] = ((stats['average_reward'] - baseline_reward) / baseline_reward) * 100
stats['%-growth'] = stats['%-growth_val'].map(lambda x: f"{x:.2f}%")

# Print formatted results
print(stats[['#episode', 'average_reward', '%-growth']].to_string(float_format="{:.2f}".format))
print("="*50 + "\n")

sns.set_theme(style="whitegrid")
# Traling window
df_episode = df.groupby(['episode', 'model_name'])['episodic_reward'].mean().reset_index()
df_episode['episodic_reward_smooth'] = df_episode.groupby('model_name')['episodic_reward'].transform(
    lambda x: x.rolling(window=11, center=True, min_periods=1).mean()
)

# Traling window
df['total_steps_binned'] = (df['total_steps'] // 1000) * 1000
df_step = df.groupby(['total_steps_binned', 'model_name'])['episodic_reward'].mean().reset_index()
df_step['episodic_reward_smooth'] = df_step.groupby('model_name')['episodic_reward'].transform(
    lambda x: x.rolling(window=21, center=True, min_periods=1).mean()
)

# --- Reward vs Total Steps ---
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=df_step, x='total_steps_binned', y='episodic_reward_smooth', hue='model_name', ax=ax)

# Axis Formatting
ax.xaxis.set_major_locator(ticker.FixedLocator([0, 200000, 400000, 600000, 800000, 1000000]))
formatter = ticker.ScalarFormatter(useMathText=False)
formatter.set_scientific(True)
formatter.set_powerlimits((6, 6))
ax.xaxis.set_major_formatter(formatter)

plt.title(f"{env_name}")
plt.xlabel('Total Steps')
plt.ylabel('Episodic Reward')
plt.xlim(0, 1000000)
plt.ylim(0, 500)
plt.legend(loc='best')
plt.savefig(os.path.join('plots', 'reward_vs_steps.png'), dpi=600, bbox_inches='tight')
plt.close()

# --- Reward vs Episode ---
plt.figure(figsize=(18, 6))
sns.lineplot(data=df_episode, x='episode', y='episodic_reward_smooth', hue='model_name')

plt.title(f"{env_name}")
plt.xlabel('Episode')
plt.ylabel('Episodic Reward')
plt.xlim(0, 5000)
plt.ylim(0, 500)
plt.legend(loc='best')
plt.savefig(os.path.join('plots', 'reward_vs_episode_first5000.png'), dpi=600, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_episode, x='episode', y='episodic_reward_smooth', hue='model_name')

plt.title(f"{env_name}")
plt.xlabel('Episode')
plt.ylabel('Episodic Reward')
plt.xlim(0, 2000)
plt.ylim(0, 500)
plt.legend(loc='best')
plt.savefig(os.path.join('plots', 'reward_vs_episode_first2000.png'), dpi=600, bbox_inches='tight')
plt.close()

print("Plots saved to ./plots.")
