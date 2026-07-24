import os
import glob
import hashlib
import pickle
import argparse
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tensorboard.backend.event_processing import event_accumulator


DEFAULT_RUNS_DIR = "runs"
DEFAULT_OUT_DIR = "plots"

CACHE_DIR = ".plot_cache"
CACHE_VERSION = 1

PLOT_ENVS = [
    "CartPole-v1",
    "Acrobot-v1",
]

PLOT_GROUPS = {
    "Group_1": [
        "Random", "Classical", "Hybrid_FOTP", "Hybrid_FO", "Hybrid_FT", "Hybrid_FP"
    ],
    "Group_2": [
        "Random", "Classical", "Hybrid_F", "Hybrid_O", "Hybrid_T", "Hybrid_P"
    ],
    "More_Options": [
        "Classical", "Classical-3options", "Classical-4options", "Hybrid_P", "Hybrid_P-3options", "Hybrid_P-4options"
    ],
    "F_ablation": [
        "Hybrid_F", "Hybrid_F_fixLam", "Hybrid_F_noEntangle", "Hybrid_F-2layers", "Hybrid_F-3layers", "Hybrid_F-6layers", "Hybrid_F-7layers"
    ],
}

PLOT_MODELS = {model for group in PLOT_GROUPS.values() for model in group}

DYNAMICS_MODELS = {"Hybrid_O", "Classical"}
DYNAMICS_TAGS = ["policy_entropy", "actor_loss", "critic_loss"]

NUM_ACTIONS = {
    "CartPole-v1": 2,
    "Acrobot-v1": 3,
}

SPECIAL_LABELS = {
    "Hybrid_F_fixLam": r"Hybrid-F (fixed $\lambda$)",
    "Hybrid_F_noEntangle": "Hybrid-F (w/o CNOT)",
    "Hybrid_F-2layers": "Hybrid-F (depth 2)",
    "Hybrid_F-3layers": "Hybrid-F (depth 3)",
    "Hybrid_F-6layers": "Hybrid-F (depth 6)",
    "Hybrid_F-7layers": "Hybrid-F (depth 7)",
}

# IEEE column / full-text-width in inches.
COL_WIDTH = 3.5
TEXT_WIDTH = 7.16

ENV_ORDER = ["CartPole-v1", "Acrobot-v1"]
X_LIMIT = 1_000_000
X_TICKS = [0, 200000, 400000, 600000, 800000, 1000000]
STEP_BIN_WIDTH = 1000
SMOOTH_WINDOW = 20


def pretty_label(model_name):
    if model_name in SPECIAL_LABELS:
        return SPECIAL_LABELS[model_name]
    return model_name.replace("_", "-")


def _cache_signature(event_files, want_dynamics):
    sig = [f"v{CACHE_VERSION}", f"dyn={int(want_dynamics)}"]
    for f in sorted(event_files):
        st = os.stat(f)
        sig.append(f"{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.md5("|".join(sig).encode()).hexdigest()


def _read_folder(folder):
    """Extract one run folder's reward and dynamics rows (cached to CACHE_DIR).
    Returns (reward_rows, dyn_rows, env_name, model_name) or None if filtered."""
    folder_name = os.path.basename(folder)
    parts = folder_name.split("_")

    env_name = parts[1]
    model_name = "_".join(parts[2:])

    if env_name not in PLOT_ENVS:
        return None
    if model_name not in PLOT_MODELS:
        return None

    want_dynamics = model_name in DYNAMICS_MODELS
    event_files = glob.glob(os.path.join(folder, "events.out.tfevents.*"))

    cache_path = os.path.join(CACHE_DIR, folder_name + ".pkl")
    try:
        signature = _cache_signature(event_files, want_dynamics)
    except OSError:
        signature = None
    if signature is not None and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                cached = pickle.load(fh)
            if cached.get("signature") == signature:
                return cached["reward_rows"], cached["dyn_rows"], env_name, model_name
        except (pickle.PickleError, EOFError, KeyError):
            pass

    size_guidance = {event_accumulator.SCALARS: 0}
    reward_rows = []
    dyn_rows = []

    for f in event_files:
        ea = event_accumulator.EventAccumulator(f, size_guidance=size_guidance)
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))

        reward_tag = "episodic_rewards_total_steps"
        if reward_tag in available:
            for i, e in enumerate(ea.Scalars(reward_tag)):
                reward_rows.append({
                    'total_steps': e.step,
                    'episode': i,
                    'episodic_reward': e.value,
                    'model_name': model_name,
                    'env_name': env_name,
                    'run_id': folder_name,
                })

        if want_dynamics:
            for tag in DYNAMICS_TAGS:
                if tag not in available:
                    continue
                for e in ea.Scalars(tag):
                    dyn_rows.append({
                        'total_steps': e.step,
                        'metric': tag,
                        'value': e.value,
                        'model_name': model_name,
                        'env_name': env_name,
                        'run_id': folder_name,
                    })

    if signature is not None:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(
                    {"signature": signature, "reward_rows": reward_rows, "dyn_rows": dyn_rows},
                    fh, protocol=pickle.HIGHEST_PROTOCOL
                )
        except OSError:
            pass

    return reward_rows, dyn_rows, env_name, model_name


def get_data(runs_dir, max_workers=None):
    all_data = []
    dyn_data = []
    run_counts = defaultdict(int)

    folders = glob.glob(os.path.join(runs_dir, "*_*_*"))
    print(f"Found {len(folders)} folders in {runs_dir}.")

    if max_workers is None:
        max_workers = min(len(folders), os.cpu_count() or 1)
    max_workers = max(max_workers, 1)

    processed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_read_folder, folder): folder for folder in folders}
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            reward_rows, dyn_rows, env_name, model_name = result
            all_data.extend(reward_rows)
            dyn_data.extend(dyn_rows)
            run_counts[(env_name, model_name)] += 1
            processed += 1
            print(f"Processed ({model_name}, {env_name}) [{processed}]")

    print("\n--- Run Counts ---")
    for (env, model), count in run_counts.items():
        print(f"Env: {env} | Model: {model} | Runs detected: {count}")
    return pd.DataFrame(all_data), pd.DataFrame(dyn_data)


def bin_smooth(df, value_col, group_cols, bin_col='total_steps_binned', window=SMOOTH_WINDOW):
    """Bin -> mean/std across runs -> rolling smooth, per group."""
    stats = df.groupby(group_cols + [bin_col])[value_col].agg(['mean', 'std']).reset_index()
    stats['std'] = stats['std'].fillna(0)
    stats = stats.sort_values(by=group_cols + [bin_col])
    stats['mean_smooth'] = stats.groupby(group_cols)['mean'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    stats['std_smooth'] = stats.groupby(group_cols)['std'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean()
    )
    return stats


def compute_stats(df, df_dyn):
    df['total_steps_binned'] = (df['total_steps'] // STEP_BIN_WIDTH) * STEP_BIN_WIDTH
    step_stats = bin_smooth(df, 'episodic_reward', ['model_name', 'env_name'])

    if not df_dyn.empty:
        df_dyn['total_steps_binned'] = (df_dyn['total_steps'] // STEP_BIN_WIDTH) * STEP_BIN_WIDTH
        dyn_stats = bin_smooth(df_dyn, 'value', ['model_name', 'env_name', 'metric'])
    else:
        dyn_stats = df_dyn

    return step_stats, dyn_stats


def _step_formatter():
    fmt = ticker.ScalarFormatter(useMathText=False)
    fmt.set_scientific(True)
    fmt.set_powerlimits((6, 6))
    return fmt


def env_ylim(env_name):
    if env_name == "CartPole-v1":
        return (0, 500)
    if env_name == "Acrobot-v1":
        return (-500, 0)
    return (None, None)


def apply_step_axis(ax):
    ax.xaxis.set_major_locator(ticker.FixedLocator(X_TICKS))
    ax.xaxis.set_major_formatter(_step_formatter())
    ax.xaxis.get_offset_text().set_fontsize(7)


def print_group_stats(df, env_name, group_name, model_list):
    """Print per-model avg episodes, avg reward, and reward relative to Classical."""
    group_raw_data = df[(df['env_name'] == env_name) & (df['model_name'].isin(model_list))]
    if group_raw_data.empty:
        return

    print(f"Average Reward per Episode: {env_name} [{group_name}]")
    print("=" * 50)

    episodes_per_run = group_raw_data.groupby(['model_name', 'run_id']).size()
    ep_stats = episodes_per_run.groupby('model_name').agg(['mean', 'std'])

    stats = group_raw_data.groupby('model_name')['episodic_reward'].agg(['sum', 'count'])
    stats['avg_episodes'] = ep_stats['mean']
    stats['episodes_std'] = ep_stats['std']
    stats['avg_reward'] = stats['sum'] / stats['count']

    if 'Classical' in stats.index:
        baseline_reward = stats.loc['Classical', 'avg_reward']
        stats['rel_reward'] = abs(stats['avg_reward']) / abs(baseline_reward)
        stats['rel_reward'] = stats['rel_reward'].map(lambda x: f"{x:.2f}x")
    else:
        stats['rel_reward'] = "N/A"

    print(stats[['avg_episodes', 'episodes_std', 'avg_reward', 'rel_reward']].to_string(float_format="{:.2f}".format))
    print("=" * 50 + "\n")


def draw_reward_curves(ax, env_name, model_list, step_stats, palette):
    env_step_data = step_stats[step_stats['env_name'] == env_name]
    for model in model_list:
        subset = env_step_data[env_step_data['model_name'] == model]
        if subset.empty:
            continue
        color = palette[model_list.index(model) % len(palette)]
        ax.plot(
            subset['total_steps_binned'], subset['mean_smooth'],
            label=pretty_label(model), color=color, linewidth=1.0
        )
        ax.fill_between(
            subset['total_steps_binned'],
            subset['mean_smooth'] - subset['std_smooth'],
            subset['mean_smooth'] + subset['std_smooth'],
            color=color, alpha=0.25, linewidth=0
        )
    ax.set_xlim(0, X_LIMIT)
    ax.set_ylim(*env_ylim(env_name))
    ax.tick_params(axis='both', which='major', labelsize=7, pad=0)
    ax.legend(loc='best', fontsize=5, framealpha=0.8)


def _save_figure(out_dir, filename, pad=0.3):
    plt.tight_layout(pad=pad)
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    return path


def plot_groups_grid(df, step_stats, palette, row_groups, filename, out_dir):
    """One group per row, one env per column; env names title the top row only."""
    _, axes = plt.subplots(
        len(row_groups), len(ENV_ORDER),
        figsize=(TEXT_WIDTH, 1.89 * len(row_groups)), sharex=True
    )
    for r, group_name in enumerate(row_groups):
        model_list = PLOT_GROUPS[group_name]
        for c, env_name in enumerate(ENV_ORDER):
            ax = axes[r][c]
            print_group_stats(df, env_name, group_name, model_list)
            draw_reward_curves(ax, env_name, model_list, step_stats, palette)
            if r == 0:
                ax.set_title(env_name, fontsize=10)
    for c in range(len(ENV_ORDER)):
        apply_step_axis(axes[-1][c])

    path = _save_figure(out_dir, filename)
    print(f"Saved grid figure: {path}")


def plot_group_stacked(df, step_stats, palette, group_name, filename, out_dir):
    """Single group, two envs stacked vertically (CartPole top, Acrobot bottom)."""
    model_list = PLOT_GROUPS[group_name]
    _, axes = plt.subplots(
        len(ENV_ORDER), 1, figsize=(COL_WIDTH, 1.89 * len(ENV_ORDER)), sharex=True
    )
    for r, env_name in enumerate(ENV_ORDER):
        ax = axes[r]
        print_group_stats(df, env_name, group_name, model_list)
        draw_reward_curves(ax, env_name, model_list, step_stats, palette)
        ax.set_title(env_name, fontsize=10)
    apply_step_axis(axes[-1])

    path = _save_figure(out_dir, filename)
    print(f"Saved stacked figure: {path}")


def plot_option_value_bottleneck(dyn_stats, palette, out_dir, filename="analysis_option_value.png"):
    """Policy entropy / actor loss / critic loss vs steps for Hybrid-O and
    Classical on both envs, with the ln|A| entropy ceiling."""
    if dyn_stats is None or dyn_stats.empty:
        print("No training-dynamics data found; skipping bottleneck figure.")
        return

    series = [
        ("Hybrid_O", "CartPole-v1", palette[0]),
        ("Hybrid_O", "Acrobot-v1", palette[1]),
        ("Classical", "CartPole-v1", palette[2]),
        ("Classical", "Acrobot-v1", palette[3]),
    ]

    panels = [
        ("policy_entropy", "Policy entropy", (0.0, 1.25)),
        ("actor_loss", "Actor Loss", (-4, 4)),
        ("critic_loss", "Critic Loss", (-30, 30)),
    ]

    _, axes = plt.subplots(len(panels), 1, figsize=(COL_WIDTH, 3.6), sharex=True)

    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        for model, env_name, color in series:
            subset = dyn_stats[
                (dyn_stats['model_name'] == model)
                & (dyn_stats['env_name'] == env_name)
                & (dyn_stats['metric'] == metric)
            ]
            if subset.empty:
                continue
            env_short = env_name.replace("-v1", "")
            ax.plot(
                subset['total_steps_binned'],
                subset['mean_smooth'],
                label=f"{pretty_label(model)} ({env_short})",
                color=color,
                linewidth=1.0
            )
            ax.fill_between(
                subset['total_steps_binned'],
                subset['mean_smooth'] - subset['std_smooth'],
                subset['mean_smooth'] + subset['std_smooth'],
                color=color,
                alpha=0.2,
                linewidth=0
            )

        if metric == "policy_entropy":
            for env_name, color in (("CartPole-v1", palette[0]), ("Acrobot-v1", palette[1])):
                ln_a = np.log(NUM_ACTIONS[env_name])
                ax.axhline(ln_a, color=color, linestyle=':', linewidth=1.0, alpha=0.9)
                ax.text(
                    X_LIMIT, ln_a, r" $\ln(|A|)$",
                    fontsize=5, color=color, va='bottom', ha='right'
                )

        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, X_LIMIT)
        ax.tick_params(axis='both', which='major', labelsize=7, pad=0)
        ax.legend(loc='best', fontsize=4.5, framealpha=0.8, ncol=1)

    apply_step_axis(axes[-1])

    path = _save_figure(out_dir, filename, pad=0.2)
    print(f"Saved option-value bottleneck figure: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the reward and training-dynamics figures from tensorboard runs."
    )
    parser.add_argument('--runs-dir', default=DEFAULT_RUNS_DIR)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--workers', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    df, df_dyn = get_data(args.runs_dir, max_workers=args.workers)
    step_stats, dyn_stats = compute_stats(df, df_dyn)

    os.makedirs(args.out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("Set1")

    plot_groups_grid(
        df, step_stats, palette,
        ["Group_1", "Group_2"], "Group_1_Group_2_reward_vs_steps.png", args.out_dir
    )
    plot_group_stacked(df, step_stats, palette, "More_Options", "More_Options_reward_vs_steps.png", args.out_dir)
    plot_group_stacked(df, step_stats, palette, "F_ablation", "F_ablation_reward_vs_steps.png", args.out_dir)
    plot_option_value_bottleneck(dyn_stats, palette, args.out_dir)

    print(f"\nPlots saved to ./{args.out_dir}.")


if __name__ == "__main__":
    main()
