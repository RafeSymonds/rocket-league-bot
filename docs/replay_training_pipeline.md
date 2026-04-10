# Replay-Based Training Pipeline

This document describes how to download Rocket League replays and integrate them into training for diverse, realistic game state resets.

## Overview

Necto's secret sauce is that 70% of training resets come from real match replay data. This provides extreme diversity of game situations that procedural scenarios can't match.

Our replay pipeline:
```
.replay files → carball parsing → parquet → training format → ReplayStateSetter → training
```

## Prerequisites

```bash
pip install carball ballchasing tqdm
```

Get a ballchasing API token from https://ballchasing.com → Settings → API

## Step 1: Download Replays

```bash
python bin/download_replays \
    --api-token YOUR_TOKEN \
    --output data/replays \
    --playlist ranked-duels \
    --count 1000
```

Options:
- `--playlist`: `ranked-duels` (1v1), `ranked-doubles` (2v2), `ranked-standard` (3v3)
- `--count`: Number of replays to download (1000 per playlist recommended)
- `--min-rank`: Minimum rank filter (default: `supersonic-legend` for SSL)

### Rate Limits

ballchasing API is free but rate-limited:
- GC patrons: 16 calls/sec
- All others: 2 calls/sec, 500/hour

The script respects these limits automatically.

## Step 2: Parse Replays

```bash
python bin/parse_replays \
    --input data/replays \
    --output data/replay_arrays
```

This converts carball output (parquet files) into training-ready game state DataFrames.

## Step 3: Integrate with Training

### Option A: Use with Curriculum (Recommended)

```python
from rocket_league_bot_src.mutators_with_replay import DynamicMatchMutatorWithReplay
from rocket_league_bot_src.curriculum import CurriculumManager

curriculum_manager = CurriculumManager()

mutator = DynamicMatchMutatorWithReplay(
    curriculum_manager=curriculum_manager,
    replay_folder="data/replays/ranked-duels",
    replay_reset_probability=0.7,  # 70% like Necto
    use_lazy_loading=True,  # For memory efficiency
)
```

### Option B: Use Standalone

```python
from rocket_league_bot_src.replay_setter import ReplayStateSetter

setter = ReplayStateSetter(
    replay_folder="data/replays/ranked-duels",
    probability=0.7,
)

env = rlgym.make(state_setter=setter, ...)
```

## How It Works

### ReplayStateSetter

Loads parsed replay data and samples game states for resets:

1. **Loading**: Reads all parsed replay folders from the data directory
2. **Episodes**: Each replay is split into ~30s gameplay segments ("episodes")
3. **Sampling**: On each reset, picks a random episode and random frame from it
4. **Application**: Converts the replay frame data into GameState and applies it

### Memory Usage

- `ReplayStateSetter`: Preloads all episodes into memory (fast but memory-intensive)
- `ReplayStateSetterV2`: Lazy loads replays on-demand (memory-efficient for large datasets)

### Integration with Curriculum

`DynamicMatchMutatorWithReplay` combines replay resets with the existing curriculum:

```
Roll random [0, 1]
  ├── < kickoff_prob → KickoffMutator
  ├── < kickoff_prob + replay_prob → ReplayStateSetter (70% of non-kickoff resets)
  └── otherwise → ScenarioResetMutator (procedural scenarios)
```

## Data Format

### Downloaded Structure

```
data/replays/
├── ranked-duels/
│   ├── abc123/
│   │   ├── abc123.replay
│   │   └── parquet/
│   │       ├── __ball.parquet
│   │       ├── __game.parquet
│   │       ├── __metadata.json
│   │       ├── __analyzer.json
│   │       └── player_{uid}.parquet
│   └── ...
```

### Parsed Structure

```
data/replay_arrays/
├── _metadata.json
└── ...episodes stored as numpy arrays
```

## Troubleshooting

### "carball not found"

```bash
pip install carball
```

### "ballchasing API error"

- Check your API token is correct
- Wait for rate limit cool-down (the script handles this automatically)
- Try reducing `--count` if hitting limits

### "No replay episodes found"

- Verify replay folders have correct structure (`__ball.parquet`, etc.)
- Check carball parsing succeeded (look for `carball.o.log` files)
- Some replays may be corrupted or in wrong format - this is normal

### Memory issues

Use `use_lazy_loading=True` in `DynamicMatchMutatorWithReplay` to enable `ReplayStateSetterV2`.

## How Many Replays?

Necto used ~3000 replays total (1000 per playlist) to achieve good diversity.

Recommended:
- **Minimum**: 500 replays
- **Good**: 1000 replays  
- **Optimal**: 3000 replays (1000 × 3 playlists)

Each replay produces ~5-20 usable 30s episodes after filtering, so 1000 replays ≈ 5000-20000 training episodes.

## Next Steps

After setting up replay data:
1. Test the pipeline with a short training run
2. Monitor that replay resets are happening (check logs)
3. Consider tuning `replay_reset_probability` (0.6-0.8 recommended)
4. Evaluate bot performance vs baseline

See `docs/necto_beat_design.md` for the full improvement plan including:
- Discrete action space (biggest impact)
- EARL attention network
- PPO hyperparameter tuning
- Enhanced rewards
