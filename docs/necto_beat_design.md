# Necto-Beat Training Plan: Design Document

## Overview

This document outlines a comprehensive plan to close the gap between our training pipeline and Necto's, with the goal of building a bot that can consistently beat Necto. The improvements are organized by impact priority and dependency order.

---

## Current Training Architecture

### Stage Progression (12 stages)

```
CONTACT → DRIBBLE → SHOOT → AERIAL_CONTACT → AERIAL_SHOOT → SHOOT_CONTESTED
    → SHADOW_DEFEND → DEFEND → DEFEND_CLEAR → POSITIONAL_DUEL → DUEL → SELF_PLAY
```

Each stage advances when performance metrics meet thresholds (or after max iterations as rescue).

### Stage Configuration

| Stage | Blue | Orange | Reset Type | Timeout | Key Rewards |
|-------|------|--------|------------|---------|-------------|
| **CONTACT** | 1 | 0 | Contact reset (ball near car) | 12s | touch=4.5, speed_to_ball=0.55 |
| **DRIBBLE** | 1 | 0 | Dribble reset | 24s | goal=6.0, ball_speed_to_goal=0.8 |
| **SHOOT** | 1 | 0 | Shoot open reset | 30s | goal=10.0, hard_hit=0.30 |
| **AERIAL_CONTACT** | 1 | 0 | Aerial contact reset | 24s | aerial_touch=1.7, goal=8.0 |
| **AERIAL_SHOOT** | 1 | 0 | Aerial shoot reset | 28s | aerial_touch=1.9, goal=12.0 |
| **SHOOT_CONTESTED** | 1 | 1 | Contested shoot reset | 26s | goal=12.0, save_clear=0.15 |
| **SHADOW_DEFEND** | 1 | 1 | Shadow defend / neutral | 22s | save_clear=1.1, goal_side=0.42 |
| **DEFEND** | 1 | 1 | Defense reset | 20s | save_clear=1.55, goal=6.0 |
| **DEFEND_CLEAR** | 1 | 1 | Defend clear reset | 22s | save_clear=1.30, attack_pressure=0.08 |
| **POSITIONAL_DUEL** | 1 | 1 | Positional duel / neutral | 22s | behind_ball=0.20, goal=14.0 |
| **DUEL** | 1 | 1 | Duel reset / neutral | 20s | goal=22.0, attack_pressure=0.22 |
| **SELF_PLAY** | 1 | 1 | Full match (GameMutator) | 35s | goal=26.0, sparse shaping |

### Reset Probabilities per Stage

Stages use `kickoff_reset_prob`, `neutral_reset_prob`, `attack_reset_prob` to choose reset type:
- `kickoff_reset_prob`: KickoffMutator (kickoff positions)
- `neutral_reset_prob`: Neutral self-play reset (random positions)
- `attack_reset_prob`: Attack scenario reset (ball toward goal)

### Current Gap vs Necto

| Aspect | Ours | Necto |
|--------|------|-------|
| Reset diversity | Procedural only | 70% replay, 30% procedural |
| Stages | 12 explicit stages | Implicit (replay provides diversity) |
| Network | MLP (512,512,256) | EARL Perceiver attention |
| Actions | Continuous 8-dim | Discrete 124-action |
| PPO epochs | 3 | 30 |
| Entropy coef | 0.003 | 0.01 |

---

## 1. Discrete Action Space

**Status:** Critical Improvement  
**Priority:** P0 (do first)

### Why This Matters

Necto uses a discretized 124-action space that consolidates continuous inputs into meaningful behavioral clusters:

- **Ground actions** (54 combos): `throttle ∈ {-1, 0, 1}` × `steer ∈ {-1, 0, 1}` × `boost ∈ {0, 1}` × `handbrake ∈ {0, 1}`, with invalid combos filtered (e.g., boost without throttle)
- **Aerial actions** (70 combos): `pitch ∈ {-1, 0, 1}` × `yaw ∈ {-1, 0, 1}` × `roll ∈ {-1, 0, 1}` × `jump ∈ {0, 1}` × `boost ∈ {0, 1}`, with constraints to avoid duplicates and enable wavedashes

Our bot uses the continuous `DefaultAction` parser from rlgym, which produces raw 8-dimensional continuous values. This makes exploration vastly harder and prevents the policy from learning precise action consequences.

### Implementation

```python
# File: rocket_league_bot_src/action_parser.py

class DiscreteNectoAction(ActionParser):
    GROUND_ACTIONS = []
    AERIAL_ACTIONS = []

    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    @staticmethod
    def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:
                                continue
                            if pitch == roll == jump == 0:
                                continue
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        return np.array(actions)

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions, state: GameState) -> np.ndarray:
        # Same as Necto's implementation
        ...
```

### Expected Impact

- Faster convergence due to structured action space
- Better learned behaviors (no more "mashing" continuous inputs)
- Easier policy optimization with discrete categories

---

## 2. Network Architecture: EARL Perceiver

**Status:** Critical Improvement  
**Priority:** P0 (do second, after discrete actions)

### Why This Matters

Necto's `EARLPerceiver` is a transformer-style architecture that:

1. Takes a **query** (player-specific state, 36 dims), **key-value** pairs for all entities (ball + 34 boosts + 2 players = 41 entities × 25 dims), and a **mask**
2. Uses attention to aggregate information from all entities
3. Produces a rich 256-dim embedding per player

Our simple MLP `(512, 512, 256)` cannot reason about relationships between the ball, boosts, and opponents effectively. The attention mechanism in EARL is what enables Necto to "understand" complex game situations.

### Architecture Details (from Necto)

```
EARLPerceiver(
    embedding_dim=256,
    num_heads=4,
    num_layers=8,
    num_queries=1,
    query_features=36,    # player query vector size
    key_value_features=25 + 30  # entity features
)

Actor: DiscretePolicy(
    Necto(EARLPerceiver, ControlsPredictorDot(256)),
    split=(90,)  # num_discrete_actions
)

Critic: Necto(EARLPerceiver, Linear(256, 1))
```

### Implementation Options

**Option A (Preferred): Port EARL Directly**
- Copy `earl_pytorch` if compatible, or reimplement EARL attention layer
- Requires `OBS_DIM` restructuring to match Necto's batched observation format

**Option B: Actor-Critic with Attention Overlay**
- Keep current obs format but add a learned attention mechanism over entities
- Use multi-head attention after initial MLP encoding

**Option C: Larger Shared Body**
- Increase MLP to `(1024, 1024, 512)` with entity pooling
- Less powerful but easier to implement

### Dependency

This change requires restructuring `obs.py` to produce batched entity observations. See Section 3.

---

## 3. Observation Feature Parity

**Status:** Critical Improvement  
**Priority:** P0 (do third)

### Current Observation (54 dims, ours)

```
forward (3) + up (3) + vel (3) + ang_vel (3) + boost (1) + on_ground (1)
+ is_supersonic (1) + has_jumped (1) + has_double_jumped (1) + is_demoed (1)
+ rel_ball_pos (3) + rel_ball_vel (3) + ball_ang_vel (3) + to_ball_dir (3)
+ to_ball_dist (1) + ball_speed (1) + ball_height (1) + speed_toward_ball (1)
+ cos_forward_to_ball (1) + cos_ball_to_goal (1)
+ to_my_goal_dir (3) + to_enemy_goal_dir (3) + to_my_goal_dist (1) + to_enemy_goal_dist (1)
+ opp_rel_pos (3) + opp_rel_vel (3) + to_opp_dir (3) + to_opp_dist (1)
```

### Necto's Observation Structure

Necto uses a **batched architecture** with three output tensors per player:

```
q: (1, 1, 25 + 8 + 3)    # player query + actions + goal_diff/time/overtime
kv: (entities, 25 + 30)  # ball + boosts + players with relative info
m: (entities,)           # mask for attention
```

Entity features (25 dims):
```
IS_SELF, IS_MATE, IS_OPP, IS_BALL, IS_BOOST (5)
POS (3) + LIN_VEL (3) + FW (3) + UP (3) + ANG_VEL (3) (15)
BOOST, DEMO, ON_GROUND, HAS_FLIP, HAS_JUMP (5)
```

Actions (8 dims) embedded into query. Scoreboard (goal_diff, time_left, is_overtime) included.

### Missing Features We Need to Add

1. **Boost pad locations and timers** - Critical for boost management
2. **Demo timers** - Demo state tracking
3. **Previous actions** - Temporal context for action consequences
4. **All opponent positions** - Not just closest
5. **Scoreboard info** - Game time, goal differential, overtime state

### Restructured Observation Design

```python
# New OBS_DIM target: Variable due to entity-based approach
# For EARL compatibility: query=36, kv=25+30, entities=41

# Player query (36 dims):
# - Car state: pos(3), lin_vel(3), ang_vel(3), forward(3), up(3), boost(1), on_ground(1), 
#             has_flip(1), has_jump(1), is_demoed(1)
# - Ball relative: pos(3), lin_vel(3)
# - Goal info: goal_diff(1), time_left(1), is_overtime(1)
# - Actions: prev_action(8) embedded

# Entity key-value (25 dims per entity):
# - Type flags (5): IS_SELF, IS_MATE, IS_OPP, IS_BALL, IS_BOOST
# - Position (3)
# - Linear velocity (3)
# - Forward (3)
# - Up (3)
# - Angular velocity (3)
# - Boost amount / demo timer (1)
# - State flags (4): ON_GROUND, HAS_FLIP, HAS_JUMP, (reserved)

# Entities:
# - Ball (1)
# - Boost pads (34)
# - Other player (1)
```

### File Changes

- `rocket_league_bot_src/obs.py`: Complete rewrite for EARL-compatible format
- `rocket_league_bot_src/config.py`: Update `OBS_DIM` to match new format
- `BotBoi_v1/src/bot.py`: Update action lookup table generation

---

## 4. PPO Hyperparameter Tuning

**Status:** High Impact  
**Priority:** P1

### Current vs Necto's Parameters

| Parameter | Ours | Necto | Recommendation |
|-----------|------|-------|-----------------|
| batch_size | 50,000 | 100,000 | Increase to 100k |
| minibatch_size | 10,000 | 10,000 | Keep |
| epochs | 3 | 30 | Increase to 20-30 |
| gamma | 0.995 | 0.995 | Keep |
| ent_coef | 0.003 | 0.01 | Increase to 0.01 |
| policy_lr | 2.5e-4 | 1e-4 | Decrease to 1e-4 |
| critic_lr | 2.5e-4 | 1e-4 | Decrease to 1e-4 |

### Rationale

- **More epochs**: With discrete actions and more complex observations, the policy needs more SGD passes to converge
- **Higher entropy**: Prevents early collapse to suboptimal deterministic policies
- **Lower learning rate**: More stable learning with larger batches and deeper networks
- **Larger batch**: More stable gradients, better for complex policies

### Implementation

Update defaults in `train.py`:
```python
parser.add_argument("--ppo-batch-size", type=int, default=100_000)
parser.add_argument("--ppo-epochs", type=int, default=25)
parser.add_argument("--ent-coef", type=float, default=0.01)
parser.add_argument("--policy-lr", type=float, default=1e-4)
parser.add_argument("--critic-lr", type=float, default=1e-4)
```

---

## 5. Reward Function Enhancement

**Status:** High Impact  
**Priority:** P1

### Necto's Reward Structure

Necto uses a sophisticated reward function with:

1. **State Quality**: Estimates game advantage using:
   - Goal distance weighted by team (exponential decay)
   - Player-ball alignment vs goal vectors
   - Win probability from scoreboard (`win_prob`)

2. **Player Quality**: Per-player positioning advantage

3. **Event Rewards**:
   - Touch height bonus (encourages aerial play)
   - Touch acceleration bonus (ball hitting reward)
   - Flip reset bonus (for flip resets)
   - Boost gain/loss
   - Demo rewards
   - Goal distance/speed bonuses at scoring

4. **Team Spirit**: 0.6 weighting for team coordination

### Our Current Rewards

We have a rich set of shaping rewards but lack:
- Win probability signal
- State/player quality estimates
- Sophisticated touch height/acceleration rewards

### Enhancement Design

```python
class NectoStyleRewardFunction(RewardFunction):
    def __init__(self, team_spirit=0.6, ...):
        # Keep existing reward components
        self.state_quality = None
        self.player_qualities = None
        self.team_spirit = team_spirit
        ...

    def _state_qualities(self, state: GameState):
        # Calculate state advantage estimate
        # Blue positive = blue winning
        # Orange positive = orange winning
        ball_pos = state.ball.position
        state_quality = 0.5 * goal_dist_weight * (
            exp(-dist(ORANGE_GOAL, ball_pos) / CAR_MAX_SPEED)
            - exp(-dist(BLUE_GOAL, ball_pos) / CAR_MAX_SPEED)
        )
        # Add win probability from scoreboard
        ...

    def get_reward(self, ...):
        # Combine existing rewards with Necto-style state quality
        # Use team spirit for coordination
        ...
```

### Key Additions

1. **Win Probability Reward**: Use scoreboard to add reward proportional to estimated win chance
2. **State Quality Delta**: Reward for improving positional advantage
3. **Enhanced Touch Rewards**: Height-based and acceleration-based bonuses
4. **Team Coordination**: Add team spirit factor for shared rewards

---

## 6. State Setter / Reset Diversity

**Status:** Medium Impact  
**Priority:** P1

### Necto's Approach

Necto uses weighted replay-based resets:
- 70% from real match replay data
- 8% purely random
- 4% kickoff-like
- 4% kickoff-symmetric
- 5% goalie practice
- 4% hoops
- 5% wall practice

The replay setter samples from actual game situations, providing extremely diverse and realistic training scenarios.

### Our Current Approach

Purely procedural scenarios:
- Contact reset (ball near car, no opponent)
- Dribble reset (car with ball, forward motion)
- Shoot open (attacking scenario, no defender)
- Aerial resets (various heights)
- Defense resets (ball threatening goal)
- Shadow defend
- Duel resets
- Positional duel

### Implementation: Replay-Based Resets

We have a full replay parsing pipeline:

```
.replay file → carball.analyze_replay_file() → parquet → to_rlgym_dfs() → ReplayStateSetter
```

**Scripts created:**
- `bin/download_replays` - Downloads SSL replays via ballchasing API
- `bin/parse_replays` - Converts parsed replays to training format
- `rocket_league_bot_src/replay_setter.py` - ReplayStateSetter for training
- `rocket_league_bot_src/mutators_with_replay.py` - DynamicMatchMutatorWithReplay

**Usage:**
```bash
# 1. Create the replay env once
bin/setup_replay_env

# 2. Download replays (need ballchasing API token from ballchasing.com)
python bin/download_replays --api-token YOUR_TOKEN --output data/replays --count 1000

# 3. Parse replays to training format
python bin/parse_replays --input data/replays --output data/replay_arrays

# 4. Integrate with training
```

**Integration with Curriculum:**

```python
from rocket_league_bot_src.mutators_with_replay import DynamicMatchMutatorWithReplay

mutator = DynamicMatchMutatorWithReplay(
    curriculum_manager=curriculum_manager,
    replay_folder="data/replays/ranked-duels",
    replay_reset_probability=0.7,  # 70% like Necto
    use_lazy_loading=True,  # For memory efficiency
)
```

**Reset Flow with Replay Integration:**

```
Roll random [0, 1]
  ├── < kickoff_prob → KickoffMutator (unchanged)
  ├── < kickoff_prob + replay_prob (0.70) → ReplayStateSetter
  └── otherwise → ScenarioResetMutator (procedural scenarios)
```

**ReplayStateSetter Options:**
- `ReplayStateSetter`: Preloads all episodes (fast but memory-intensive)
- `ReplayStateSetterV2`: Lazy loads replays on-demand (memory-efficient)

**ballchasing API:**
- Free for public replays
- Rate limits: GC patrons 16 calls/sec, everyone else 2 calls/sec, 500/hour
- Get token at https://ballchasing.com → Settings → API

**How Many Replays:**
- Minimum: 500 replays
- Good: 1000 replays per playlist
- Optimal: 3000 replays (1000 × 3 playlists: ranked-duels, ranked-doubles, ranked-standard)

Each replay produces ~5-20 usable 30s episodes, so 1000 replays ≈ 5000-20000 training episodes.

### Enhancement: BetterRandom

Necto's `BetterRandom` uses triangular distributions for more realistic placement:

```python
class BetterRandom(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
        # Ball position with triangular z (more realistic ground distribution)
        state_wrapper.ball.set_pos(
            x=np.random.uniform(-LIM_X, LIM_X),
            y=np.random.uniform(-LIM_Y, LIM_Y),
            z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
        )
        # Exponential ball speed (tail matches real velocity distribution)
        ball_speed = np.random.exponential(-BALL_MAX_SPEED / np.log(1 - 0.999))
        ...
```

---

## 7. Full-Match Training

**Status:** Medium Impact  
**Priority:** P2

### Current

Full-match training via `rlgym-tools` `GameMutator` when `SELF_PLAY` stage is reached.

### Necto's Approach

Necto trains on full 1v1 matches throughout, using the replay setter as the primary reset mechanism. They don't use explicit curriculum stages - instead, all skills are learned simultaneously with replay data providing the full diversity.

### Consideration

Once we have discrete actions, EARL, and enhanced rewards, we may want to consider:
1. A simpler curriculum with fewer stages
2. More reliance on replay-based resets
3. Earlier introduction of full-match scenarios

---

## Implementation Order

| Phase | Change | Status | Files to Modify | Effort |
|-------|--------|--------|-----------------|--------|
| 1 | Discrete action parser | **Done** | New `action_parser.py`, update `env.py` | Low |
| 2 | Update obs.py for entity format | **Pending** | `obs.py`, `config.py` | Medium |
| 3 | Implement EARL/attention network | **Pending** | `train.py`, possibly new network module | High |
| 4 | PPO hyperparameter tuning | **Done** | `train.py` | Low |
| 5 | Enhanced reward function | **Done** | `rewards.py` | Medium |
| 6 | Replay-based state setters | **Done** | `replay_setter.py`, `mutators_with_replay.py`, `bin/download_replays`, `bin/parse_replays` | High |
| 7 | Bot runtime update | **Pending** | `BotBoi_v1/src/bot.py`, `runtime_config.json` | Low |

### Completed Items

- **Discrete action parser**: `rocket_league_bot_src/action_parser.py` - 124 discrete actions matching Necto
- **PPO hyperparameter tuning**: `train.py` - batch 100k, epochs 25, ent_coef 0.01, lr 1e-4
- **Win-prob reward**: `rocket_league_bot_src/rewards.py` - WinProbReward class added to CurriculumReward
- **Replay pipeline**: `bin/download_replays`, `bin/parse_replays`, `rocket_league_bot_src/replay_setter.py`
- **Replay curriculum integration**: `rocket_league_bot_src/mutators_with_replay.py`
- **Documentation**: `docs/replay_training_pipeline.md`, this document updated

### Pending Priority Items

1. **EARL attention network** (future improvement)
   - Requires restructuring obs.py for entity-based format
   - Would enable attention over ball/boosts/opponents
   - Fresh-training boundary - revisit after discrete actions + replay prove out
2. **Bot runtime update** for discrete actions

### New Training Command Options

```bash
# Basic training with new defaults (discrete actions, tuned PPO)
python train.py

# With replay data for 70% replay-based resets
python train.py --replay-folder data/replays/ranked-duels

# Force continuous actions (for testing)
python train.py --use-continuous-actions

# Tuned PPO params (now defaults)
python train.py --ppo-epochs 25 --ent-coef 0.01 --policy-lr 1e-4 --ppo-batch-size 100000
```

---

## Risk Considerations

1. **EARL Implementation**: Requires careful debugging; consider starting with Option C (larger MLP) if EARL proves problematic
2. **Observation Breaking Change**: Any obs change invalidates existing checkpoints; treat as fresh training boundary
3. **Discrete Actions**: Requires careful handling of action distribution in policy head
4. **Reward Changes**: May need tuning of weight parameters

---

## Success Metrics

1. Eval against Necto checkpoint shows >50% win rate
2. Eval against fixed older checkpoint shows consistent improvement over time
3. Training metrics show stable convergence without divergence
4. Bot demonstrates: aerial plays, shadow defense, effective clearing, 1v1 dribble/shot conversion

---

## Related Documentation

- `docs/replay_training_pipeline.md` - Full replay pipeline setup and usage guide
- `docs/necto_beat_design.md` - This document - comprehensive improvement plan
- `rocket_league_bot_src/replay_setter.py` - ReplayStateSetter implementation
- `rocket_league_bot_src/mutators_with_replay.py` - Curriculum integration with replay support
- `rocket_league_bot_src/curriculum.py` - Curriculum stage management
- `rocket_league_bot_src/config.py` - Stage configurations and reward weights
- `bin/download_replays` - Replay download script
- `bin/parse_replays` - Replay parsing script
