# Replay Action Usage - Design Decision

## Date: 2026-04-09

## Decision: Use Replay States Only, Not Actions

After code review of `reply-training` and Necto's implementation:

### What Replay Data Provides
- `to_rlgym_dfs()` returns `(gamestate_df, controls)` - state AND actions
- Necto's `NectoStateSetter` only uses **state** component
- Actions in replay are for potential BC/pre-training but NOT used by Necto

### Why We Use States Only
1. **Necto validates state-only approach** - Pure RL + replay states works
2. **Action space mismatch** - Replay actions are continuous 8-dim, ours are 124 discrete
3. **BC can limit exploration** - Imitation can trap policy in local optima
4. **Simplicity** - No extra training machinery needed
5. **Curriculum handles early learning** - Procedural stages compensate for slower initial learning

### Future Exploration: Auxiliary BC Loss

**If baseline training underperforms**, consider adding action matching as auxiliary loss:

```python
# NOT IMPLEMENTED - Future experiment

class AuxiliaryBCLoss:
    """
    Add cross-entropy loss between policy action distribution
    and replay player actions.

    Problem: Action space mismatch
    - Replay: continuous [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
    - Ours:   124 discrete action bins

    Options to handle mismatch:
    1. Map replay actions to nearest discrete action
    2. Train action-probability predictor on replay actions
    3. Use continuous actions during pre-training only
    """

    def compute_loss(self, policy_action_logits, replay_actions):
        # replay_actions: continuous 8-dim
        # Map to discrete bins
        replay_discrete = map_to_discrete_actions(replay_actions)

        # Cross entropy between policy output and replay
        return F.cross_entropy(policy_action_logits, replay_discrete)
```

### When to Consider This
- If discrete-action training plateaus early
- If bot lacks "GC-level" micro-control techniques
- If we switch to continuous/reduced-discrete action space

### Implementation Steps (If Revisited)
1. Create `ReplayActionMapper` to convert continuous → discrete
2. Add `AuxBCLoss` class with proper handling of action mismatch
3. Test with small weighting (0.1-0.2) on top of PPO loss
4. Compare vs state-only baseline

## Current Implementation
- Replay provides **state diversity only**
- Training is pure RL via PPO
- No behavioral cloning or action imitation
