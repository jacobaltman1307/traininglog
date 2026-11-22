/** Card-game RL metric presets tailored to CleanRL PPO (`cardgame/agents/ppo.py`). */

export const CARDGAME_PRESETS = [
  {
    id: "overview",
    label: "Overview",
    patterns: [
      "game/win_rate_overall",
      "charts/episodic_return",
      "charts/SPS",
      "losses/policy_loss",
      "losses/value_loss",
      "losses/approx_kl",
      "losses/entropy",
      "losses/explained_variance",
    ],
  },
  {
    id: "winrates",
    label: "Winrates & Seats",
    patterns: [
      "game/win_rate_overall",
      "game/win_rate_p1",
      "game/win_rate_p2",
      "charts/winrate_*",
    ],
  },
  {
    id: "rewards",
    label: "Rewards & Annealing",
    patterns: [
      "charts/dense_reward_scale",
      "rewards/win_loss",
      "rewards/damage",
      "rewards/trades",
      "rewards/mana_waste",
      "rewards/dominance",
      "rewards/kills",
    ],
  },
  {
    id: "entropy_lanes",
    label: "Entropy & Lanes",
    patterns: [
      "losses/entropy",
      "losses/entropy_hand",
      "losses/entropy_board",
      "losses/clipfrac",
      "actions/lane_0_rate",
      "actions/lane_1_rate",
      "actions/lane_2_rate",
      "actions/lane_3_rate",
      "actions/episode_pass_rate",
      "actions/invalid_action_rate",
    ],
  },
  {
    id: "economy",
    label: "Board Economy",
    patterns: [
      "game/artifacts_unlocked_avg",
      "game/artifact_unlock_rate",
      "game/mana_efficiency",
      "game/trade_ratio",
      "game/card_advantage",
      "game/board_power_margin",
      "game/final_hp_margin",
      "game/game_length_rounds",
      "game/damage_dealt",
      "game/cards_played_per_turn",
    ],
  },
  {
    id: "system",
    label: "System & GPU",
    patterns: [
      "sys/gpu_util",
      "sys/gpu_vram_pct",
      "sys/gpu_temp_c",
      "sys/gpu_power_w",
      "sys/cpu_pct",
      "sys/ram_pct",
    ],
  },
];

function matchPattern(metricName, pattern) {
  if (!pattern.includes("*")) return metricName === pattern;
  const escaped = pattern.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*");
  return new RegExp(`^${escaped}$`).test(metricName);
}

export function selectPresetMetrics(availableMetrics, presetId) {
  const preset = CARDGAME_PRESETS.find((p) => p.id === presetId);
  if (!preset) return availableMetrics;
  const picked = [];
  const seen = new Set();
  for (const pat of preset.patterns) {
    for (const m of availableMetrics) {
      if (!seen.has(m) && matchPattern(m, pat)) {
        seen.add(m);
        picked.push(m);
      }
    }
  }
  return picked;
}
