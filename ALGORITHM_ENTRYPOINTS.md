# Algorithm Entrypoints

This project keeps algorithm train/navigation entrypoints centralized in `algorithm_entrypoints.py`.

## Teacher Entry (shared by DQN/PPO/HER)

- Q-learning teacher train: `q_learning.train_q_learning_full_map`

## Train Entrypoints

- Q-learning: `q_learning.train_q_learning_full_map`
- DQN: `dqn.train_dqn_full_map`
- PPO: `ppo.train_ppo_full_map`
- HER: `her.train_her_full_map`

## Navigation Path Entrypoints

- Q-learning: `q_learning.build_q_learning_path`
- DQN: `dqn.build_dqn_path`
- PPO: `ppo.build_ppo_path`
- HER: `her.build_her_path`
