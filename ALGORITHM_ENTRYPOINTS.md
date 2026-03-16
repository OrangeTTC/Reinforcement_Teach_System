# Algorithm Entrypoints

This project keeps algorithm train/navigation entrypoints centralized in `algorithm/algorithm_entrypoints.py`.
Algorithm kernel implementations now live in dedicated files:

- `algorithm/algorithm_a_star.py`
- `algorithm/algorithm_q_learning.py`
- `algorithm/algorithm_dqn.py`
- `algorithm/algorithm_ppo.py`
- `algorithm/algorithm_her.py`

## Teacher Entry (shared by DQN/PPO/HER)

- Q-learning teacher train: `algorithm.algorithm_q_learning.train_q_learning_full_map`

## Train Entrypoints

- Q-learning: `algorithm.algorithm_q_learning.train_q_learning_full_map`
- DQN: `algorithm.algorithm_dqn.train_dqn_full_map`
- PPO: `algorithm.algorithm_ppo.train_ppo_full_map`
- HER: `algorithm.algorithm_her.train_her_full_map`

## Navigation Path Entrypoints

- Q-learning: `algorithm.algorithm_q_learning.build_q_learning_path`
- DQN: `algorithm.algorithm_dqn.build_dqn_path`
- PPO: `algorithm.algorithm_ppo.build_ppo_path`
- HER: `algorithm.algorithm_her.build_her_path`
