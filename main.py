import gymnasium as gym
import numpy as np
import random


def choose_action(state, q_table, epsilon, action_size):
    """
    根据 epsilon-greedy 策略选择动作
    """
    # TODO 1:
    # 如果随机数小于 epsilon，就随机探索
    # 否则选择当前状态下 Q 值最大的动作
    pass


def train_q_learning(
    env,
    episodes=500,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
):
    """
    训练 Q-learning
    """
    state_size = env.observation_space.n
    action_size = env.action_space.n

    # Q 表：行是状态，列是动作
    q_table = np.zeros((state_size, action_size))

    # 记录每一轮总奖励
    episode_rewards = []

    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # 1. 选择动作
            action = choose_action(state, q_table, epsilon, action_size)

            # 2. 与环境交互
            next_state, reward, terminated, truncated, info = env.step(action)

            # 3. 计算 Q-learning 的更新目标
            # TODO 2:
            # 如果当前回合结束，则 td_target = reward
            # 否则 td_target = reward + gamma * np.max(q_table[next_state])
            td_target = None

            # 4. 更新 Q 值
            # TODO 3:
            # 按照 Q-learning 更新公式更新 q_table[state][action]
            #
            # Q(s,a) = Q(s,a) + alpha * (td_target - Q(s,a))
            #

            # 5. 状态推进
            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        # 6. 衰减 epsilon
        # TODO 4:
        # 让 epsilon 逐渐减小，但不要低于 epsilon_min

        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    return q_table, episode_rewards


def test_agent(env, q_table, max_steps=100):
    """
    使用训练好的 Q 表进行测试
    测试阶段不探索，只选最优动作
    """
    state, info = env.reset()
    total_reward = 0
    path = [state]

    for step in range(max_steps):
        # TODO 5:
        # 选择当前状态下 Q 值最大的动作
        action = None

        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        path.append(state)

        if terminated or truncated:
            break

    return total_reward, path


def print_policy(q_table):
    """
    打印最终策略
    动作编码：
    0: Up
    1: Right
    2: Down
    3: Left
    """
    action_symbols = ['U', 'R', 'D', 'L']
    policy = []

    for state in range(q_table.shape[0]):
        best_action = np.argmax(q_table[state])
        policy.append(action_symbols[best_action])

    policy = np.array(policy).reshape(4, 12)

    print("\nLearned Policy:")
    for row in policy:
        print(" ".join(row))


def main():
    env = gym.make("CliffWalking-v1")

    print("Action space:", env.action_space.n)
    print("Observation space:", env.observation_space.n)

    q_table, rewards = train_q_learning(env)

    print("\nTraining finished.")
    print("Final Q-table:")
    print(q_table)

    test_reward, path = test_agent(env, q_table)
    print("\nTest reward:", test_reward)
    print("Path:", path)

    print_policy(q_table)

    env.close()


if __name__ == "__main__":
    main()
