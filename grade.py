import importlib
import traceback
import gymnasium as gym
import numpy as np
import random


TOTAL_SCORE = 90


def score_item(name, got, full, detail=""):
    print(f"[{'PASS' if got > 0 else 'FAIL'}] {name}: {got}/{full}")
    if detail:
        print(f"       {detail}")
    return got


def evaluate():
    score = 0

    try:
        student = importlib.import_module("main")
    except Exception as e:
        print("[FATAL] 无法导入 main.py")
        print(traceback.format_exc())
        return 0

    # ----------------------------
    # 1. 检查函数是否存在
    # ----------------------------
    required_funcs = ["choose_action", "train_q_learning", "test_agent", "print_policy"]
    for fn in required_funcs:
        if hasattr(student, fn):
            score += score_item(f"函数存在: {fn}", 5, 5)
        else:
            score += score_item(f"函数存在: {fn}", 0, 5, "缺少该函数")

    # 如果核心函数缺失，后续很多测试没法做
    if not all(hasattr(student, fn) for fn in ["choose_action", "train_q_learning", "test_agent"]):
        print(f"\nFinal Score: {score}/{TOTAL_SCORE}")
        return score

    # ----------------------------
    # 2. 检查 choose_action
    # ----------------------------
    try:
        q_table = np.array([
            [1.0, 2.0, 0.5, -1.0],
            [0.0, 0.0, 0.0, 0.0]
        ])

        # epsilon = 0 时必须贪心
        action = student.choose_action(0, q_table, 0.0, 4)
        if action == 1:
            score += score_item("choose_action 贪心选择", 10, 10)
        else:
            score += score_item("choose_action 贪心选择", 0, 10, f"期望 1，实际 {action}")

        # epsilon = 1 时应随机探索
        actions = [student.choose_action(0, q_table, 1.0, 4) for _ in range(50)]
        unique_actions = len(set(actions))
        if unique_actions > 1:
            score += score_item("choose_action 随机探索", 10, 10)
        else:
            score += score_item("choose_action 随机探索", 0, 10, "看起来没有随机探索")
    except Exception:
        score += score_item("choose_action 功能测试", 0, 20, traceback.format_exc())

    # ----------------------------
    # 3. 检查训练函数能否正常运行
    # ----------------------------
    try:
        env = gym.make("CliffWalking-v1")
        q_table, rewards = student.train_q_learning(
            env,
            episodes=300,
            max_steps=100,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )

        ok = True
        detail = []

        if not isinstance(q_table, np.ndarray):
            ok = False
            detail.append("q_table 不是 numpy.ndarray")

        if not isinstance(rewards, list):
            ok = False
            detail.append("rewards 不是 list")

        if isinstance(q_table, np.ndarray) and q_table.shape != (48, 4):
            ok = False
            detail.append(f"q_table 形状错误: {q_table.shape}，应为 (48, 4)")

        if isinstance(rewards, list) and len(rewards) != 300:
            ok = False
            detail.append(f"rewards 长度错误: {len(rewards)}，应为 300")

        if ok:
            score += score_item("train_q_learning 基本运行", 20, 20)
        else:
            score += score_item("train_q_learning 基本运行", 5, 20, "; ".join(detail))

        env.close()
    except Exception:
        score += score_item("train_q_learning 基本运行", 0, 20, traceback.format_exc())
        print(f"\nFinal Score: {score}/{TOTAL_SCORE}")
        return score

    # ----------------------------
    # 4. 检查测试函数能否运行
    # ----------------------------
    try:
        env = gym.make("CliffWalking-v1")
        test_reward, path = student.test_agent(env, q_table, max_steps=100)

        ok = True
        detail = []

        if not isinstance(test_reward, (int, float, np.integer, np.floating)):
            ok = False
            detail.append("test_reward 类型不正确")

        if not isinstance(path, list):
            ok = False
            detail.append("path 不是 list")

        if isinstance(path, list) and len(path) == 0:
            ok = False
            detail.append("path 为空")

        if ok:
            score += score_item("test_agent 基本运行", 15, 15)
        else:
            score += score_item("test_agent 基本运行", 5, 15, "; ".join(detail))

        env.close()
    except Exception:
        score += score_item("test_agent 基本运行", 0, 15, traceback.format_exc())

    # ----------------------------
    # 5. 简单效果评估
    # ----------------------------
    # 用训练后的策略和随机策略比较平均表现
    try:
        def run_random_policy(episodes=30, max_steps=100):
            env = gym.make("CliffWalking-v1")
            total = 0
            for _ in range(episodes):
                state, _ = env.reset()
                ep_reward = 0
                for _ in range(max_steps):
                    action = env.action_space.sample()
                    state, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    if terminated or truncated:
                        break
                total += ep_reward
            env.close()
            return total / episodes

        def run_greedy_policy(q_table, episodes=30, max_steps=100):
            env = gym.make("CliffWalking-v1")
            total = 0
            for _ in range(episodes):
                state, _ = env.reset()
                ep_reward = 0
                for _ in range(max_steps):
                    action = int(np.argmax(q_table[state]))
                    state, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    if terminated or truncated:
                        break
                total += ep_reward
            env.close()
            return total / episodes

        random_avg = run_random_policy()
        greedy_avg = run_greedy_policy(q_table)

        print(f"\n随机策略平均回报: {random_avg:.2f}")
        print(f"训练后贪心策略平均回报: {greedy_avg:.2f}")

        if greedy_avg > random_avg:
            score += score_item("训练效果优于随机策略", 15, 15)
        else:
            score += score_item(
                "训练效果优于随机策略",
                5,
                15,
                "代码能运行，但训练效果没有明显优于随机策略"
            )
    except Exception:
        score += score_item("训练效果评估", 0, 15, traceback.format_exc())

    print(f"\nFinal Score: {score}/{TOTAL_SCORE}")
    return score


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    evaluate()
