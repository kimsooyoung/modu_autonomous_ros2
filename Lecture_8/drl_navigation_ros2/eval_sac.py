from pathlib import Path

from TD3.TD3 import TD3
from SAC.SAC import SAC
from ros_python import ROS_env
from replay_buffer import ReplayBuffer
import torch
import numpy as np
from utils import record_eval_positions
from pretrain_utils import Pretraining


def main(args=None):
    """Main training function"""
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 25  # number of input values in the neural network (vector length of state input)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    nr_eval_episodes = 10  # how many episodes to use to run evaluation
    max_steps = 300  # maximum number of steps in single episode
    save_every = 100  # save the model every n training cycles

    model = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        save_every=save_every,
        load_model=True,
    )  # instantiate a model

    ros = ROS_env()  # instantiate ROS environment
    eval_scenarios = record_eval_positions(
        n_eval_scenarios=nr_eval_episodes
    )  # save scenarios that will be used for evaluation

    print("Evaluation...")
    eval(
        model=model,
        env=ros,
        scenarios=eval_scenarios,
        epoch=0,
        max_steps=max_steps,
    )  # run evaluation


def eval(model, env, scenarios, epoch, max_steps):
    """Function to run evaluation"""
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating {len(scenarios)} scenarios")
    avg_reward = 0.0
    col = 0
    gl = 0
    for scenario in scenarios:
        count = 0
        latest_scan, distance, cos, sin, collision, goal, a, reward = env.eval(
            scenario=scenario
        )
        while count < max_steps:
            state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            if terminal:
                break
            action = model.get_action(state, False)
            a_in = [(action[0] + 1) / 2, action[1]]
            latest_scan, distance, cos, sin, collision, goal, a, reward = env.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            avg_reward += reward
            count += 1
            col += collision
            gl += goal
            
    avg_reward /= len(scenarios)
    avg_col = col / len(scenarios)
    avg_goal = gl / len(scenarios)
    print(f"Average Reward: {avg_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print("..............................................")
    model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    model.writer.add_scalar("eval/avg_col", avg_col, epoch)
    model.writer.add_scalar("eval/avg_goal", avg_goal, epoch)


if __name__ == "__main__":
    main()
