import numpy as np
from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1 as mamujoco


def main():
    env = mamujoco.parallel_env(
        scenario='Ant',
        agent_conf='2x4',
        agent_obsk=1,
        render_mode=None,
    )
    obs, infos = env.reset(seed=0)
    print('agents:', env.agents)
    print('obs_shape(agent_0):', np.array(obs[env.agents[0]]).shape)

    total_reward = 0.0
    for t in range(5):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        team_reward = float(sum(rewards.values()))
        total_reward += team_reward
        print(f'step={t} team_reward={team_reward:.6f}')
        if all(terms.values()) or all(truncs.values()):
            print('episode ended early at step', t)
            break

    print('total_reward:', total_reward)
    env.close()


if __name__ == '__main__':
    main()
