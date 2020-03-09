import gym


class Environment:
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.init_frame = self.env.reset()
        self.action_count = self.env.action_space.n
        self.state_count = self.env.observation_space.shape[0]

    def reset(self):
        return self.env.reset()

    def run(self, agent, should_replay=True):
        state = self.env.reset()
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, is_done, info = self.env.step(action)
            total_reward += reward
            if is_done:
                next_state = None

            agent.observe( (state, action, reward, next_state) )
            if should_replay:
                agent.replay()

            state = next_state

            if is_done:
                break
        return total_reward
