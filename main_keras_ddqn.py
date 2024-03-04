import gym
from gym import wrappers
import numpy as np
from ddqn_kerasLP import DDQNAgent
from utils import plotLearning

if __name__ == '__main__': 
    env = gym.make('LunarLander-v2')
    ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=4, epsilon=1.0, batch_size=64, input_dims=8)

    n_games = 500
    # dd1n_agent.load_model()
    ddqn_scores = []
    eps_history = []

    env = wrappers.Monitor( env, "tmp/lunar-lander", video_callable=lambda episode_id: True, force=True)

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done: 
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            ddqn_agent.learn()
        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        print('episode',i, 'avg score %.1f' % avg_score, 'score %.1f' % score, 'epsilon %.2f' % ddqn_agent.epsilon)

        if i % 10 == 0 and i > 0:
            ddqn_agent.save_model()

    filename = 'lunar_lander.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, ddqn_scores, eps_history, filename)