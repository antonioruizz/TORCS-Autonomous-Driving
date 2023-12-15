from gym_torcs import TorcsEnv
from sample_agent import Agent
import numpy as np

# Configuraciones iniciales
num_episodes = 100
max_steps = 10  # Pasos máximos por episodio

# Inicializar el entorno y el agente
env = TorcsEnv(vision=False, throttle=True, gear_change=False)
agent = Agent(dim_action=env.action_space.shape[0], dim_observation=env.observation_space.shape[0])

# Bucle de episodios
for episode in range(num_episodes):
    total_reward = 0
    state = env.reset()
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.train(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    print(f'Episodio: {episode}, Recompensa Total: {total_reward}')

# Cerrar el entorno
env.close()

