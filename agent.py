import numpy as np
from actor_network import ActorNetwork
from critic_network import CriticNetwork


import torch
import torch.nn as nn
import torch.optim as optim




class Agent:
    def __init__(self, dim_action, dim_observation, learning_rate=0.01, model_file=None):

        self.dim_action = dim_action
        self.dim_observation = dim_observation
        self.learning_rate = learning_rate
        self.control_gear = True
        self.was_out_of_track = False
        self.last_action = []
        self.random_counter = -1
        self.track_length = 2057.56
        self.MAX_REWARD = 1000
        self.cruzaMetaInicio = False
        self.cercaFinal = False

        #CheckPoints
        self.lastCheckPointTime = 0.0
        self.next_checkpoint = 2035
        self.last_checkpoint = 2034
        self.lastReward = 0

        # Exploración disminute exponencialmente
        self.exp_inicial = 1
        self.exp_tasa = self.exp_inicial
        self.min_exp = 0
        self.exp_decay = 0.995



        if model_file:
            print("-----------------------------------------")
            try:
                print("Cargando ", model_file + "actor.pth")
                print("Cargando ", model_file + "critic.pth")
                self.load_model(model_file)
            except Exception as e:
                print("Error al cargar el modelo: ", str(e))
                print("Creando modelo desde cero")
                self.actor_model = ActorNetwork(dim_observation, dim_action)
                self.critic_model = CriticNetwork(dim_observation, dim_action)
        else:
            print("Archivo de modelo no encontrado o no especificado. Creando modelo desde cero")
            self.actor_model = ActorNetwork(dim_observation, dim_action)
            self.critic_model = CriticNetwork(dim_observation, dim_action)
        print("-----------------------------------------")

        self.optimizer = optim.Adam(self.actor_model.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=learning_rate)
        # print("Pesos de fc1 después de la inicialización:", self.critic_model.fc1.weight)


    def update_exploration_rate(self):

        # Aplicar decaimiento exponencial a la tasa de exploración
        self.exp_tasa *= self.exp_decay

        # Asegurarse de que la tasa de exploración no caiga por debajo del mínimo
        self.exp_tasa = max(self.exp_tasa, self.min_exp)


    def normalize_tanh(self, speed):
        return np.tanh(speed)

    def normalize_state(self, state):

        # Valores aproximados, pueden cambiar
        max_lap_time = 10000
        normalized_state = []

        # Normalizar los diferentes componentes del estado

        normalized_state.append(state[0])  # Angle
        normalized_state.extend([s / 200 for s in state[1]])  # Track sensors
        norm_trackPos = self.normalize_tanh(state[2])
        normalized_state.append(norm_trackPos)  # TrackPos

        # Normalizar SpeedX, SpeedY, SpeedZ usando la función normalize_speed
        normalized_state.extend([self.normalize_tanh(s) for s in state[3:6]])

        # Normalizar WheelSpin
        normalized_state.extend([self.normalize_tanh(s) for s in state[6]])

        # Normalizar DistFromStart
        normalized_state.append(state[7] / self.track_length)

        # Normalizar CurLapTime, LastLapTime
        normalized_state.extend([self.normalize_tanh(s / max_lap_time) for s in state[8:10]])

        normalized_state = np.nan_to_num(normalized_state, nan=-1)

        return normalized_state


    def act(self, state):

        # print("STATE:")
        # print("Angle: ",state[0])
        # # print("Track: ",state[1:20])
        # print("TrackPos: ",state[20])
        # print("SpeedX: ",state[21])
        # # print("SpeedY: ",state[22])
        # # print("SpeedZ: ",state[23])
        # # print("WheelSpin: ",state[24:28])
        # print("DistFromStart: ", state[28])
        # print("CurLapTime: ",state[29])
        # # print("LastLapTime: ",state[30])

        # Convertir 'state' a un tensor PyTorch
        state = np.array(state)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        # Colocar el modelo en modo de evaluación
        self.actor_model.eval()

        if self.random_counter != -1:
                self.random_counter -= 1
                # print("Acción aleatoria: ", self.last_action)
                return self.last_action

        if np.random.rand() <= self.exp_tasa:

            self.random_counter = 5

            action = np.random.uniform(-1, 1, 5)

            self.last_action = action

            return action

        # Obtener la predicción
        # with torch.no_grad():
        action = self.actor_model(state_tensor).squeeze(0).detach().numpy()

        # print("Acción: ", action)
        return action

    def update_reward(self, state):

        reward = 0

        currentTime = state[29]

        sensor1 = state[1]

        current_dist = state[28] * self.track_length

        # Distancia hasta la meta
        distStart = state[28]
        if not self.cruzaMetaInicio:
            if distStart > 0.1:
                distStart = 0
            else:
                self.cruzaMetaInicio = True
        elif not self.cercaFinal:
            if distStart > 0.9:
                distStart = 0
                self.cruzaMetaInicio = False
            elif distStart > 0.1:
                self.cercaFinal = True

        # Calculo de velocidad
        speedFactor = 100
        speedX = state[21]
        wheelSpin = state[24:28]
        # meanWheelSpin = abs(sum(wheelSpin) / 4)
        # speed = speedX * meanWheelSpin / speedFactor
        speed = speedX / speedFactor

        # Distancia a centro del carril
        trackPos = state[20]
        centerTrack = 1- abs(trackPos)

        # print("distStart: ", distStart)
        # print("speedX: ", speedX)
        # print("wheelSpin: ", wheelSpin)
        # print("trackPos: ", trackPos)
        # print("speed: ", speed)

        reward = ((distStart * 0.4) + (speed * 0.6 * centerTrack)) * 1000
        # reward = distStart
        # if reward < 0000.1:
        #     reward = 0000.1
        # print("REWARD: ", reward)


        # reward = 0
        # speed_bonus_factor = 10  # Factor para escalar el premio por velocidad
        # penalty_out_of_track = -5  # Penalización por salirse del carril
        # penalty_stop = -10
        # reward_back_on_track = 5  # Recompensa por volver al carril
        # reward_new_lap_record = 100  # Recompensa por nueva vuelta más rápida

        # change_dist =  current_dist - previous_dist

        # # Si está en carril
        # if sensor1 >= 0:

        #     # Premio por moverse rápido y centrado
        #     reward = ((change_dist * speed_bonus_factor)) * (1 - abs(track_pos))

        #     # Si quieto o retrocede -> CASTIGO
        #     # if change_dist <= 0:
        #     #     reward = -0.1

        # # Si fuera de carril
        if sensor1 < 0:
            # Si antes estaba dentro -> CASTIGO FUERTE
            if not self.was_out_of_track:
                reward = -100
                self.was_out_of_track = True
        #     # Si ya estaba fuera, premio por acercarse. Castigo si no se acerca
        # #     else:
        # #         reward = (abs(previous_track_pos) - abs(track_pos)) * 10
        # #         if reward <= 0.1:
        # #             reward = -0.1
        # # else:
        # #     # Si antes estaba fuera -> PREMIO
        # #     if self.was_out_of_track:
        # #         print("ENTRA EN CARRIL")
        # #         reward = 0.1
        # #         self.was_out_of_track = False

        if self.next_checkpoint > self.track_length and current_dist < 100 :
                    self.next_checkpoint = 1
                    self.cruzaMetaInicio = True

        # elif self.next_checkpoint <= 0:
        #         self.next_checkpoint = int(self.track_length)

        if (current_dist >= self.next_checkpoint):
            # if sensor1 >= 0:
            #     reward = 1
            # print("==================================")
            # print("ALCANZADO CHECKPOINT ", str(self.next_checkpoint))
            # print("==================================")
            # else:
            #     reward = 0.1
            #     color = "\033[91m"  # Rojo
            #     end_color = "\033[0m"  # Fin del color
            #     print(f"{color}==================================")
            #     print(f"ALCANZADO CHECKPOINT ", str(self.next_checkpoint))
            #     print(f"=================================={end_color}")

            if not self.cruzaMetaInicio:
        #         reward = (1 - (currentTime - self.lastCheckPointTime)) * 1000
        #         self.lastCheckPointTime = currentTime
                self.last_checkpoint = self.next_checkpoint
                self.next_checkpoint += 1
        #         self.lastReward = reward

            elif not self.cercaFinal:
                if current_dist > 2000:
                    self.next_checkpoint = 1
                elif current_dist > 1000:
        #             reward = (1 - (currentTime - self.lastCheckPointTime)) * 1000
        #             self.lastCheckPointTime = currentTime
                    self.last_checkpoint = self.next_checkpoint
                    self.next_checkpoint += 1
        #             self.lastReward = reward
                    self.cercaFinal = True
                else:
        #             reward = (1 - (currentTime - self.lastCheckPointTime)) * 1000
        #             self.lastCheckPointTime = currentTime
                    self.last_checkpoint = self.next_checkpoint
                    self.next_checkpoint += 1
        #             self.lastReward = reward

            elif self.cruzaMetaInicio and self.cercaFinal:
        #             reward = (1 - (currentTime - self.lastCheckPointTime)) * 1000
        #             self.lastCheckPointTime = currentTime
                    self.last_checkpoint = self.next_checkpoint
                    self.next_checkpoint += 1
        #             self.lastReward = reward


        # if self.lastCheckPointTime != currentTime:
        #     self.lastReward -= 1
        #     reward = self.lastReward

        # # elif (current_dist <= self.next_checkpoint -4):
        # #     if sensor1 >= 0:
        # #         reward = -100
        # #         print("==================================")
        # #         print("RETROCEDIDO CHECKPOINT ", str(self.next_checkpoint))
        # #         print("==================================")
        # #     else:
        # #         reward = -100
        # #         color = "\033[91m"  # Rojo
        # #         end_color = "\033[0m"  # Fin del color
        # #         print(f"{color}==================================")
        # #         print(f"RETROCEDIDO CHECKPOINT ", str(self.next_checkpoint))
        # #         print(f"=================================={end_color}")
        # #     self.next_checkpoint -= 2

        # normalized_reward = reward * 100

        # if normalized_reward < 0.00:
        #     color = "\033[91m"  # Rojo
        # else:
        #     color = "\033[93m"  # Verde

        # end_color = "\033[0m"  # Fin del color
        # # print(f"{color}MI REWARD: {normalized_reward}{end_color}")
        # print(speedX)

        # if speedX > 0:
        #     max_speed = 300  # Asumiendo 300 como un valor máximo razonable para speedX
        #     normalized_speed = min(speedX / max_speed, 1) * 1000
        #     reward = normalized_speed * self.MAX_REWARD
        # else:
        #     reward = 1

        if reward <= 0:
            reward = 0.00001
        # print("REWARD: ", reward)
        return min(reward, self.MAX_REWARD)



        return min(reward, self.MAX_REWARD)

    def train(self, state, action, reward, done):

        state_tensor = torch.tensor(state, dtype=torch.float)
        action_tensor = torch.tensor(action, dtype=torch.float)
        reward_tensor = torch.tensor(reward, dtype=torch.float)

        # Predecir la acción usando la red del Actor
        self.actor_model.train()
        predicted_action = self.actor_model(state_tensor)


        # Evaluar la acción usando la red del Crítico
        self.critic_model.train()
        value = self.critic_model(state_tensor, action_tensor)
        predicted_value = self.critic_model(state_tensor, predicted_action)

        target_value = reward_tensor
        if not done:
            target_value = reward_tensor + 0.95 * predicted_value
        critic_loss = torch.nn.functional.mse_loss(value, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Paso 2: Actualizar el Actor
        self.actor_model.train()
        predicted_action = self.actor_model(state_tensor)
        actor_loss = -self.critic_model(state_tensor, predicted_action).mean()


        # print("state_tensor: ", state_tensor)
        # print("ACCION REAL: ", action_tensor)
        # print("ACCION ACTOR: ", predicted_action)
        # print("MI REWARD: ", reward_tensor)
        # print("REWARD CRITICO REAL: ", value)
        # print("REWARD CRITICO DEL ACTOR: ", predicted_value)
        # print("ACTOR LOSS: ", actor_loss)
        # print("CRITIC LOSS: ", critic_loss)
        # print("++++++++++++++++++++++++++++++++++++++++++++")

        # Backpropagation y optimización
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()


        # Opcional: Imprimir los gradientes
        # for name, param in self.actor_model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad)




    def save_model(self, file_name):
        torch.save(self.actor_model.state_dict(), file_name + 'actor.pth')
        torch.save(self.critic_model.state_dict(), file_name + 'critic.pth')

    def load_model(self, file_name):
        self.actor_model = ActorNetwork(self.dim_observation, self.dim_action)
        self.actor_model.load_state_dict(torch.load(file_name + 'actor.pth'))
        self.critic_model = CriticNetwork(self.dim_observation, self.dim_action)
        self.critic_model.load_state_dict(torch.load(file_name + 'critic.pth'))

