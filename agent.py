import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Concatenate, Dense, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from keras.initializers import he_normal
from keras.regularizers import l2



class Agent(object):
    def __init__(self, dim_action, dim_observation, learning_rate=0.01, model_file=None):
        self.dim_action = dim_action
        self.dim_observation = dim_observation
        self.learning_rate = learning_rate
        self.control_gear = True
        self.was_out_of_track = False
        self.last_action = []
        self.random_counter = -1
        self.next_checkpoint = 2035
        self.track_length = 2057.56
        
        # Exploración disminute exponencialmente
        self.exp_inicial = 0.3
        self.exp_tasa = self.exp_inicial
        self.min_exp = 0.05
        self.exp_decay = 0.995

        # self.best_distance = 3289.55# La mejor distancia alcanzada ACTUALIZAR MANUALMENTE
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        if model_file:
            print("Cargando ", model_file)
            self.actor_model = load_model(model_file + "actor.h5")
            self.critic_model = load_model(model_file + "critic.h5")
        else:
            print("Creando modelo desde cero")
            self.actor_model = self.create_actor_model()
            self.critic_model = self.create_critic_model()

    def update_exploration_rate(self):

        # Aplicar decaimiento exponencial a la tasa de exploración
        self.exp_tasa *= self.exp_decay

        # Asegurarse de que la tasa de exploración no caiga por debajo del mínimo
        self.exp_tasa = max(self.exp_tasa, self.min_exp)


    def create_actor_model(self):
        # Inicializador de pesos
        initializer = he_normal()

        l2_regularizer = l2(0.01)  # L2 Regularization

        # Entrada
        input_layer = Input(shape=(self.dim_observation,))

        # Capas ocultas más profundas y con más neuronas
        hidden_layer = Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2_regularizer)(input_layer)
        # hidden_layer = BatchNormalization()(hidden_layer)
        # hidden_layer = Dropout(0.1)(hidden_layer)

        hidden_layer = Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2_regularizer)(hidden_layer)
        # hidden_layer = BatchNormalization()(hidden_layer)
        # hidden_layer = Dropout(0.1)(hidden_layer)

        hidden_layer = Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2_regularizer)(hidden_layer)
        # hidden_layer = BatchNormalization()(hidden_layer)

        # Capa de salida para los dos primeros valores (0 a 1)
        output_2 = Dense(2, activation='sigmoid', kernel_initializer=initializer)(hidden_layer)

        # Capa de salida para el tercer valor (-1 a 1)
        output_1 = Dense(1, activation='tanh', kernel_initializer=initializer)(hidden_layer)

        # Capa de salida para la marcha (clasificación multiclase)
        gear_output = Dense(1, activation='tanh', kernel_initializer=initializer)(hidden_layer)

        clutch_output= Dense(1, activation='sigmoid', kernel_initializer=initializer)(hidden_layer)

        # meta= Dense(1, activation='sigmoid', kernel_initializer=initializer)(hidden_layer)

        # Combinar las salidas
        combined_output = Concatenate()([output_1, output_2, gear_output, clutch_output])

        # Crear el modelo
        model = Model(inputs=input_layer, outputs=combined_output)

        # Compilar el modelo con una función de pérdida personalizada
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model


    def create_critic_model(self):
        initializer = he_normal()
        l2_regularizer = l2(0.01)  # L2 Regularization

        # Entrada para el estado observado
        state_input = Input(shape=(self.dim_observation,))
        state_h1 = Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2_regularizer)(state_input)
        hidden_layer = BatchNormalization()(state_h1)

        # Entrada para la acción
        action_input = Input(shape=(self.dim_action,))
        action_h1 = Dense(64, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2_regularizer)(action_input)
        hidden_layer = BatchNormalization()(action_h1)

        # Combinar las dos redes
        merged = Concatenate()([state_h1, action_h1])

        # Reducción de dimensiones de las capas intermedias
        merged_h1 = Dense(128, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2_regularizer)(merged)
        merged_h1 = Dropout(0.2)(merged_h1)  # Dropout para regularización

        # Salida - Valor de la acción en el estado dado
        output = Dense(1, activation='linear', kernel_initializer=initializer)(merged_h1)  # Activación lineal en la salida

        # Crear modelo
        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def normalize_tanh(self, speed):
        return np.tanh(speed)

    def normalize_state(self, state):

        # Valores aproximados, pueden cambiar
        max_lap_time = 10000
        normalized_state = []

        # Normalizar los diferentes componentes del estado
        
        # normalized_state.append(state[0] / np.pi)  # Angle
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

        return normalized_state


    def act(self, state):

        # print("STATE:")
        # print("Angle: ",state[0])
        # print("Track: ",state[1:20])
        # print("TrackPos: ",state[20])
        # print("SpeedX: ",state[21])
        # print("SpeedY: ",state[22])
        # print("SpeedZ: ",state[23])
        # print("WheelSpin: ",state[24:28])
        # print("DistFromStart: ", state[28])
        # print("CurLapTime: ",state[29])
        # print("LastLapTime: ",state[30])

        if self.random_counter != -1:
                self.random_counter -= 1
                action = self.last_action
                # print("Acción aleatoria: ", action)
                return action

        if np.random.rand() <= self.exp_tasa:

            self.random_counter = 10

            # Generar los dos primeros valores en el rango [0, 1]
            first_two_actions = np.random.uniform(0, 1, 2)
            
            # Generar el tercer valor en el rango [-1, 1]
            third_action = np.random.uniform(-1, 1)

            marcha = np.random.choice([-1, 0, 1, 2, 3, 4, 5, 6])
            # marcha = np.random.choice([-1, 1])

            clutch = np.random.uniform(0, 1)

            # Combinar en una sola acción
            action = np.concatenate([[third_action], first_two_actions, [marcha], [clutch]])
            self.last_action = action

            # print("Acción aleatoria: ", action)
            return action
        
        # Aplanar el estado si es necesario
        flattened_state = self.flatten_state(state)

        # Convertir a un arreglo NumPy con tipo float32
        np_state = np.array(flattened_state, dtype=np.float32).reshape(1, -1)

        # Utilizar el modelo del actor para predecir la acción basada en el estado
        action = self.actor_model.predict(np_state, verbose=0)[0]

        # Transformar el valor de la marcha
        valor_marcha_continuo = action[3]  # action[3] es el valor de la marcha en rango -1 a 1
        valor_marcha_escalado = (valor_marcha_continuo + 1) * 3.5  # Escalar y desplazar a rango 0-7
        marcha_redondeada = int(round(valor_marcha_escalado))  # Redondear a entero más cercano

        # Ajustar para obtener la marcha en rango -1 a 6
        marcha_redondeada -= 1
        action[3] = marcha_redondeada
        
        print("Acción: ", action)
        return action

    def flatten_state(self, state):
        # Aplanar los datos para que sea una lista de float
        flattened_state = []
        for element in state:
            if isinstance(element, list):
                # Extiende la lista aplanada con los elementos de la sublista
                flattened_state.extend(element)
            else:
                # Añade el elemento individual
                flattened_state.append(element)
        return flattened_state

    def update_reward(self, current_dist, previous_dist, track_pos, previous_track_pos, sensor1):
        reward = 0
        speed_bonus_factor = 1  # Factor para escalar el premio por velocidad
        penalty_out_of_track = -5  # Penalización por salirse del carril
        penalty_stop = -10
        reward_back_on_track = 5  # Recompensa por volver al carril
        reward_new_lap_record = 100  # Recompensa por nueva vuelta más rápida

        change_dist =  current_dist - previous_dist
        # if -0.1< speedX < 0.1:
        #     reward = penalty_stop

        # # Premio por velocidad dentro del carril
        # if -1.0 <= track_pos <= 1.0:
        #     reward +=  -speed_bonus_factor * change_dist

        #     # Verificar si ha vuelto al carril
        #     if self.was_out_of_track:
        #         reward += reward_back_on_track
        #         self.was_out_of_track = False

        # else:    
        #     # Penalización por salirse del carril    
        #     if not self.was_out_of_track:
        #         reward += penalty_out_of_track
        #         self.was_out_of_track = True
        #     else:
        #         reward = -0.5 * abs(track_pos)

        # Si está en carril
        if sensor1 >= 0:

            # Premio por moverse rápido y centrado
            reward = ((change_dist * speed_bonus_factor)) * (1 - abs(track_pos))

            # Si quieto o retrocede -> CASTIGO
            if change_dist <= 0:
                reward = -1

            
            # self.was_out_of_track = True
            if self.was_out_of_track:
                color = "\033[93m"  # Amarillo
                end_color = "\033[0m"  # Fin del color
                print(f"{color}=================================={end_color}")
                print(f"{color}VUELVE AL CARRIL{end_color}")
                print(f"{color}=================================={end_color}")
                self.was_out_of_track = False
                reward = 100
        # Si fuera de carril
        else:
            # Si antes estaba dentro -> CASTIGO FUERTE
            if not self.was_out_of_track:
                color = "\033[91m"  # Rojo
                end_color = "\033[0m"  # Fin del color
                print(f"{color}=================================={end_color}")
                print(f"{color}SE SALE DEL CARRIL{end_color}")
                print(f"{color}=================================={end_color}")
                reward = -1000
                self.was_out_of_track = True
            # Si ya estaba fuera, premio por acercarse. Castigo si no se acerca
            else:
                reward = (abs(previous_track_pos) - abs(track_pos)) * 100
                if reward <= 0:
                    reward = -10

        # # Si fuera de carril
        # if abs(track_pos) >= 1:
        #     # Si antes estaba dentro -> CASTIGO FUERTE
        #     if not self.was_out_of_track:
        #         print("SE SALE DEL CARRIL")
        #         reward = -1000
        #         self.was_out_of_track = True
        #     # Si ya estaba fuera, premio por acercarse. Castigo si no se acerca
        #     else:
        #        reward = -10 
        # else:
        #     # Si antes estaba fuera -> PREMIO
        #     if self.was_out_of_track:
        #         print("ENTRA EN CARRIL")
        #         reward = 100
        #         self.was_out_of_track = False


        if (current_dist >= self.next_checkpoint):
            if sensor1 >= 0:
                reward = 100
                print("==================================")
                print("ALCANZADO CHECKPOINT ", str(self.next_checkpoint))
                print("==================================")
            else:
                color = "\033[91m"  # Rojo
                end_color = "\033[0m"  # Fin del color
                print(f"{color}==================================")
                print(f"ALCANZADO CHECKPOINT ", str(self.next_checkpoint))
                print(f"=================================={end_color}")
            self.next_checkpoint += 1
        
        if self.next_checkpoint > self.track_length and current_dist < 1 :
                self.next_checkpoint = 1
            

        # print("DIST ANTES: ", previous_dist)
        # print("DIST AHORA: ", current_dist)
        # print("DIST CAMBIO: ", change_dist)

        # # Recompensa por mejorar el tiempo de vuelta
        # if self.last_lap_time and current_lap_time < self.last_lap_time:
        #     reward += reward_new_lap_record
        #     self.last_lap_time = current_lap_time

        # Normalizar la recompensa total
        # normalized_reward = np.clip(reward, -1000, 10000)
        
        normalized_reward = reward

        if normalized_reward < 0.00:
            color = "\033[91m"  # Rojo
        else:
            color = "\033[93m"  # Verde        

        end_color = "\033[0m"  # Fin del color
        # print(f"{color}MI REWARD: {normalized_reward}{end_color}")

        return normalized_reward


    def train(self, state, action, reward, next_state, done):

        # Aplanar estados
        flattened_state = self.flatten_state(state)
        flattened_next_state = self.flatten_state(next_state)

        # Convertir a arreglos NumPy
        np_state = np.array(flattened_state, dtype=np.float32).reshape(1, -1)
        np_next_state = np.array(flattened_next_state, dtype=np.float32).reshape(1, -1)

        # Predecir la acción futura
        predicted_action = self.actor_model.predict(np_next_state, verbose=0)[0]

        # Predecir el valor del estado actual y el siguiente estado
        current_value = self.critic_model.predict([np_state, action.reshape(1, -1)], verbose=0)[0]
        current_value = np.clip(current_value, -100, 100)
        next_value = self.critic_model.predict([np_next_state, predicted_action.reshape(1, -1)], verbose=0)[0]
        next_value = np.clip(next_value, -100, 100)
        # Calcular el valor objetivo
        target = reward
        
        if not done:
            target = reward + 0.9 * next_value  # 0.9 es el factor de descuento

        # if current_value < 0.00:
        #     color = "\033[91m"  # Rojo
        # else:
        #     color = "\033[92m"  # Verde        

        # end_color = "\033[0m"  # Fin del color
        # print(f"{color}REWARD CRITIC: {current_value}{end_color}")


        # Calcular el error (TD error)
        delta = target - current_value

        # Actualizar el crítico
        self.critic_model.fit([np_state, action.reshape(1, -1)], np.array([delta]), epochs=1, verbose=0)

        # Actualizar el actor
        # Para el actor, queremos mover la acción en la dirección que mejora su valor según el crítico
        with tf.GradientTape() as tape:
            # Calcula el valor para la acción predicha
            predicted_action_for_gradient = self.actor_model(np_state)
            values = self.critic_model([np_state, predicted_action_for_gradient])
            # Se busca maximizar los valores (minimizar -valores)
            actor_loss = -tf.reduce_mean(values)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    def save_model(self, file_name):
        print("Guardando modelo del actor en", file_name + "actor.h5")
        self.actor_model.save(file_name + "actor.h5")

        print("Guardando modelo del crítico en", file_name + "critic.h5")
        self.critic_model.save(file_name + "critic.h5")
