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
            print("-----------------------------------------")
            try:
                print("Cargando ", model_file + "actor.h5")
                self.actor_model = load_model(model_file + "actor.h5")
            except Exception as e:
                print("Error al cargar el modelo: ", str(e))
                print("Creando modelo desde cero")
                self.actor_model = self.create_actor_model()
        else:
            print("Archivo de modelo no encontrado o no especificado. Creando modelo desde cero")
            self.actor_model = self.create_actor_model()
        print("-----------------------------------------")

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
        hidden_layer = Dropout(0.1)(hidden_layer)

        # hidden_layer = Dense(16, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2_regularizer)(hidden_layer)
        # # hidden_layer = BatchNormalization()(hidden_layer)
        # hidden_layer = Dropout(0.1)(hidden_layer)

        # hidden_layer = Dense(8, activation='relu', kernel_initializer=initializer, kernel_regularizer=l2_regularizer)(hidden_layer)
        # hidden_layer = BatchNormalization()(hidden_layer)

        # Capa de salida para los dos primeros valores (0 a 1)
        output_2 = Dense(2, activation='sigmoid', kernel_initializer=initializer)(hidden_layer)

        # Capa de salida para el tercer valor (-1 a 1)
        output_1 = Dense(1, activation='tanh', kernel_initializer=initializer)(hidden_layer)

        # Capa de salida para la marcha
        gear_output = Dense(1, activation='sigmoid', kernel_initializer=initializer)(hidden_layer)

        # Capa de salida para el embrague
        clutch_output= Dense(1, activation='sigmoid', kernel_initializer=initializer)(hidden_layer)

        # meta= Dense(1, activation='sigmoid', kernel_initializer=initializer)(hidden_layer)

        # Combinar las salidas
        combined_output = Concatenate()([output_1, output_2, gear_output, clutch_output])

        # Crear el modelo
        model = Model(inputs=input_layer, outputs=combined_output)

        # Compilar el modelo con una función de pérdida personalizada
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

            self.random_counter = 5

            # Generar los dos primeros valores en el rango [0, 1]
            first_two_actions = np.random.uniform(0, 1, 2)
            
            # Generar el tercer valor en el rango [-1, 1]
            third_action = np.random.uniform(-1, 1)

            marcha = np.random.choice([-1, 0, 1, 2, 3, 4, 5, 6])
            # marcha = np.random.choice([-1, 1])

            clutch = np.random.uniform(0, 1)

            # Combinar en una sola acción
            action = np.concatenate([[third_action], first_two_actions, [marcha], [clutch], [0]])
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
        valor_marcha_continuo = action[3]  # action[3] es el valor de la marcha en rango 0 a 1
        valor_marcha_escalado = (valor_marcha_continuo * 7) -1  # Escalar y desplazar a rango -1 a 6
        marcha_redondeada = int(round(valor_marcha_escalado))  # Redondear a entero más cercano
        action[3] = marcha_redondeada
        action = np.concatenate([action, [0]])
        
        # print("Acción: ", action)
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
        speed_bonus_factor = 100  # Factor para escalar el premio por velocidad
        penalty_out_of_track = -5  # Penalización por salirse del carril
        penalty_stop = -10
        reward_back_on_track = 5  # Recompensa por volver al carril
        reward_new_lap_record = 100  # Recompensa por nueva vuelta más rápida

        change_dist =  current_dist - previous_dist

        # Si está en carril
        if sensor1 >= 0:

            # Premio por moverse rápido y centrado
            reward = ((change_dist * speed_bonus_factor)) * (1 - abs(track_pos))

            # Si quieto o retrocede -> CASTIGO
            if change_dist <= 0:
                reward = -0.1

        # Si fuera de carril
        if sensor1 < 0:
            # Si antes estaba dentro -> CASTIGO FUERTE
            if not self.was_out_of_track:
                print("SE SALE DEL CARRIL")
                reward = -1
                self.was_out_of_track = True
            # Si ya estaba fuera, premio por acercarse. Castigo si no se acerca
        #     else:
        #         reward = (abs(previous_track_pos) - abs(track_pos)) * 10
        #         if reward <= 0.1:
        #             reward = -0.1
        # else:
        #     # Si antes estaba fuera -> PREMIO
        #     if self.was_out_of_track:
        #         print("ENTRA EN CARRIL")
        #         reward = 0.1
        #         self.was_out_of_track = False

        if self.next_checkpoint > self.track_length and current_dist < 2 :
                self.next_checkpoint = 2

        elif self.next_checkpoint <= 0:
                self.next_checkpoint = int(self.track_length)

        if (current_dist >= self.next_checkpoint):
            if sensor1 >= 0:
                reward = 1
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
            self.next_checkpoint += 2

        # elif (current_dist <= self.next_checkpoint -4):
        #     if sensor1 >= 0:
        #         reward = -100
        #         print("==================================")
        #         print("RETROCEDIDO CHECKPOINT ", str(self.next_checkpoint))
        #         print("==================================")
        #     else:
        #         reward = -100
        #         color = "\033[91m"  # Rojo
        #         end_color = "\033[0m"  # Fin del color
        #         print(f"{color}==================================")
        #         print(f"RETROCEDIDO CHECKPOINT ", str(self.next_checkpoint))
        #         print(f"=================================={end_color}")
        #     self.next_checkpoint -= 2
        
        normalized_reward = reward * 100

        if normalized_reward < 0.00:
            color = "\033[91m"  # Rojo
        else:
            color = "\033[93m"  # Verde        

        end_color = "\033[0m"  # Fin del color
        # print(f"{color}MI REWARD: {normalized_reward}{end_color}")

        return normalized_reward


    def train(self, state, action, reward, next_state, done):

        action[3] = (action[3]+1) / 7
        action = action[:-1]

        # Aplanar estados
        flattened_state = self.flatten_state(state)
        flattened_next_state = self.flatten_state(next_state)

        # Convertir a arreglos NumPy
        np_state = np.array(flattened_state, dtype=np.float32).reshape(1, -1)
        np_next_state = np.array(flattened_next_state, dtype=np.float32).reshape(1, -1)

        # Predecir la acción futura
        predicted_action = self.actor_model.predict(np_next_state, verbose=0)[0]
        

        # Actualizar el actor directamente en función de la recompensa
        with tf.GradientTape() as tape:
            # Calcular la predicción de la acción actual
            current_action_prediction = self.actor_model(np_next_state)

            # Usar alguna función o lógica para calcular la recompensa en relación con la acción
            # Esto puede ser una función simple o una más compleja dependiendo del problema
            actor_reward = self.calculate_actor_reward(reward, action, current_action_prediction)

            # Se busca maximizar la recompensa (minimizar -actor_reward)
            actor_loss = -tf.reduce_mean(actor_reward)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

        # target = reward
        # if not done:
        #     target = (reward + 0.95 * np.amax(self.actor_model.predict(np_next_state.reshape(1, -1), verbose=0)[0]))
        # target_f = self.actor_model.predict(np_state.reshape(1, -1), verbose=0)
        # target_f[0][np.argmax(action)] = target
        # self.actor_model.fit(np_state.reshape(1, -1), target_f, epochs=1, verbose=0)


    def calculate_actor_reward(self, reward, action, predicted_action):
        # Implementar la lógica para calcular la recompensa del actor
        # Esto puede incluir la comparación de la acción predicha con la acción real,
        # y el uso de la recompensa recibida del entorno
        # Ejemplo simple:
        return reward - tf.reduce_mean(tf.square(action - predicted_action))


    def save_model(self, file_name):
        print("Guardando modelo del actor en", file_name + "actor.h5")
        self.actor_model.save(file_name + "actor.h5")
