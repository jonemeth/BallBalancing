import time
import serial
from PIL import Image
from adafruit_servokit import ServoKit

import numpy as np
from filterpy.kalman import KalmanFilter

from bblib.defs import EnvironmentConfig, Position, Angle, EnvironmentState, Observation, Action, Speed
from bblib.environments.Environment import Environment, EnvironmentFactory
from bblib.utils import compute_angle, init_motion_kalman, random_environment_state


class RealEnvironment(Environment):
    def __init__(self, config: EnvironmentConfig, init_env_state: EnvironmentState):
        super().__init__(config, init_env_state)
        
        self.ser = serial.Serial('/dev/ttyACM0', 9600)
        self.kit = ServoKit(channels=16)
        self.kalman: Optional[KalmanFilter] = None

        self.center = Angle(85.1, 79.0)
        self.set_servo()
        self.observe_position()
        time.sleep(2.0)
        
    def set_servo(self):
        angle = self.observe_angle()
        x = self.center.x + angle.x * 180 / 3.141592
        y = self.center.y + angle.y * 180 / 3.141592
        self.kit.servo[0].angle = x
        self.kit.servo[1].angle = y


    def observe_position(self) -> Position:
        self.ser.reset_input_buffer()
        while True:
            try:
                read_serial = self.ser.readline().decode("utf-8").strip()
                x, y = [int(v) for v in read_serial.split(' , ')]
                break
            except:
                pass
        return Position(x/1000, y/1000)
          

    def observe_angle(self) -> Angle:
        return compute_angle(self.config, self.state)

    def update(self, action: Action) -> Observation:
        self.actions.append(action)

        self.state.rot.x = max(-self.config.max_rotation.x,
                               min(self.config.max_rotation.x, self.state.rot.x + action.x))
        self.state.rot.y = max(-self.config.max_rotation.y,
                               min(self.config.max_rotation.y, self.state.rot.y + action.y))
                               
        self.set_servo()
        time.sleep(self.config.d_t)
        
        return self.observe()
                       
        
    def observe(self) -> Observation:
        observed_pos = self.observe_position()

        if not self.kalman:
            self.kalman = init_motion_kalman(observed_pos, self.config.d_t)
        else:
            self.kalman.predict()
            self.kalman.update([observed_pos.x, observed_pos.y])

        estimated_pos = Position(float(self.kalman.x[0]), float(self.kalman.x[1]))
        estimated_speed = Speed(float(self.kalman.x[2]), float(self.kalman.x[3]))

        angle = self.observe_angle()
        action = self.actions[-1] if 1 <= len(self.actions) else None

        return Observation(estimated_pos, estimated_speed, angle, action, observed_pos,
                           None, self._compute_reward(observed_pos), False)
    
    def render(self, observation: Observation) -> np.ndarray:
        img = super().render(observation)
        return img


class RealEnvironmentFactory(EnvironmentFactory):
    def __init__(self, env_config: EnvironmentConfig):
                     
        super().__init__(env_config)


    def create(self) -> Environment:
        return RealEnvironment(self.env_config,
                                  random_environment_state())
