import time
from typing import Optional

import numpy as np
import serial
from adafruit_servokit import ServoKit

from bblib.defs import EnvironmentConfig, Position, Angle, EnvironmentState, Observation, Action
from bblib.environments.Environment import Environment, EnvironmentFactory
from bblib.utils import compute_angle, random_environment_state

from bblib.sensor import Sensor


sensor = Sensor()

class RealEnvironment(Environment):
    def __init__(self, config: EnvironmentConfig, init_env_state: EnvironmentState):
        super().__init__(config, init_env_state)

        #self.ser = serial.Serial('/dev/ttyACM0', 9600)
        self.kit = ServoKit(channels=16)

        self.center = Angle(78.5, 85.0)
        self.set_servo()
        self.observe_position()
        time.sleep(1.0)
        self.last_time = time.time()
        self.sensor = sensor

    def set_servo(self):
        angle = self.observe_angle()
        x = self.center.x + angle.x * 180 / 3.141592
        y = self.center.y + angle.y * 180 / 3.141592
        
        self.kit.servo[0].angle = y
        self.kit.servo[1].angle = x

    def observe_position(self) -> Position:
        #self.ser.reset_input_buffer()
        while True:
            #self.ser.readline()
            #read_serial = self.ser.readline().decode("utf-8").strip()
            read_serial = "0 , 0"
            tokens = read_serial.split(' , ')
            if 2 != len(tokens):
                continue
            try:
                x, y = [int(v) for v in tokens]
                return Position(x / 1000, y / 1000)
            except ValueError:
                pass

    def observe_real_position(self) -> Optional[Position]:
        return None

    def observe_angle(self) -> Angle:
        return compute_angle(self.config, self.state)

    def update(self, action: Action) -> Observation:
        self.actions.append(action)

        self.state.rot.x = max(-self.config.max_rotation.x,
                               min(self.config.max_rotation.x, self.state.rot.x + action.x))
        self.state.rot.y = max(-self.config.max_rotation.y,
                               min(self.config.max_rotation.y, self.state.rot.y + action.y))

        self.set_servo()

        d_t = time.time() - self.last_time

        if d_t < self.config.d_t:
            time.sleep(self.config.d_t-d_t)
        else:
            print("Warning: too slow!")

        self.last_time = time.time()
        return self.observe()

    def render(self, observation: Observation) -> np.ndarray:
        img = super().render(observation)
        return img


class RealEnvironmentFactory(EnvironmentFactory):
    def __init__(self, env_config: EnvironmentConfig):
        super().__init__(env_config)

    def create(self) -> Environment:
        return RealEnvironment(self.env_config,
                               random_environment_state())
