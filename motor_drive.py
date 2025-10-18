"""
Motor and Servo Control Module

Provides classes for controlling standard DC motors via an H-bridge driver and for controlling
servo motors. This module requires the pigpio daemon to be running.
"""

import functools
import pigpio
import time


class DC:
    """Controls a DC motor using an H-bridge motor driver and the pigpio library."""

    def __init__(self, pi: pigpio.pi, en_pin: int, in1: int, in2: int):
        """
        Initializes the DC motor controller.

        Args:
            pi (pigpio.pi): An instance of the pigpio library.
            en_pin (int): The GPIO pin for PWM speed control (Enable pin).
            in1 (int): The first GPIO pin for direction control.
            in2 (int): The second GPIO pin for direction control.
        """

        self.pi = pi
        self.en_pin = en_pin
        self.in1 = in1
        self.in2 = in2

        # Setup GPIO pins as outputs
        self.pi.set_mode(self.en_pin, pigpio.OUTPUT)
        self.pi.set_mode(self.in1, pigpio.OUTPUT)
        self.pi.set_mode(self.in2, pigpio.OUTPUT)

        # Initialize PWM on the enable pin for speed control
        self.pi.set_PWM_range(self.en_pin, 255)  # pigpio uses a PWM range of 0-255
        self.pi.set_PWM_frequency(self.en_pin, 800)  # 0.8kHz

        # Initial State: Stopped
        self.stop()

    def set_speed(self, speed: int):
        """
        Sets the speed and direction of the motor.

        Args:
            speed (int): A value from -100 (full reverse) to 100 (full forward).
        """

        speed = max(-100, min(100, speed))  # Constrain speed
        duty_cycle = int(abs(speed) * 2.55)  # Map speed to 0-255

        # Set direction
        if speed >= 0:
            self.pi.write(self.in1, 1)  # High
            self.pi.write(self.in2, 0)  # Low
        else:
            self.pi.write(self.in1, 0)  # Low
            self.pi.write(self.in2, 1)  # High

        # Set the speed magnitude
        self.pi.set_PWM_dutycycle(self.en_pin, duty_cycle)

    def stop(self):
        """Stops the motor by setting speed to zero."""

        self.pi.write(self.in1, 0)
        self.pi.write(self.in2, 0)
        self.pi.set_PWM_dutycycle(self.en_pin, 0)

    def cleanup(self):
        """Stops the motor."""

        self.stop()


class Servo:
    """Controls a servo motor using the pigpio library."""

    def __init__(self, pi: pigpio.pi, pin: int, center_angle: float = 90.0, rate: int = 300):
        """
        Initializes the servo controller.

        Args:
            pi (pigpio.pi): An instance of the pigpio library.
            pin (int): The GPIO pin connected to the servo's signal wire.
            center_angle (float): The servo's calibrated center angle (default 90.0).
            rate (int): The servo's maximum rotational speed in degrees/second.
        """

        self.pi = pi
        self.pin = pin
        self.center_angle = center_angle
        self.rate = rate
        self._last_time = None

        # Setup GPIO pin as output
        self.pi.set_mode(self.pin, pigpio.OUTPUT)

        # Pulse width values for 0 and 180 degrees
        self.min_pulse = 500  # 500us for 0 deg
        self.max_pulse = 2500  # 2500us for 180 deg

        # Initial State: Centered
        self.current_angle = 0.0
        self.set_angle(0.0)

    def _angle_to_pulsewidth(self, angle: float) -> float:
        """Converts an angle (0-180) to a PWM pulse width (e.g., 500-2500)."""

        return self.min_pulse + (angle / 180.0) * (self.max_pulse - self.min_pulse)

    @staticmethod
    def _limiter(func):
        """Limits the servo's rate of rotation."""

        @functools.wraps(func)
        def wrapper(self, angle_offset: int):
            current_time = time.perf_counter()

            if self._last_time is None:
                dt = 1e-6  # Initial value
            else:
                dt = current_time - self._last_time

            self._last_time = current_time
            max_change = self.rate * dt  # Allowed change for the time delta
            delta = angle_offset - self.current_angle  # Angle delta
            delta = max(-max_change, min(max_change, delta))
            self.current_angle += delta
            func(self, self.current_angle)

        return wrapper

    @_limiter
    def set_angle(self, angle_offset: float):
        """
        Sets the servo to a specific angle based on an offset from center.

        Args:
            angle_offset (float): The angle offset from the center position.
                                  A positive value turns right, negative turns left.
        """

        target_angle = self.center_angle + angle_offset
        target_angle = max(0.0, min(180.0, target_angle))  # Constrain angle
        pulse_width = self._angle_to_pulsewidth(target_angle)
        self.pi.set_servo_pulsewidth(self.pin, pulse_width)

    def cleanup(self):
        """Returns servo to center and stops the PWM."""

        while abs(self.current_angle) > 0.0:
            self.set_angle(0.0)

        self.pi.set_servo_pulsewidth(self.pin, 0.0)
        self._last_time = None
