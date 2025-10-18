"""
PID Controller Module

This module provides a simple PID (Proportional-Integral-Derivative) controller class.
"""

import time


class PID:
    """A controller that calculates an output to correct an error over time."""

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0,
                 out_min: float = -1.0, out_max: float = 1.0, integral_limit: float = None):
        """
        Initializes the PID controller.

        Args:
            kp (float): The Proportional gain.
            ki (float): The Integral gain.
            kd (float): The Derivative gain.
            out_min (float): The minimum value for the controller's output.
            out_max (float): The maximum value for the controller's output.
            integral_limit (float, optional): The anti-windup limit for the integral term.
                                              Defaults to out_min/out_max.
        """

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = out_min
        self.out_max = out_max
        self.integral_limit = integral_limit or max(abs(out_min), abs(out_max))
        self._integral = 0.0
        self._prev_error = 0.0
        self._last_time = None

    def compute(self, setpoint: float, measurement: float) -> float:
        """
        Computes the PID output.

        Args:
            setpoint (float): The desired value.
            measurement (float): The current measured value.

        Returns:
            float: The calculated control output, clamped to out_min/out_max.
        """

        error = setpoint - measurement

        # Proportional Term
        P = self.kp * error

        # Time Delta
        current_time = time.perf_counter()

        if self._last_time is None:
            dt = 1e-16  # Initial value
        else:
            dt = current_time - self._last_time

        self._last_time = current_time

        # Integral Term (with anti-windup)
        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        I = self.ki * self._integral

        # Derivative Term
        derivative_var = error  # Or (-)measurement
        derivative = (derivative_var - self._prev_error) / dt
        D = self.kd * derivative

        # Update state for next iteration
        self._prev_error = derivative_var

        # Total Output
        output = P + I + D

        return max(self.out_min, min(self.out_max, output))

    def reset(self):
        """Resets the integral and derivative states of the controller."""

        self._integral = 0.0
        self._prev_error = 0.0
        self._last_time = None
