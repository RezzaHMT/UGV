"""
PID Controller Module

This module provides a simple PID (Proportional-Integral-Derivative)
controller class.
"""


class PID:
    """A controller that calculates an output to correct an error over time."""

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0,
                 dt: float = 0.02, out_min: float = -1.0, out_max: float = 1.0,
                 integral_limit: float = None):
        """
        Initializes the PID controller.

        Args:
            kp (float): The Proportional gain.
            ki (float): The Integral gain.
            kd (float): The Derivative gain.
            dt (float): The time delta between compute calls (sample time).
            out_min (float): The minimum value for the controller's output.
            out_max (float): The maximum value for the controller's output.
            integral_limit (float, optional): The anti-windup limit for the integral term.
                                              Defaults to out_max.
        """

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.out_min = out_min
        self.out_max = out_max
        self.integral_limit = integral_limit or abs(out_max)
        self._integral = 0.0
        self._prev_error = 0.0

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

        # Integral Term (with anti-windup)
        self._integral += error * self.dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        I = self.ki * self._integral

        # Derivative Term
        derivative_var = error
        derivative = (derivative_var - self._prev_error) / self.dt
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
