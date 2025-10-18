"""
Manual Keyboard Control Module

This script provides a KeyboardController class that uses the 'pynput' library to listen for key
presses and translate them into steering and throttle commands.

Controls:
- Throttle: Keys '1' through '5' for increasing speed
- Reverse: Hold 'space' while pressing a throttle key
- Steering: 'h' (hard left), 'j' (soft left), 'k' (soft right), 'l' (hard right)
- Quit: 'q'
"""

from motor_drive import Servo, DC
from pynput import keyboard
import pigpio
import threading
import time


class KeyboardController:
    """Manages keyboard input for vehicle control."""

    def __init__(self, max_steering: float = 28.0, max_throttle: int = 100):
        """
        Initializes state variables and starts the keyboard listener.

        Args:
            max_steering (float): Max steering angle from center.
            max_throttle (int): Max throttle as PWM duty-cycle percentage.
        """

        self._throttle = 0.0  # Range: -1.0 to +1.0
        self._steering = 0.0  # Range: -1.0 to +1.0
        self._max_steering = max_steering
        self._max_throttle = max_throttle
        self._space_pressed = False
        self._pressed_keys = set()
        self.running = True

        # Start the listener in a non-daemon thread
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        """Handles key press events."""

        try:
            key_char = key.char
            if key_char is None: return
        except AttributeError:
            if key == keyboard.Key.space:
                self._space_pressed = True
                if self._throttle != 0.0: self._throttle = -abs(self._throttle)

            return

        self._pressed_keys.add(key_char)

        # Throttle Keys
        if key_char in '12345':
            magnitude = int(key_char) * 0.2  # Mapping [1, ..., 5] to [0.2, ..., 1.0]
            self._throttle = -magnitude if self._space_pressed else magnitude

        # Steering Keys
        steering_lut = {'h': -1.0, 'j': -0.5, 'k': 0.5, 'l': 1.0}

        if key_char in steering_lut:
            self._steering = steering_lut[key_char]

        # Quit Key
        if key_char == 'q':
            self.running = False
            self.stop()  # Ensure listener stops

    def _on_release(self, key):
        """Handles key release events."""

        try:
            key_char = key.char
            if key_char is None: return
        except AttributeError:
            if key == keyboard.Key.space:
                self._space_pressed = False
                if self._throttle != 0.0: self._throttle = abs(self._throttle)

            return

        if key_char in self._pressed_keys:
            self._pressed_keys.remove(key_char)

        # Throttle Keys: Reset throttle only if no other throttle keys are pressed
        if key_char in '12345' and not any(d in self._pressed_keys for d in '12345'):
            self._throttle = 0.0

        # Steering Keys: Reset steering only if no other steering keys are pressed
        if key_char in 'hjkl' and not any(s in self._pressed_keys for s in 'hjkl'):
            self._steering = 0.0

    def is_override_active(self) -> bool:
        """
        Checks if any control key is currently being pressed.

        Returns:
            bool: True if throttle or steering keys are active, False otherwise.
        """

        return self._throttle != 0.0 or self._steering != 0.0

    @property
    def steering(self) -> float:
        """Calculates the target servo angle in degrees."""

        return self._steering * self._max_steering

    @property
    def throttle(self) -> int:
        """Calculates the target motor speed as a percentage."""

        return int(self._throttle * self._max_throttle)

    def stop(self):
        """Stops the keyboard listener."""

        if self.listener and self.listener.is_alive():
            self.listener.stop()


def main():
    """Initializes hardware and runs the main control loop."""

    # Pigpio Daemon
    pi = pigpio.pi()

    if not pi.connected:
        print("Could not connect to pigpiod daemon!")
        print("Run 'sudo pigpiod' to start.")

        return

    servo_steering = Servo(pi, 18, 90.0, 200)
    dc_motor_left = DC(pi, 21, 6, 5)
    dc_motor_right = DC(pi, 26, 19, 13)
    keyboard_controller = KeyboardController()
    print("Manual drive running... (Press 'q' to quit)")

    try:
        while keyboard_controller.running:
            # Set servo angle and motor speed based on keyboard input
            servo_steering.set_angle(keyboard_controller.steering)

            if keyboard_controller.throttle == 0.0:
                dc_motor_left.stop()
                dc_motor_right.stop()
            else:
                dc_motor_left.set_speed(keyboard_controller.throttle)
                dc_motor_right.set_speed(keyboard_controller.throttle)

            time.sleep(0.02)  # Loop delay to reduce CPU usage

    except KeyboardInterrupt:
        print("\nInterrupted by user!")
    finally:
        print("\nShutting down...")
        dc_motor_left.cleanup()
        dc_motor_right.cleanup()
        servo_steering.cleanup()
        keyboard_controller.stop()


if __name__ == '__main__':
    drive_thread = threading.Thread(target=main, daemon=True)
    drive_thread.start()
    drive_thread.join()
