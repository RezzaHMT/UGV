"""
Main Control Program of the UGV

This program integrates the following modules:
- Lane Detection: To identify lanes and calculate steering value.
- PID Controller: To compute steering corrections based on the error.
- Manual Drive: To allow for keyboard-based manual override.
- Motor Drive: To control the physical motors and servos.

The main loop runs the lane detection, feeds the output to the PID controller
to steer the car autonomously. The manual drive runs in a parallel thread and
can override the autonomous controls at any time.

NOTE: This program requires the pigpio daemon to be running.
Start it with: `sudo pigpiod`
"""

from lane_detection import LaneDetector
from manual_drive import KeyboardController
from motor_drive import Servo, DC
from picamera2 import Picamera2
from pid import PID
import cv2 as cv
import pigpio

# --- Configuration Constants ---
# Camera
FRAME_HEIGHT = 480
FRAME_WIDTH = 640

# Motor & Servo GPIO Pins (BCM numbering)
SERVO_PIN = 18
DC_MOTOR_L_EN = 21
DC_MOTOR_L_IN1 = 6
DC_MOTOR_L_IN2 = 5
DC_MOTOR_R_EN = 26
DC_MOTOR_R_IN1 = 19
DC_MOTOR_R_IN2 = 13

# Control Parameters
MAX_STEERING = 28  # Max angle servo is allowed to turn from center (degrees)
MAX_THROTTLE = 80  # Max motor speed as a percentage (0-100)
AUTONOMOUS_THROTTLE = 30  # Fixed speed for autonomous driving
SERVO_CENTER_ANGLE = 90.0  # Calibrated center angle for the servo
SERVO_RATE = 200  # Max rotational speed of the servo in degrees per second

# PID Controller Gains
PID_KP = 2.75  # Proportional gain
PID_KI = 0.05  # Integral gain
PID_KD = 0.10  # Derivative gain


def main():
    """Initializes all components and runs the main control loop."""

    print("Initializing components...")

    # --- Initialize Hardware and Controllers ---
    # Pigpio Daemon
    pi = pigpio.pi()

    if not pi.connected:
        print("Could not connect to pigpiod daemon!")
        print("Run 'sudo pigpiod' to start.")

        return

    # Camera
    picam = Picamera2()
    cfg = picam.create_video_configuration(
        main={'size': (FRAME_WIDTH, FRAME_HEIGHT), 'format': 'RGB888'},
        controls={'FrameDurationLimits': (33333, 33333)}  # 30 fps
    )
    picam.configure(cfg)
    picam.start()
    print("Camera initialized.")

    # Actuators
    servo_steering = Servo(pi, SERVO_PIN, center_angle=SERVO_CENTER_ANGLE, rate=SERVO_RATE)
    dc_motor_left = DC(pi, DC_MOTOR_L_EN, DC_MOTOR_L_IN1, DC_MOTOR_L_IN2)
    dc_motor_right = DC(pi, DC_MOTOR_R_EN, DC_MOTOR_R_IN1, DC_MOTOR_R_IN2)
    print("Actuators initialized.")

    # Controllers
    lane_detector = LaneDetector(frame_size=(FRAME_HEIGHT, FRAME_WIDTH))
    pid_controller = PID(
        kp=PID_KP, ki=PID_KI, kd=PID_KD,
        out_min=-MAX_STEERING,
        out_max=MAX_STEERING,
        integral_limit=5.0
    )
    keyboard_controller = KeyboardController(
        max_steering=MAX_STEERING,
        max_throttle=MAX_THROTTLE
    )
    print("Controllers initialized.")

    print("Initialization complete. Starting main loop... (Press 'q' to quit)")

    try:
        # --- Main Control Loop ---
        while keyboard_controller.running:
            # Capture Frame
            frame = picam.capture_array()
            # frame = cv.flip(frame, 0)  # 0: Vertical, 1: Horizontal, -1: Vertical and Horizontal

            # Lane Detection
            annotated_frame, detected_steering = lane_detector.detect(frame)
            scaled_steering = detected_steering * 90.0

            # Control Logic
            if keyboard_controller.is_override_active():
                # Manual Override
                steering_ang = keyboard_controller.steering
                throttle = keyboard_controller.throttle

                # Reset PID to prevent integral windup while in manual mode
                pid_controller.reset()

            else:
                # Autonomous Mode: Use PID controller to calculate steering angle
                # The setpoint is 0 (the center), measurement is the steering error
                steering_ang = pid_controller.compute(setpoint=0.0,
                                                      measurement=-scaled_steering)
                throttle = AUTONOMOUS_THROTTLE

            # Actuator Control
            servo_steering.set_angle(steering_ang)
            dc_motor_left.set_speed(throttle)
            dc_motor_right.set_speed(throttle)

            # Display Output
            cv.putText(annotated_frame, f"Steering Cmd: {scaled_steering:.3f}", (10, 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (128, 80, 80), 2)
            cv.putText(annotated_frame, f"Steering Angle: {steering_ang:.3f} deg", (10, 35),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (80, 128, 80), 2)
            cv.imshow("Lane Follower", annotated_frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'): break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        # Cleanup
        print("Shutting down...")
        cv.destroyAllWindows()
        picam.stop()
        keyboard_controller.stop()
        dc_motor_left.cleanup()
        dc_motor_right.cleanup()
        servo_steering.cleanup()

        # Disconnect from pigpiod daemon
        pi.stop()
        print("Shutdown complete.")


if __name__ == '__main__': main()
