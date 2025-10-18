"""
OpenCV Lane Detection Module

This module provides a LaneDetector class to process video frames and identify lane
lines on a road. The final output is an annotated image and a steering value,
which represents a weighted sum of deviation from the lane center and its angle.

Detection Pipeline:
1.  Color filtering in HLS space to isolate white/yellow lane markings.
2.  Conversion to grayscale and applying the color mask.
3.  Gaussian blur to reduce noise.
4.  Canny edge detection to find sharp gradients.
5.  Applying a region of interest (ROI) mask to focus on the road ahead.
6.  Hough line transformation to find line segments in the ROI.
7.  Averaging and extrapolating line segments to form single left and right lanes.
8.  Calculating a steering value based on lane position and orientation.
"""

import cv2 as cv
import numpy as np
import os
import time


class LaneDetector:
    """Detects road lanes in a video frame and calculates steering value."""

    def __init__(self, frame_size: tuple = (480, 640)):
        """
        Initializes the lane detector.

        Args:
            frame_size (tuple): The (height, width) of the video frames.
        """

        self.frame_height, self.frame_width = frame_size
        self.avg_steering1 = 0.0  # Based on lane center position
        self.avg_steering2 = 0.0  # Based on lane angle
        self.tot_steering = 0.0
        self.avg_lane_width = self.frame_width / 5  # Initial guess for lane width

        # Tunable Parameters
        self.roi_vert_frac = 0.60  # Consider bottom N% of the frame
        self.canny_low, self.canny_high = 80, 180  # Canny edge detection thresholds
        self.hough_rho, self.hough_theta = 2, np.pi / 180  # Distance and angle resolution
        self.hough_thresh, self.min_len, self.max_gap = 50, 20, 40  # Votes, Min Length, Max Gap
        self.hls_low = np.array([0, 165, 0], dtype=np.uint8)
        self.hls_high = np.array([255, 255, 60], dtype=np.uint8)
        # NOTE: A lightness range of (165-255) is used to detect bight colors. To detect dark
        # colors use (0-90) for the lightness thresholds.

        # Pre-calculate the Region of Interest mask
        self._roi_mask = self._create_roi_mask()

        # Check OS to determine if program is running on a Raspberry Pi (posix)
        self.is_pi = os.name == 'posix'

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Performs the full lane detection pipeline on a single frame.

        Args:
            frame (np.ndarray): The input image frame (in BGR or RGB format).

        Returns:
            tuple: Containing:
                   - annotated_frame (np.ndarray): The original frame with detected lines.
                   - steering_error (float): The steering value from -1.0 (left) to 1.0 (right).
        """

        # Image Pre-processing
        if self.is_pi:
            hls = cv.cvtColor(frame, cv.COLOR_RGB2HLS)
        else:
            hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)

        color_mask = cv.inRange(hls, self.hls_low, self.hls_high)

        if self.is_pi:
            gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        else:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        filtered_gray = cv.bitwise_and(gray, gray, mask=color_mask)
        blur = cv.GaussianBlur(filtered_gray, (5, 5), 0)
        edges = cv.Canny(blur, self.canny_low, self.canny_high)
        edges_roi = cv.bitwise_and(edges, self._roi_mask)

        # Line Detection (Hough Transform)
        lines = cv.HoughLinesP(
            edges_roi, self.hough_rho, self.hough_theta, self.hough_thresh,
            minLineLength=self.min_len, maxLineGap=self.max_gap
        )

        # Line Averaging and Extrapolation
        left_line, right_line = self._average_lines(lines)
        left_pts = self._extrapolate_line(left_line)
        right_pts = self._extrapolate_line(right_line)

        # Steering Calculation
        steering1 = 0.0
        steering2 = 0.0
        lane_width = self.frame_width / 5
        frame_center_x = self.frame_width / 2

        if left_pts is not None and right_pts is not None:
            # Both lanes detected: The most reliable case
            lane_width = right_pts[2] - left_pts[2]  # Lane width at top of ROI
            lane_center_x = (left_pts[2] + right_pts[2]) / 2  # Lane center at top of ROI

            steering1 = lane_center_x - frame_center_x
            steering1 = (max(-self.frame_width, min(self.frame_width, steering1)) /
                         self.frame_width)
            steering2 = -np.arctan(2 / (1 / left_line[0] + 1 / right_line[0])) / (np.pi / 2)
            steering2 = -np.sign(steering2) * (np.abs(steering2) - 1)  # Remap arctan output
        elif left_pts is not None:
            # Only left lane detected: Estimate right lane position.
            steering1 = (left_pts[2] + self.avg_lane_width / 2) - frame_center_x
            steering1 = (max(-self.frame_width, min(self.frame_width, steering1)) /
                         self.frame_width)
            steering2 = -np.arctan(left_line[0]) / (np.pi / 2)
            steering2 = -np.sign(steering2) * (np.abs(steering2) - 1) / 2
        elif right_pts is not None:
            # Only right lane detected: Estimate left lane position.
            steering1 = (right_pts[2] - self.avg_lane_width / 2) - frame_center_x
            steering1 = (max(-self.frame_width, min(self.frame_width, steering1)) /
                         self.frame_width)
            steering2 = -np.arctan(right_line[0]) / (np.pi / 2)
            steering2 = -np.sign(steering2) * (np.abs(steering2) - 1) / 2
        else:
            # No lanes detected: Fallback to the last known average steering.
            steering1 = self.avg_steering1
            steering2 = self.avg_steering2

        self.avg_steering1 = (0.95 * self.avg_steering1 + 0.05 * steering1)
        self.avg_steering2 = (0.95 * self.avg_steering2 + 0.05 * steering2)
        self.tot_steering = self.avg_steering1 * 0.8 + self.avg_steering2 * 0.2
        self.avg_lane_width = self.avg_lane_width * 0.9 + lane_width * 0.1

        # Annotation
        # if lines is not None:
        #     for line in lines:  # Hough lines in red
        #         x1, y1, x2, y2 = line[0]
        #         cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if left_pts is not None:  # Left line
            cv.line(frame, (left_pts[0], left_pts[1]), (left_pts[2], left_pts[3]),
                    (0, 255, 0), 3)
        if right_pts is not None:  # Right line
            cv.line(frame, (right_pts[0], right_pts[1]), (right_pts[2], right_pts[3]),
                    (0, 255, 0), 3)

        return frame, self.tot_steering

    def _create_roi_mask(self) -> np.ndarray:
        """
        Creates a trapezoidal mask to define the region of interest.

        Returns:
            np.ndarray: A binary mask array of the same size as the frame.
        """

        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        roi_top = int(self.frame_height * (1 - self.roi_vert_frac))

        # Four vertices of the trapezoid
        pts = np.array([
            [0, self.frame_height],
            [0, roi_top],
            [self.frame_width, roi_top],
            [self.frame_width, self.frame_height]
        ], np.int32)
        cv.fillPoly(mask, [pts], 255)

        return mask

    def _extrapolate_line(self, line: tuple) -> np.ndarray | None:
        """
        Extrapolates a line defined by slope and intercept to the ROI boundaries.

        Args:
            line (tuple): The (slope, intercept) of the line.

        Returns:
            np.ndarray: An array [x1, y1, x2, y2] for the extrapolated line, or None.
        """

        if line is None:
            return None

        slope, intercept = line
        y1 = self.frame_height  # Bottom of the frame
        y2 = int(y1 * (1 - self.roi_vert_frac))  # Top of the ROI

        if abs(slope) < 1e-6:  # Avoid division by zero for near-horizontal lines
            return None

        # Calculate x coordinates
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return np.array([x1, y1, x2, y2])

    @staticmethod
    def _average_lines(lines: np.ndarray) -> tuple:
        """
        Averages detected line segments into a single left and right lane line.

        Args:
            lines (np.ndarray): Array of lines from Hough transform.

        Returns:
            tuple: A tuple containing two lines (left_line, right_line) defined as (slope,
            intercept). Returns None if not found.
        """

        left_lines, right_lines = [], []

        if lines is None:
            return None, None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2: continue  # Skip vertical lines

            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)  # Fit a line to the points
            if abs(slope) < 0.25: continue  # Skip near-horizontal lines

            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

        # Average the slope and intercept of all lines found for each side
        left_line = np.average(left_lines, axis=0) if left_lines else None
        right_line = np.average(right_lines, axis=0) if right_lines else None

        return left_line, right_line


def main():
    """Main function to run the video processing loop."""

    FRAME_HEIGHT, FRAME_WIDTH = 480, 640

    if os.name == 'posix':
        # Initialize, Configure and Start PiCamera2
        from picamera2 import Picamera2

        picam = Picamera2()
        cfg = picam.create_video_configuration(
            main={'size': (FRAME_WIDTH, FRAME_HEIGHT), 'format': 'RGB888'},
            controls={'FrameDurationLimits': (33333, 33333)})  # 30 fps
        picam.configure(cfg)
        picam.start()
    else:
        # Initialize and Configure Non-Pi Camera
        cap = cv.VideoCapture(0)  # Camera or path to a video file
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)

    lane_detector = LaneDetector(frame_size=(FRAME_HEIGHT, FRAME_WIDTH))
    fps = 0.0
    print("Lane detection running... (Press 'q' to quit)")

    while True:
        current_time = time.perf_counter()

        if os.name == 'posix':
            frame = picam.capture_array()
        else:
            ret, frame = cap.read()
            if not ret: break

        # frame = cv.flip(frame, 0)  # 0: Vertical, 1: Horizontal, -1: Vertical and Horizontal
        result, steering = lane_detector.detect(frame)

        # Textual Annotation
        fps = 0.9 * fps + 0.1 / (time.perf_counter() - current_time)
        cv.putText(result, f"FPS: {int(fps)}", (10, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (128, 80, 80), 2)
        cv.putText(result, f"Steering Cmd: {steering:+.3f}", (10, 35),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (80, 128, 80), 2)
        cv.imshow("CV Lane Detection", result)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'): break

    # Cleanup
    cv.destroyAllWindows()

    if os.name == 'posix':
        picam.stop()
    else:
        cap.release()


if __name__ == '__main__': main()
