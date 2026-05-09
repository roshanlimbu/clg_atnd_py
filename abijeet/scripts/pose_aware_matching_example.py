"""
========================================================
pose_aware_matching_example.py — Usage Examples
========================================================
Demonstrates how to use the new head pose detection and
pose-aware matching features in your code.
========================================================
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from camera import CameraFeed
from detector import FaceDetector
from face_identity import FaceIdentityManager
from database import DatabaseManager
from head_pose_detector import HeadPoseDetector, HeadPose
from enrollment_mode import EnrollmentMode


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 1: Basic Profile Detection in Live Attendance
# ─────────────────────────────────────────────────────────────────────────────

def example_live_attendance_with_profiles():
    """
    Shows how to detect and recognize faces at any angle during live attendance.
    """
    db = DatabaseManager(Path("attendance.db"))
    detector = FaceDetector()
    identity_manager = FaceIdentityManager(db)
    
    camera = CameraFeed(camera_index=0)
    
    print("Starting live attendance with profile detection...")
    print("Press 'Q' to quit\n")

    frame_count = 0
    while camera.is_running:
        frame = camera.read()
        if frame is None:
            continue

        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame
            continue

        # Detect all faces
        detected_faces = detector.detect_faces(frame)
        if not detected_faces:
            continue

        # Identify each face with pose detection
        for face in detected_faces:
            result = identity_manager.identify(face.aligned_image)

            # NEW: Access pose information
            if result.head_pose:
                print(
                    f"Person: {result.person_id} | "
                    f"Pose: {result.head_pose.value} | "
                    f"Confidence: {result.confidence:.2%}"
                )
                
                # Log pose angles if available
                if result.pose_angles:
                    print(
                        f"  Angles - Yaw: {result.pose_angles['yaw']:.1f}° | "
                        f"Pitch: {result.pose_angles['pitch']:.1f}° | "
                        f"Roll: {result.pose_angles['roll']:.1f}°"
                    )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 2: Multi-Angle Enrollment
# ─────────────────────────────────────────────────────────────────────────────

def example_multi_angle_enrollment():
    """
    Shows how to use enrollment mode to capture faces from multiple angles.
    """
    db = DatabaseManager(Path("attendance.db"))
    detector = FaceDetector()
    identity_manager = FaceIdentityManager(db)
    
    camera = CameraFeed(camera_index=0)
    enrollment = EnrollmentMode(output_dir=Path("enrollment_captures"))
    
    # Start enrollment
    person_name = input("Enter person's name: ")
    enrollment_session = enrollment.start_enrollment(person_name)
    
    print(f"\nStarting enrollment for: {person_name}")
    print("Instructions:")
    print("  'S' - Capture current frame for current pose")
    print("  'N' - Move to next pose")
    print("  'Q' - Cancel enrollment")
    print()

    pose_index = 0
    poses = EnrollmentMode.REQUIRED_POSES

    while camera.is_running:
        frame = camera.read()
        if frame is None:
            continue

        # Detect faces
        detected_faces = detector.detect_faces(frame)
        if not detected_faces:
            # Draw "no face detected" message
            cv2.putText(
                frame,
                "No face detected - Please look at camera",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Enrollment Mode", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                enrollment.cancel_enrollment()
                break
            continue

        # Process first detected face
        face = detected_faces[0]
        
        # Identify and get pose
        result = identity_manager.identify(face.aligned_image)
        current_pose = poses[pose_index]

        # Check if pose matches what we're looking for
        if result.head_pose and result.head_pose.value == current_pose:
            status_color = (0, 255, 0)  # Green if pose matches
        else:
            status_color = (0, 165, 255)  # Orange if waiting for pose

        # Draw pose instruction
        cv2.putText(
            frame,
            f"Current Pose: {current_pose.upper()}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            status_color,
            2,
        )

        # Draw progress
        progress = enrollment.get_enrollment_progress()
        poses_complete = sum(
            1 for p in progress["pose_status"].values() if p["complete"]
        )
        cv2.putText(
            frame,
            f"Poses: {poses_complete}/{len(poses)} | "
            f"Frames for {current_pose}: "
            f"{progress['pose_status'][current_pose]['frames']}/3",
            (50, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
        )

        cv2.imshow("Enrollment Mode", frame)

        # Handle user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            enrollment.cancel_enrollment()
            break
        elif key == ord('s'):
            # Capture this frame
            status = enrollment.add_frame(
                face_image=face.aligned_image,
                pose_label=current_pose,
                confidence=result.confidence,
                pose_angles=result.pose_angles,
            )
            print(f"Frame captured: {status}")
        elif key == ord('n'):
            # Move to next pose
            if pose_index < len(poses) - 1:
                pose_index += 1
                print(f"Moving to next pose: {poses[pose_index]}")

    cv2.destroyAllWindows()
    camera.release()

    # Finish enrollment if all poses complete
    result = enrollment.get_enrollment_progress()
    if result["is_complete"]:
        final_data = enrollment.finish_enrollment()
        print(f"\nEnrollment complete! {len(final_data['enrollment_data']['poses'])} poses captured.")
    else:
        print("\nEnrollment cancelled or incomplete.")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 3: Pose-Based Threshold Analysis
# ─────────────────────────────────────────────────────────────────────────────

def example_pose_threshold_analysis():
    """
    Shows how pose affects matching thresholds.
    """
    from head_pose_detector import get_pose_adjusted_threshold

    print("Pose-Based Matching Threshold Analysis")
    print("=" * 60)

    pose_pairs = [
        (HeadPose.FRONT, HeadPose.FRONT),
        (HeadPose.FRONT, HeadPose.LEFT),
        (HeadPose.FRONT, HeadPose.LEFT_PROFILE),
        (HeadPose.FRONT, HeadPose.RIGHT_PROFILE),
        (HeadPose.LEFT, HeadPose.LEFT_PROFILE),
        (HeadPose.LEFT, HeadPose.RIGHT),
        (HeadPose.UP, HeadPose.DOWN),
    ]

    for source, target in pose_pairs:
        threshold = get_pose_adjusted_threshold(source, target)
        print(f"{source.value:15} → {target.value:15} : {threshold:.3f}")

    print("\nInterpretation:")
    print("- Lower threshold (0.60): Stricter matching for similar poses")
    print("- Higher threshold (0.75): Lenient matching for very different poses")
    print("- Allows recognizing same person from any angle!")


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE 4: Standalone Head Pose Detection
# ─────────────────────────────────────────────────────────────────────────────

def example_head_pose_detection_only():
    """
    Shows how to use head pose detection independently.
    """
    detector = HeadPoseDetector()
    
    # Load a face image
    face_img = cv2.imread("test_face.jpg")
    if face_img is None:
        print("Test image not found. Using camera instead.")
        camera = CameraFeed()
        face_detector = FaceDetector()
        
        while camera.is_running:
            frame = camera.read()
            if frame is None:
                continue

            detected_faces = face_detector.detect_faces(frame)
            if not detected_faces:
                continue

            face = detected_faces[0]
            pose = detector.estimate_pose(face.aligned_image)

            if pose:
                print(f"Pose: {pose.pose_label.value}")
                print(f"  Yaw:   {pose.yaw:6.1f}°  (left-right)")
                print(f"  Pitch: {pose.pitch:6.1f}°  (up-down)")
                print(f"  Roll:  {pose.roll:6.1f}°  (tilt)")
                print()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
    else:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pose = detector.estimate_pose(face_rgb)

        if pose:
            print(f"Detected Pose: {pose.pose_label.value}")
            print(f"Yaw (left-right): {pose.yaw:.1f}°")
            print(f"Pitch (up-down):  {pose.pitch:.1f}°")
            print(f"Roll (tilt):      {pose.roll:.1f}°")
        else:
            print("Could not detect pose")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nAdvanced Profile Detection Examples")
    print("=" * 60)
    print("1. Live Attendance with Profile Detection")
    print("2. Multi-Angle Enrollment")
    print("3. Pose Threshold Analysis")
    print("4. Head Pose Detection (Standalone)")
    print("5. Exit")
    print()

    choice = input("Select example (1-5): ").strip()

    if choice == "1":
        example_live_attendance_with_profiles()
    elif choice == "2":
        example_multi_angle_enrollment()
    elif choice == "3":
        example_pose_threshold_analysis()
    elif choice == "4":
        example_head_pose_detection_only()
    elif choice == "5":
        print("Exiting...")
    else:
        print("Invalid choice")
