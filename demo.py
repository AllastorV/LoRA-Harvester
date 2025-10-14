"""
Demo script to test Video Smart Cropper
Creates a simple test video with a moving circle (simulates object)
"""

import cv2
import numpy as np
import os
from pathlib import Path


def create_test_video(output_path='test_video.mp4', duration=10, fps=30):
    """
    Create a test video with a moving circle
    
    Args:
        output_path: Output video path
        duration: Video duration in seconds
        fps: Frames per second
    """
    print(f"📹 Creating test video: {output_path}")
    
    width, height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Calculate circle position (moves across screen)
        progress = frame_num / total_frames
        center_x = int(width * 0.2 + width * 0.6 * progress)
        center_y = height // 2 + int(100 * np.sin(progress * 4 * np.pi))
        
        # Draw moving circle (simulates a person/object)
        radius = 150
        color = (0, 255, 0)  # Green
        cv2.circle(frame, (center_x, center_y), radius, color, -1)
        
        # Add frame number
        cv2.putText(frame, f'Frame: {frame_num}', (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add some "subtitle" text at bottom for some frames
        if frame_num % 90 < 30:  # Add subtitle every 3 seconds for 1 second
            cv2.putText(frame, 'Subtitle Text Here', 
                       (width//2 - 200, height - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_num % 30 == 0:
            print(f"  Progress: {int(progress * 100)}%")
    
    out.release()
    print(f"✅ Test video created: {output_path}")
    print(f"   Duration: {duration}s, Resolution: {width}x{height}, FPS: {fps}")


def run_demo():
    """Run a complete demo"""
    print("="*60)
    print("🎬 VIDEO SMART CROPPER - DEMO")
    print("="*60)
    print()
    
    # Create test video
    test_video_path = 'demo_test_video.mp4'
    
    if not os.path.exists(test_video_path):
        create_test_video(test_video_path, duration=10, fps=30)
    else:
        print(f"✅ Test video already exists: {test_video_path}")
    
    print()
    print("="*60)
    print("📋 DEMO INSTRUCTIONS")
    print("="*60)
    print()
    print("1. Test video created: demo_test_video.mp4")
    print()
    print("2. Run GUI mode:")
    print("   python main.py")
    print("   Then drag & drop the test video")
    print()
    print("3. Or run CLI mode:")
    print(f"   python cli.py {test_video_path} -i 30 -f 9:16")
    print()
    print("4. Check output folder for cropped frames")
    print()
    print("="*60)
    print()
    print("💡 TIP: The test video has:")
    print("   • A moving green circle (simulates object)")
    print("   • Periodic subtitles (to test text detection)")
    print("   • Frame numbers for tracking")
    print()
    print("="*60)
    print()
    print("🚀 Ready to test! Choose GUI or CLI mode above.")
    print()


if __name__ == "__main__":
    run_demo()
