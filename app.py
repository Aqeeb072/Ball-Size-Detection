import cv2
import numpy as np

# Initialize webcam (0 for default camera)
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define HSV color ranges for different colored balls
# You can adjust these values based on your ball color and lighting conditions

# RED BALL COLOR RANGE
# Red wraps around in HSV, so we need two ranges
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# GREEN BALL COLOR RANGE (uncomment to use)
# lower_green = np.array([40, 50, 50])
# upper_green = np.array([80, 255, 255])

# BLUE BALL COLOR RANGE (uncomment to use)
# lower_blue = np.array([100, 150, 50])
# upper_blue = np.array([130, 255, 255])

print("Ball Detection System Started!")
print("Press 'q' to quit the program")
print("Detecting RED colored ball...")

while True:
    # Capture frame-by-frame from webcam
    ret, frame = cap.read()
    
    # Check if frame was captured successfully
    if not ret:
        print("Error: Failed to capture frame")
        break
    
    # Flip frame horizontally for mirror effect (optional)
    frame = cv2.flip(frame, 1)
    
    # Convert BGR color space to HSV
    # HSV is better for color detection as it separates color info from lighting
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for red color
    # For red, we need to combine two masks due to hue wrapping
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # For other colors, use single mask (example for green):
    # mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to remove noise
    # Erosion removes small white noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    # Dilation enlarges the object area
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours in the mask
    # Contours are curves joining continuous points with same color/intensity
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour found
    if contours:
        # Find the largest contour (assumed to be the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate area of the largest contour
        area = cv2.contourArea(largest_contour)
        
        # Only process if area is above threshold (filters out noise)
        if area > 500:
            # Get minimum enclosing circle around the contour
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            
            # Calculate center point
            center = (int(x), int(y))
            radius = int(radius)
            
            # Only draw if radius is reasonable
            if radius > 10:
                # Draw green circle around the detected ball
                cv2.circle(frame, center, radius, (0, 255, 0), 3)
                
                # Draw red center point
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Display ball coordinates and radius
                text = f"Ball: ({center[0]}, {center[1]}) R:{radius}"
                cv2.putText(frame, text, (center[0] - 50, center[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display "BALL DETECTED" message
                cv2.putText(frame, "BALL DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display instructions on frame
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show the original frame with detection overlay
    cv2.imshow('Ball Detection - Original', frame)
    
    # Show the mask (useful for debugging and calibration)
    cv2.imshow('Ball Detection - Mask', mask)
    
    # Wait for 1ms and check if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
print("Program terminated successfully!")