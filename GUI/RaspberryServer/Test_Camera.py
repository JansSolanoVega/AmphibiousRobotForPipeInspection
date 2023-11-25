from picamera2.encoders import H264Encoder
from picamera2 import Picamera2
import time
picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)
encoder = H264Encoder(bitrate=10000000)
output = "test.h264"
picam2.start_recording(encoder, output)
time.sleep(10)
picam2.stop_recording()
#from picamera2 import Picamera2
#import time
#import cv2

# Create a PiCamera object
#camera = Picamera2()

# Set the resolution of the camera
#camera.resolution = (640, 480)

# Start previewing the camera feed
#camera.start()

#try:
    # Create an OpenCV window
    #cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

    # Continuously capture frames
    #while True:
        # Capture an image and save it
        #filename = f"frame.jpg"
        #camera.capture_file(filename)

        # Read the captured frame using OpenCV
        #frame = cv2.imread(filename)

        # Display the frame in the OpenCV window
        #cv2.imshow("Camera Feed", frame)

        # Wait for a short duration before capturing the next frame
        #time.sleep(0.1)

        # Check if the 'q' key is pressed to exit the loop
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

#finally:
    # Close the OpenCV window
    #cv2.destroyAllWindows()

    # Stop previewing and close the camera
    #camera.stop_preview()
    #camera.close()
