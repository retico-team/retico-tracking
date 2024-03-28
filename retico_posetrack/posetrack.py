# retico
from os import stat
import time

from PIL.Image import Image

# Needed libraries
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import threading
import matplotlib
import retico_core
from collections import deque
from retico_vision.vision import PosePositionsIU, ImageIU

class PoseTrackingModule(retico_core.AbstractModule):
    """

    A pose tracking module using Google MediaPipe pose tracking library

    """
    @staticmethod
    def name():
        return "Google MediaPipe Posetracking"

    @staticmethod
    def description():
        return "A pose tracking module using Google MediaPipe pose tracking library"

    @staticmethod
    def input_ius():
        return [ImageIU]
    
    @staticmethod
    def output_iu():
        return PosePositionsIU

    def __init__(self, debug=False, static_image_mode=False, smooth_landmarks=True, model_complexity=1, enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.7,
                 min_tracking_confidence=0.4, **kwargs):
        """
        
        Intializes the pose tracking module

        Args:
        - debug: boolean to determine if webcam output with annotations is shown
        - static_image_mode: If set to false, the solution treats the input images as a video stream. It will try to detect the most prominent person in the very first images, and 
          upon a successful detection further localizes the pose landmarks. In subsequent images, it then simply tracks those landmarks without invoking another detection until it 
          loses track, on reducing computation and latency. If set to true, person detection runs every input image, ideal for processing a batch of static, possibly unrelated, images. 
          Default to false.
        - model_compleixty: Complexity of the pose landmark model: 0, 1 or 2. Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1.
        - smooth_landmarks: If set to true, the solution filters pose landmarks across different input images to reduce jitter, but ignored if static_image_mode is also set to true. Default to true.
        - enable_segmentation: If set to true, in addition to the pose landmarks the solution also generates the segmentation mask. Default to false.
        - smooth_segmentation: If set to true, the solution filters segmentation masks across different input images to reduce jitter. Ignored if enable_segmentation is false or static_image_mode is true. Default to true.
        - min_detection_confidence: Minimum confidence value ([0.0, 1.0]) from the person-detection model for the detection to be considered successful. Default to 0.5.
        - min_tracking confidence: Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the pose landmarks to be considered tracked successfully, or otherwise 
          person detection will be invoked automatically on the next input image. Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency. 
          Ignored if static_image_mode is true, where person detection simply runs on every image. Default to 0.5.


        """
        super().__init__(**kwargs)
        self.debug = debug
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.queue = deque(maxlen=1)

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_video = self.mp_pose.Pose(static_image_mode=False, model_complexity = self.model_complexity, smooth_landmarks=self.smooth_landmarks, 
                                            enable_segmentation=self.enable_segmentation, smooth_segmentation=self.smooth_segmentation, min_detection_confidence=self.min_detection_confidence, 
                                            min_tracking_confidence=self.min_tracking_confidence)
    def detectPose(self, image, pose, display=True):
        '''
        This function performs pose detection on an image.
        Args:
            image: The input image with a prominent person whose pose landmarks needs to be detected.
            pose: The pose setup function required to perform the pose detection.
            display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                    and the pose landmarks in 3D plot and returns nothing.
        Returns:
            output_image: The input image with the detected pose landmarks drawn.
            landmarks: A list of detected landmarks converted into their original scale.
        '''
        
        # Create a copy of the input image.
        output_image = image.copy()
        
        # Convert the image from BGR into RGB format.
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform the Pose Detection.
        results = pose.process(imageRGB)
        
        # # Retrieve the height and width of the input image.
        # height, width, _ = image.shape
        
        # Initialize a list to store the detected landmarks.
        # landmarks = []
        
        # Check if any landmarks are detected.
        if results.pose_landmarks:
        
            # Draw Pose landmarks on the output image.
            self.mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                    connections=self.mp_pose.POSE_CONNECTIONS)
            
            # # Iterate over the detected landmarks.
            # for landmark in results.pose_landmarks.landmark:
                
            #     # Append the landmark into the list.
            #     landmarks.append((int(landmark.x * width), int(landmark.y * height),
            #                         (landmark.z * width)))
        
        # Check if the original input image and the resultant image are specified to be displayed.
        if display:
        
            # Display the original input image and the resultant image.
            plt.figure(figsize=[22,22])
            plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
            plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
            
            # Also Plot the Pose landmarks in 3D.
            self.mp_drawing.plot_landmarks(results.pose_world_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
        # Otherwise
        else:
            
            # Return the output image and the found landmarks.
            return output_image, results

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            self.queue.append(iu)
        return None

    def run_tracker(self):

        video = cv2.VideoCapture(0)


        # Initialize a variable to store the time of the previous frame.
        # time1 = 0

        # Iterate until the video is accessed successfully.
        while True:
            
            if len(self.queue) == 0:
                time.sleep(0.05)
                continue
            
            input_iu = self.queue.popleft()
            image = input_iu.payload
            frame = np.asarray(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
            
            # Flip the frame horizontally for natural (selfie-view) visualization.
            frame = cv2.flip(frame, 1)
            
            # Get the width and height of the frame
            frame_height, frame_width, _ =  frame.shape
            
            # Resize the frame while keeping the aspect ratio.
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
            
            # Perform Pose landmark detection.
            frame, results = self.detectPose(frame, self.pose_video, display=False)
            print("Results pose landmarks", results.pose_landmarks)
            print("Results pose world landmarks", results.pose_world_landmarks)
            print("Results Segmentation Mask", results.segmentation_mask)
            
            # # Set the time for this frame to the current time.
            # time2 = time()
            
            # # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
            # if (time2 - time1) > 0:
            
            #     # Calculate the number of frames per second.
            #     frames_per_second = 1.0 / (time2 - time1)
                
            #     # Write the calculated number of frames per second on the frame. 
            #     cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            
            # # Update the previous frame time to this frame time.
            # # As this frame will become previous frame in next iteration.
            # time1 = time2
            
            # Display the frame.
            if self.debug:
                cv2.imshow('Pose Detection', frame)
            
            # Wait until a key is pressed.
            # Retreive the ASCII code of the key pressed
            k = cv2.waitKey(1) & 0xFF
            
            # Check if 'ESC' is pressed.
            if(k == 27):
                
                # Break the loop.
                break

            output_iu = self.create_iu(input_iu)
            output_iu.set_landmarks(image, results.pose_landmarks, results.segmentation_mask)
            self.append(retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD))

    def setup(self):
        t = threading.Thread(target=self.run_tracker)
        t.start()

        
