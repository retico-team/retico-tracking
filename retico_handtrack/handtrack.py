# retico
from os import stat
import time

# Needed libraries
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import threading
import queue
import matplotlib
import retico_core
from collections import deque
from retico_vision.vision import HandPositionsIU, ImageIU


class HandTrackingModule(retico_core.AbstractModule):
    """

    A hand tracking module using Google MediaPipe hand tracking library

    """

    @staticmethod
    def name():
        return "Google MediaPipe Handtracking"

    @staticmethod
    def description():
        return "A module for hand tracking using Google MediaPipe hand tracking library"

    @staticmethod
    def input_ius():
        return [ImageIU]

    @staticmethod
    def output_iu():
        return HandPositionsIU

    def __init__(self, side_by_side=False, draw_bounding_box=True, debug=False, static_image_mode=False, max_num_hands=2, model_complexity=1,
                min_detection_confidence=0.7, min_tracking_confidence=0.4, image_dims=(240, 320, 3), **kwargs):
        """ 
        
        Intializes the hand tracking module

        Args: 
        - debug: boolean to determine if webcam output with annotations is shown
        - side_by_side: boolean to determine if output is a sidebyside view of default hand tracking annotation vs custom hand tracking annotation
        - draw_bounding_box: boolean to determine if bounding box with hand classification label (left or right hand) should be drawn on image
        - static_image_mode: If set to false, the solution treats the input images as a video stream. It will try to detect hands in the first input
          images, and upon a successful detection further localizes the hand landmarks. In subsequent images, once all max_num_hands hands are detected
          and the corresponding hand landmarks are localized, it simply tracks those landmarks without invoking another detection until it loses track
          of any of the hands. This reduces latency and is ideal for processing video frames. If set to true, hand detection runs on every input image,
          ideal for processing a batch of static, possibly unrelated, images. Default to false.
        - max_num_hands: Maximum number of hands to detect. Default to 2.
        - model_complexity: Complexity of the hand landmark model: 0 or 1. Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1.
        - min_detection_confidence: Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered successful. Default to 0.5.
        - min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be considered tracked successfully,
         or otherwise hand detection will be invoked automatically on the next input image. Setting it to a higher value can increase robustness of the solution, at the
          expense of a higher latency. Ignored if static_image_mode is true, where hand detection simply runs on every image. Default to 0.5.

        """ 
        super().__init__(**kwargs)
        self.debug = debug
        self.side_by_side = side_by_side
        self.draw_bounding_box = draw_bounding_box
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.count = 0

        self.queue = queue.Queue(maxsize=1)

        self.count = 0

        self.image_dims = image_dims

        # Initialize the mediapipe hands class.
        self.mp_hands = mp.solutions.hands

        # Initialize the mediapipe drawing class.
        self.mp_drawing = mp.solutions.drawing_utils

        # Setup Hands function for video.
        self.hands_video = self.mp_hands.Hands(static_image_mode=False, max_num_hands=self.max_num_hands,
                             min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence)


    def customLandmarksAnnotation(self, image, landmark_dict):
        """
        This function draws customized landmarks annotation utilizing the z-coordinate (depth) values of the hands.
        Args:
            image:         The image of the hands on which customized landmarks annotation of the hands needs to be drawn.
            landmark_dict: The dictionary that stores the hand(s) landmarks as different elements with keys as hand 
                        types(i.e., left and right). 
        Returns:
            output_image: The image of the hands with the customized annotation drawn.
            depth:        A dictionary that contains the average depth of all landmarks of the hand(s) in the image.
        """
        
        # Create a copy of the input image to draw annotation on.
        output_image = image.copy()
        
        # Initialize a dictionary to store the average depth of all landmarks of hand(s).
        depth = {}
        
        # Initialize a list with the arrays of indexes of the landmarks that will make the required 
        # line segments to draw on the hand.
        segments = [np.arange(0,5), np.arange(5,9) , np.arange(9,13), np.arange(13, 17), np.arange(17, 21),
                    np.arange(5,18,4), np.array([0,5]), np.array([0,17])]
        
        # Iterate over the landmarks dictionary.
        for hand_type, hand_landmarks in landmark_dict.items():
            
            # Get all the z-coordinates (depth) of the landmarks of the hand.
            depth_values = np.array(hand_landmarks)[:,-1]
            
            # Calculate the average depth of the hand.
            average_depth = int(sum(depth_values) / len(depth_values))
            
            # Get all the x-coordinates of the landmarks of the hand.
            x_values = np.array(hand_landmarks)[:,0]
            
            # Get all the y-coordinates of the landmarks of the hand.
            y_values = np.array(hand_landmarks)[:,1]
            
            # Initialize a list to store the arrays of x and y coordinates of the line segments for the hand.
            line_segments = []
            
            # Iterate over the arrays of indexes of the landmarks that will make the required line segments.
            for segment_indexes in segments:
                
                # Get an array of a line segment coordinates of the hand.
                line_segment = np.array([[int(x_values[index]), int(y_values[index])] for index in segment_indexes])
                
                # Append the line segment coordinates into the list.
                line_segments.append(line_segment)
            
            # Check if the average depth of the hand is less than 0.
            if average_depth < 0:
                
                # Set the thickness of the line segments of the hand accordingly to the average depth. 
                line_thickness = int(np.ceil(0.1*abs(average_depth))) + 2
                
                # Set the thickness of the circles of the hand landmarks accordingly to the average depth. 
                circle_thickness = int(np.ceil(0.1*abs(average_depth))) + 3
            
            # Otherwise.
            else:
                
                # Set the thickness of the line segments of the hand to 2 (i.e. the minimum thickness we are specifying).
                line_thickness = 2
                
                # Set the thickness of the circles to 3 (i.e. the minimum thickness) 
                circle_thickness = 3
            
            # Draw the line segments on the hand.
            cv2.polylines(output_image, line_segments, False, (100,250,55), line_thickness)
            
            # Write the average depth of the hand on the output image. 
            cv2.putText(output_image,'Depth: {}'.format(average_depth),(10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (20,25,255), 1,
                        cv2.LINE_AA)
            
            # Iterate over the x and y coordinates of the hand landmarks.
            for x, y in zip(x_values, y_values):
                
                # Draw a circle on the x and y coordinate of the hand.
                cv2.circle(output_image,(int(x), int(y)), circle_thickness, (55,55,250), -1)
            
            # Store the calculated average depth in the dictionary.
            depth[hand_type] = average_depth
        
        # Return the output image and the average depth dictionary of the hand(s). 
        return output_image, depth

    def drawBoundingBoxes(self, image, results, hand_status, padd_amount = 10, draw=True, display=True):
        '''
        This function draws bounding boxes around the hands and write their classified types near them.
        Args:
            image:       The image of the hands on which the bounding boxes around the hands needs to be drawn and the 
                        classified hands types labels needs to be written.
            results:     The output of the hands landmarks detection performed on the image on which the bounding boxes needs
                        to be drawn.
            hand_status: The dictionary containing the classification info of both hands. 
            padd_amount: The value that specifies the space inside the bounding box between the hand and the box's borders.
            draw:        A boolean value that is if set to true the function draws bounding boxes and write their classified 
                        types on the output image. 
            display:     A boolean value that is if set to true the function displays the output image and returns nothing.
        Returns:
            output_image:     The image of the hands with the bounding boxes drawn and hands classified types written if it 
                            was specified.
            output_landmarks: The dictionary that stores both (left and right) hands landmarks as different elements.
        '''
        
        # Create a copy of the input image to draw bounding boxes on and write hands types labels.
        output_image = image.copy()
        
        # Initialize a dictionary to store both (left and right) hands landmarks as different elements.
        output_landmarks = {}

        # Get the height and width of the input image.
        height, width, _ = image.shape

        # Iterate over the found hands.
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Initialize a list to store the detected landmarks of the hand.
            landmarks = []

            # Iterate over the detected landmarks of the hand.
            for landmark in hand_landmarks.landmark:

                # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                    (landmark.z * width)))

            # Get all the x-coordinate values from the found landmarks of the hand.
            x_coordinates = np.array(landmarks)[:,0]
            
            # Get all the y-coordinate values from the found landmarks of the hand.
            y_coordinates = np.array(landmarks)[:,1]
            
            # Get the bounding box coordinates for the hand with the specified padding.
            x1  = int(np.min(x_coordinates) - padd_amount)
            y1  = int(np.min(y_coordinates) - padd_amount)
            x2  = int(np.max(x_coordinates) + padd_amount)
            y2  = int(np.max(y_coordinates) + padd_amount)

            # Initialize a variable to store the label of the hand.
            label = "Unknown"
            
            # Check if the hand we are iterating upon is the right one.
            if hand_status['Right_index'] == hand_index:
                
                # Update the label and store the landmarks of the hand in the dictionary. 
                label = 'Right Hand'
                output_landmarks['Right'] = landmarks
            
            # Check if the hand we are iterating upon is the left one.
            elif hand_status['Left_index'] == hand_index:
                
                # Update the label and store the landmarks of the hand in the dictionary. 
                label = 'Left Hand'
                output_landmarks['Left'] = landmarks
            
            # Check if the bounding box and the classified label is specified to be written.
            if draw:
                
                # Draw the bounding box around the hand on the output image.
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8)
                
                # Write the classified label of the hand below the bounding box drawn. 
                cv2.putText(output_image, label, (x1, y2+25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (20,255,155), 1, cv2.LINE_AA)
        
        # Check if the output image is specified to be displayed.
        if display:

            # Display the output image.
            plt.figure(figsize=[10,10])
            plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Otherwise
        else:

            # Return the output image and the landmarks dictionary.
            return output_image, output_landmarks

    def getHandType(self, image, results, draw=True, display = True):
        '''
        This function performs hands type (left or right) classification on hands.
        Args:
            image:   The image of the hands that needs to be classified, with the hands landmarks detection already performed.
            results: The output of the hands landmarks detection performed on the image in which hands types needs 
                    to be classified.
            draw:    A boolean value that is if set to true the function writes the hand type label on the output image. 
            display: A boolean value that is if set to true the function displays the output image and returns nothing.
        Returns:
            output_image: The image of the hands with the classified hand type label written if it was specified.
            hands_status: A dictionary containing classification info of both hands.
        '''
        
        # Create a copy of the input image to write hand type label on.
        output_image = image.copy()
        
        # Initialize a dictionary to store the classification info of both hands.
        hands_status = {'Right': False, 'Left': False, 'Right_index' : None, 'Left_index': None}
        
        # Iterate over the found hands in the image.
        for hand_index, hand_info in enumerate(results.multi_handedness):
            
            # Retrieve the label of the found hand.
            hand_type = hand_info.classification[0].label
            
            # Update the status of the found hand.
            hands_status[hand_type] = True
            
            # Update the index of the found hand.
            hands_status[hand_type + '_index'] = hand_index 
            
            # Check if the hand type label is specified to be written.
            if draw:
            
                # Write the hand type on the output image. 
                cv2.putText(output_image, hand_type + ' Hand Detected', (10, (hand_index+1) * 30),cv2.FONT_HERSHEY_PLAIN,
                            2, (0,255,0), 2)
        
        # Check if the output image is specified to be displayed.
        if display:

            # Display the output image.
            plt.figure(figsize=[10,10])
            plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Otherwise
        else:
            
            # Return the output image and the hands status dictionary that contains classification info.
            return output_image, hands_status

    def detectHandsLandmarks(self, image, hands, display = True):
        '''
        This function performs hands landmarks detection on an image.
        Args:
            image:   The input image with prominent hand(s) whose landmarks needs to be detected.
            hands:   The hands function required to perform the hands landmarks detection.
            display: A boolean value that is if set to true the function displays the original input image, and the output 
                    image with hands landmarks drawn and returns nothing.
        Returns:
            output_image: The input image with the detected hands landmarks drawn.
            results: The output of the hands landmarks detection on the input image.
        '''
        
        # Create a copy of the input image to draw landmarks on.
        output_image = image.copy()
        
        # Convert the image from BGR into RGB format.
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform the Hands Landmarks Detection.
        results = hands.process(imgRGB)
        
        # Check if landmarks are found.
        if results.multi_hand_landmarks:
            
            # Iterate over the found hands.
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Draw the hand landmarks on the copy of the input image.
                self.mp_drawing.draw_landmarks(image = output_image, landmark_list = hand_landmarks,
                                    connections = self.mp_hands.HAND_CONNECTIONS) 
        
        # Check if the original input image and the output image are specified to be displayed.
        if display:
            
            # Display the original input image and the output image.
            plt.figure(figsize=[15,15])
            plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
            plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off');
            
        # Otherwise
        else:
            
            # Return the output image and results of hands landmarks detection.
            return output_image, results

    def process_update(self, update_message):
        # if self.queue.full(): self.queue.get_nowait()
        # self.queue.put_nowait(update_message)
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            if self.queue.full(): self.queue.get_nowait()
            self.queue.put_nowait(iu)
        return None

    def run_tracker(self):
        while True:
           
            # if len(self.queue) == 0:
            #     time.sleep(0.1)
            #     continue
            
            input_iu = self.queue.get()
            image = input_iu.payload # assume PIL image
            # print(image)
            frame = np.asarray(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 

            # Flip the frame horizontally for natural (selfie-view) visualization.
            frame = cv2.flip(frame, 1)
            
            # Perform Hands landmarks detection.
            annotated_frame, results = self.detectHandsLandmarks(frame, self.hands_video, display=False)
            # print("Results multi hand landmarks: ", results.multi_hand_landmarks)
            # print("Results multi hand world landmarks: ", results.multi_hand_world_landmarks)
            # print("Results multi_handedness: ", results.multi_handedness)
            
            # Check if landmarks are found in the frame.
            if results.multi_hand_landmarks:

                # print(results.multi_hand_landmarks)
                
                # self.count += 1
                # Perform hand(s) type (left or right) classification.
                _, hands_status = self.getHandType(frame.copy(), results, draw=False, display=False)
                # print("Hands Status: ", hands_status)
                
                # Get the landmarks dictionary that stores each hand landmarks as different elements. 
                frame, landmark_dict = self.drawBoundingBoxes(frame, results, hands_status, draw=self.draw_bounding_box, display=False)
                # print("Landmark dict: ", landmark_dict)
                # print(self.count)
                # Draw customized landmarks annotation ultilizing the z-coordinate (depth) values of the hand(s).
                custom_ann_frame, depth = self.customLandmarksAnnotation(frame, landmark_dict)
                # print("Depth: ", depth)
                
                if self.side_by_side:
                    # Stack the frame annotated using mediapipe with the customized one.
                    final_output = np.hstack((annotated_frame, custom_ann_frame))
                else:
                    #Only show custom annotated frame
                    final_output = custom_ann_frame
                
            # Otherwise.
            else:
                
                if self.side_by_side:
                    # Stack the frame two time.
                    final_output = np.hstack((frame, frame))
                
                else:
                    #Only show custom annotated frame
                    final_output = frame
            
            # Display the stacked frame.
            if self.debug:
                cv2.imshow('Hands Landmarks Detection', final_output)
            
            # Wait for 1ms. If a a key is pressed, retreive the ASCII code of the key.
            k = cv2.waitKey(1) & 0xFF    
            
            # Check if 'ESC' is pressed and break the loop.
            if(k == 27):
                break
            
            output_iu = self.create_iu(input_iu)
            output_iu.set_landmarks(image, results.multi_hand_landmarks, results.multi_handedness)
            output_iu.payload_to_vector(self.count)
            self.count += 1
            self.append(retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD))


    def setup(self):
        t = threading.Thread(target=self.run_tracker)
        t.start()



    

        

    

