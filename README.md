# retico-tracking
Contains retico modules for handtracking and posetracking that use the Google MediaPipe libraries. The modules can use a standard webcam or the Mistyrobotics Misty II robot's camera as visual input.

The following example is set up to do handtracking with a webcam. The commented out code can be used to switch the type of tracking and camera.
### Example
```import os, sys
prefix = '/prefix/path/'

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


sys.path.append(prefix+'retico-core')
sys.path.append(prefix+'retico-vision')
sys.path.append(prefix+'retico-mistyrobot')
sys.path.append(prefix+'retico-tracking')

from retico_vision.vision import WebcamModule
from retico_mistyrobot.misty_camera import MistyCameraModule
from retico_handtrack.handtrack import HandTrackingModule
from retico_posetrack.posetrack import PoseTrackingModule
from retico_core.debug import DebugModule


misty_ip = '192.168.0.101' # example robot-specific IP

webcam = WebcamModule(pil=True)
# mistycam = MistyCameraModule(misty_ip)
handtracking = HandTrackingModule(False, True, True, False, 2, 1, 0.7, 0.4)
# posetracking = PoseTrackingModule(debug=True)
debug = DebugModule()

webcam.subscribe(handtracking)
# webcam.subscribe(posetracking)
# mistycam.subscribe(handtracking)
# mistycam.subscribe(posetracking)
handtracking.subscribe(debug)
# posetracking.subscribe(debug)

webcam.run()
# mistycam.run()
handtracking.run()
# posetracking.run()
debug.run()

input()

webcam.stop()
# mistycam.stop()
handtracking.stop()
# posetracking.stop()
debug.stop()
```
