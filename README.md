# Air Gestures

Control your mouse pointer by waving your hand with Air Gestures!
Exactly like it sounds. It uses the webcam video feed to track your hand and detects landmark points. This is used to detect gestures and automatically control the pointer on your screen. 

Kind of inspired by all the AR/VR headsets.
But initial developments started before Nov 2022.
Commiting to a public repo now after the recent launch of another AR/VR headset from a big company.

**NOTE**: Since this is in experimental stage, it is buggy and the code is not at all structured. Refactoring of the code and optimizations will be done in future updates.

## Features

- Pointer follows tip of index finger. (Absolute position of scaled video feed.)
- Bringing tip of index and tip of thumb is MouseDown. (However, experimenting with MouseDown and LeftClick.)
- Moving tips of index and thumb is MouseUp.
- Drag is tips together + move.
- Live plotting of coordinates of index tip. (Saves to a csv file. Separate process picks up the values and plots.  By default, code for plotting is commented out. Looking for a better to plot rather than saving to a file.)

### Upcoming Features

- Relative pointer following.
- Static hand gestures (plans to experiment with the basic static gestures provided by MediaPipe)
- **Dynamic hand gestures** (this is where the actual fun begins! - swipes, pinch-to-zoom, scrolls, and all other 2D motion gestures in a smartphone.)
- 3D Dynamic gestures (more fun!!)
- **Expand to** eye tracking, voice inputs (in combination with hand gestures - even more fun :D) 

## Libraies 

- OpenCV (camera feed and visualization)
- MediaPipe (uses TensorFlow Lite - uses CPU only or GPU also if available)
- PyAutoGUI (for mouse control)
- Matplotlib (live plotting - optional)

please suggest a better name for the project. and new features. thanks!