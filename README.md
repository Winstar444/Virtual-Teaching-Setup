# Virtual Teaching Board


Real-time finger-tracked virtual whiteboard with an upward-swipe "slide up" pager. Built with OpenCV and MediaPipe.


## Features
- Real-time fingertip tracking (index finger tip) using MediaPipe.
- A tall canvas (virtual whiteboard) that you can write on with your finger.
- Gesture: swipe up to slide the board upward (like moving to next page) — old content remains.
- Save the whole board as an image.
- Smoothening to reduce jitter.

## Install requirements.txt
pip install opencv-python numpy filterpy cvzone scipy(run this in vs code terminal)

## virtual environment 
Create virtual environment in python 3.10 or 3.11 mediapipe doesn't work in the latest version
. https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe(Go to browser with this link install it and open)
. back to vs code click (ctrl+shift+p)
. choose virtual environment option and choose version3.10 option
. ask for permission and to select to install dependency.
. virtual environment created successfully 
. check version of  it with the help pf command ( --version )
. if issue comes then go to ChatGpt for help

## to run
python main.py



