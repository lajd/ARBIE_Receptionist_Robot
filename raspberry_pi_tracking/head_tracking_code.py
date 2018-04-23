#!/usr/bin/python
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_StepperMotor
import time
import cv2
import cv2.cv as cv
import sys
import numpy as np
import imutils
import atexit
import threading
import random
import socket
import sys


global steps_from_ref_horizontal, steps_from_ref_vertical, max_vert, max_horiz
steps_from_ref_horizontal = 0
steps_from_ref_vertical = 0
max_vert = 25
max_horiz = 50

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

mh = Adafruit_MotorHAT()
global st1
global st2
global sleep_time
sleep_time = 0.005
st1 = threading.Thread()
st2 = threading.Thread()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind to port
server_address = ('10.42.0.42', 10000)
sock.bind(server_address)

# Listen to Jon
sock.listen(1)

def turnOffMotors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)


atexit.register(turnOffMotors)

myStepper1 = mh.getStepper(200, 1)  # 200 steps/rev, motor port
myStepper2 = mh.getStepper(200, 2)  # 200 steps/rev, motor port #1
myStepper1.setSpeed(60)  # 7.5 RPM? Used to say 60 but unsure if that was true because speed was set to 30
myStepper2.setSpeed(60)  # 7.5 RPM?

stepstyles = [Adafruit_MotorHAT.SINGLE, Adafruit_MotorHAT.DOUBLE, Adafruit_MotorHAT.INTERLEAVE,
              Adafruit_MotorHAT.MICROSTEP]


def stepper_worker(stepper, numsteps, direction, style):
    # print("Steppin!")
    stepper.step(numsteps, direction, style)
    # print("Done")


x1 = 640 / 2
y1 = 480 / 2

while True:
    connection, client_address = sock.accept()
    print 'Socket has been accepted'

    try:
        in_data = connection.recv(1024)
        # data = [float(x) for x in data1.split(" ")]
        # Might be constantly accepting. Limit to 4 points?
        xy = in_data.split("$")

        x1, y1 = xy[0].split(" ")
        x1, y1 = int(x1), int(y1)

        #### TRACKING
        def rightmov():
            global st1
            if not st1.isAlive():
                dire = Adafruit_MotorHAT.BACKWARD
                numsteps = int(((x1 - 320) / 640.0) * 110 * (5.0 / 9.0))
                global steps_from_ref_horizontal
                if steps_from_ref_horizontal + numsteps >= max_horiz:
                    numsteps = max_horiz - steps_from_ref_horizontal
                    steps_from_ref_horizontal = max_horiz
                else:
                    steps_from_ref_horizontal += numsteps
                st1 = threading.Thread(target=stepper_worker, args=(myStepper1, numsteps, dire, stepstyles[3],))
                st1.start()
		time.sleep(sleep_time*numsteps)

        def leftmov():
            global st1
            if not st1.isAlive():
                dire = Adafruit_MotorHAT.FORWARD
                numsteps = int(((320 - x1) / 640.0) * 110 * (5.0 / 9.0))
                global steps_from_ref_horizontal
                if steps_from_ref_horizontal - numsteps <= -max_horiz:
                    numsteps = max_horiz - abs(steps_from_ref_horizontal)
                    steps_from_ref_horizontal = -max_horiz
                else:
                    steps_from_ref_horizontal -= numsteps
                st1 = threading.Thread(target=stepper_worker, args=(myStepper1, numsteps, dire, stepstyles[3],))
                st1.start()
		time.sleep(sleep_time*numsteps)

        def upmov():
            global st2
            if not st2.isAlive():
                dire = Adafruit_MotorHAT.FORWARD
                numsteps = int(((240 - y1) / 480.0) * 82 * (5.0 / 9.0))
                global steps_from_ref_vertical
                if steps_from_ref_vertical + numsteps >= max_vert:
                    numsteps = max_vert - steps_from_ref_vertical
                    steps_from_ref_vertical = max_vert
                else:
                    # Check if numsteps takes us outside max range
                    steps_from_ref_vertical += numsteps
                st2 = threading.Thread(target=stepper_worker, args=(myStepper2, numsteps, dire, stepstyles[3],))
                st2.start()
		time.sleep(sleep_time*numsteps)

        def downmov():
            global st2
            if not st2.isAlive():
                dire = Adafruit_MotorHAT.BACKWARD
                numsteps = int(((y1 - 240) / 480.0) * 82 * (5.0 / 9.0))
                global steps_from_ref_vertical
                if steps_from_ref_vertical - numsteps <= -max_vert:
                    numsteps = max_vert - abs(steps_from_ref_vertical)
                    steps_from_ref_vertical = -max_vert
                else:
                    # Check if numsteps takes us outside max range
                    steps_from_ref_vertical -= numsteps
                    st2 = threading.Thread(target=stepper_worker, args=(myStepper2, numsteps, dire, stepstyles[3],))
                st2.start()
		time.sleep(sleep_time*numsteps)

	xbig = 370
	xsmall = 270
	ybig = 280
	ysmall = 200


        if ((y1 < ysmall) & (x1 > xbig)):
            upmov()
            rightmov()
            #time.sleep(sleep_time)
        elif ((y1 > ybig) & (x1 > xbig)):
            downmov()
            rightmov()
            #time.sleep(sleep_time)
        elif ((y1 < ysmall) & (x1 < xsmall)):
            upmov()
            leftmov()
            #time.sleep(sleep_time)
        elif ((y1 > ybig) & (x1 < xsmall)):
            downmov()
            leftmov()
            #time.sleep(sleep_time)
        elif (y1 < ysmall):
            upmov()
            #time.sleep(sleep_time)
        elif (y1 > ybig):
            downmov()
            #time.sleep(sleep_time)
        elif (x1 > xbig):
            rightmov()
            #time.sleep(sleep_time)
        elif (x1 < xsmall):
            leftmov()
            #time.sleep(sleep_time)
        else:
            # pass
            time.sleep(sleep_time)

    except Exception as e:
        print e

    finally:
        connection.close()
