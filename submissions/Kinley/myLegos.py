from ev3dev.auto import OUTPUT_A, OUTPUT_B, OUTPUT_D, LargeMotor, MediumMotor
from ev3dev.auto import INPUT_1, INPUT_2, ColorSensor, UltrasonicSensor
import time
import ev3dev.ev3 as ev3

ultrasonicSensor = UltrasonicSensor(INPUT_1)
colorSensor = ColorSensor(INPUT_2)
clawMotor = MediumMotor(OUTPUT_B)
leftTire = LargeMotor(OUTPUT_A)
rightTire = LargeMotor(OUTPUT_D)

def getUltrasonic():
    ultrasonicSensor.mode='US-DIS-CM'
    return ultrasonicSensor.units

def getColor():
    colorSensor.mode='COL-REFLECT'
    return colorSensor.value()


#def findObject():
while getUltrasonic > 5.5:
    leftTire.run_timed(power=15, rotations=0.2)
    rightTire.run_timed(power=15, rotations=0.2)
    time.sleep(1)
clawMotor.run_timed(power=75, rotations=0.8)
time.sleep(1)

#def findTarget():
while getColor > 15:
    leftTire.run_timed(power=15, rotations=0.2)
    rightTire.run_timed(power=15, rotations=0.2)
    time.sleep(1)
clawMotor.run_timed(power=75, rotations=-0.8)
time.sleep(1)