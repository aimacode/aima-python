from ev3dev.auto import OUTPUT_A, OUTPUT_B, OUTPUT_D, LargeMotor, MediumMotor, InfraredSensor
from ev3dev.auto import INPUT_1, INPUT_2, ColorSensor, UltrasonicSensor
import time
import ev3dev.auto as auto
import ev3dev.ev3 as ev3

uss = UltrasonicSensor(INPUT_1)
colorSensor = ColorSensor(INPUT_2)
clawMotor = MediumMotor(OUTPUT_B)
leftTire = LargeMotor(OUTPUT_A)# and LargeMotor(OUTPUT_D)
rightTire = LargeMotor(OUTPUT_D)


rightTire.run_timed(speed_sp=720, time_sp=600)
time.sleep(1)