from ev3dev.auto import OUTPUT_A, OUTPUT_B, OUTPUT_C, LargeMotor, MediumMotor
from ev3dev.auto import INPUT_1, INPUT_2, INPUT_3, TouchSensor, ColorSensor
import time
import ev3dev.ev3 as ev3

baseTouch = TouchSensor(INPUT_1)
armTouch = TouchSensor(INPUT_2)
colorSensor = ColorSensor(INPUT_3)
clawMotor = MediumMotor(OUTPUT_A)
armMotor = LargeMotor(OUTPUT_B)
baseMotor = LargeMotor(OUTPUT_C)

# Claw Opening
ev3.Sound.speak("Now opening claw.").wait()
clawMotor.run_forever(speed_sp = 100)
time.sleep(.5)
clawMotor.stop()

# Arm Moving to down position
ev3.Sound.speak("Now moving arm down.").wait()
armMotor.run_forever(speed_sp = 120)
time.sleep(2.5)
armMotor.stop()

# Claw closing on object to pick up
ev3.Sound.speak("Now closing claw.").wait()
clawMotor.run_forever(speed_sp = -100)
time.sleep(2)

# Arm moving to the up position
ev3.Sound.speak("Now moving arm up.").wait()
armMotor.run_forever(speed_sp = -360)
time.sleep(.85)
armMotor.stop()

# Base moving to center position
ev3.Sound.speak("Now moving base.").wait()
baseMotor.run_forever(speed_sp = -130)
time.sleep(2.25)
baseMotor.stop()

# Claw dropping object
ev3.Sound.speak("Now releasing claw.").wait()
clawMotor.stop()

# Base moving to its starting point
ev3.Sound.speak("Now returning to start point.").wait()
baseMotor.run_forever(speed_sp = 130)
time.sleep(2.25)
baseMotor.stop()

# :)
ev3.Sound.speak("Task complete!").wait()
ev3.Sound.speak("Dr. Hooper, Please Give Us an Aye Plus!").wait()