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


# ev3.Sound.speak("Now moving claw.").wait()
# clawMotor.run_forever(speed_sp = 100)
# time.sleep(.5)
# clawMotor.stop()
#
# ev3.Sound.speak("Now moving arm.").wait()
# armMotor.run_forever(speed_sp = 100)
# time.sleep(2.5)
# armMotor.stop()
#
# clawMotor.run_forever(speed_sp = -100)
# time.sleep(2)
# armMotor.run_forever(speed_sp = -100)
# time.sleep(3)
# armMotor.stop()

ev3.Sound.speak("Now moving base.").wait()
baseMotor.run_forever(speed_sp = -100)
time.sleep(2)
baseMotor.stop()


# clawMotor.stop()

# armMotor.run_forever(speed_sp = -100)
# time.sleep(3)
# armMotor.stop()


# ev3.Sound.speak("Hello, how are you").wait()

# ev3.Sound.speak("Now moving base.").wait()
# baseMotor.run_forever(speed_sp = 100)
# time.sleep(3)
# baseMotor.stop()

# ev3.Sound.speak("Now moving arm.").wait()
# armMotor.run_forever(speed_sp = 100)
# time.sleep(3)
# armMotor.stop()

#
# baseMotor.run_forever(speed_sp = -100)
# time.sleep(3)
# baseMotor.stop()
#
# armMotor.run_forever(speed_sp = -100)
# time.sleep(3)
# armMotor.stop()
#



# armtouch = False
# basetouch = False
# while(basetouch == False):
#     baseMotor.run_forever(speep_sp = 100)
#     time.sleep(5)
#     baseMotor.stop()
#     basetouch = baseTouch.is_pressed
#
# while(armtouch == False):
#     armMotor.run_forever(speed_sp = 100)
#     time.sleep(5)
#     armMotor.stop()
#     armtouch = armTouch.is_pressed
