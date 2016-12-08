from ev3dev.auto import OUTPUT_A, OUTPUT_B, OUTPUT_C, LargeMotor, MediumMotor
from ev3dev.auto import INPUT_1, INPUT_2, INPUT_3, TouchSensor, ColorSensor
import time
import ev3dev.ev3 as ev3


    # def __init__(self):
baseTouch = TouchSensor(INPUT_1)
armTouch = TouchSensor(INPUT_2)
colorSensor = ColorSensor(INPUT_3)
clawMotor = MediumMotor(OUTPUT_A)
armMotor = LargeMotor(OUTPUT_B)
baseMotor = LargeMotor(OUTPUT_C)
        # ClawBot.reset(self)


    # def test(self):
    #     ev3.Sound.speak("Hello, how are you").wait()
    #     ev3.Sound.speak("Now moving base.").wait()
    #     ClawBot.baseMotor.run_forever(speed_sp = 500)
    #     time.sleep(2)
    #     ClawBot.baseMotor.stop()
    #     ev3.Sound.speak("Now moving arm.").wait()
    #     ClawBot.armMotor.run_forever(speed_sp = 500)
    #     time.sleep(2)
    #     ClawBot.armMotor.stop()
    #     ev3.Sound.speak("Now moving claw.").wait()
    #     ClawBot.clawMotor.run_forever(speed_sp = 500)
    #     time.sleep(2)
    #     ClawBot.clawMotor.stop()
    #     return 0

    # def reset(self):
armtouch = False
basetouch = False
while(basetouch == False):
    baseMotor.run_forever(speep_sp = 360)
    time.sleep(1)
    basetouch = baseTouch.is_pressed()
while(armtouch == False):
    armMotor.run_forever(speed_sp = -360)
    time.sleep(1)
    armtouch = armTouch.is_pressed()
