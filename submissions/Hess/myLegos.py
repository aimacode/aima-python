from ev3dev.auto import OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D, LargeMotor, MediumMotor
import time
import ev3dev.ev3 as ev3

clawMotor = MediumMotor(OUTPUT_A)
armMotor = LargeMotor(OUTPUT_B)
baseMotor = LargeMotor(OUTPUT_C)

ev3.Sound.speak("Hello, how are you").wait()

ev3.Sound.speak("Now moving base.").wait()
baseMotor.run_forever(speed_sp = 500)
time.sleep(2)
baseMotor.stop()

ev3.Sound.speak("Now moving arm.").wait()
armMotor.run_forever(speed_sp = 500)
time.sleep(2)
armMotor.stop()

ev3.Sound.speak("Now moving claw.").wait()
clawMotor.run_forever(speed_sp = 500)
time.sleep(2)
clawMotor.stop()
