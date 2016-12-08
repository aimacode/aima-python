# import nxt
#
# # brick = nxt.locator.find_one_brick(name = 'Hooper')
# b = nxt.locator.find_one_brick(
#     name="NXT", strict=True,
#     method=nxt.locator.Method(
#         bluetooth=False, fantomusb=True, fantombt=False, usb=False))

from ev3dev.auto import OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D, LargeMotor
import time
import ev3dev.ev3 as ev3

ev3.Sound.speak("Hello, how are you").wait()
mA = LargeMotor(OUTPUT_A)
mB = LargeMotor(OUTPUT_B)
mC = LargeMotor(OUTPUT_C)
mD = LargeMotor(OUTPUT_D)
m.run_forever(speed_sp = 360)
time.sleep(1)
m.stop()
