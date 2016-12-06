# import nxt
#
# # brick = nxt.locator.find_one_brick(name = 'Hooper')
# b = nxt.locator.find_one_brick(
#     name="NXT", strict=True,
#     method=nxt.locator.Method(
#         bluetooth=False, fantomusb=True, fantombt=False, usb=False))

from ev3dev.auto import OUTPUT_D, LargeMotor
import time

m = LargeMotor(OUTPUT_D)
m.run_forever(duty_cycle_sp = 100)
time.sleep(1)
m.stop()