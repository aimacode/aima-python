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
print(m)
m.run_forever(speed_sp = 360)
time.sleep(1)
m.stop()
