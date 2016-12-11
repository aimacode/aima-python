# import nxt
#
# # brick = nxt.locator.find_one_brick(name = 'Hooper')
# b = nxt.locator.find_one_brick(
#     name="NXT", strict=True,
#     method=nxt.locator.Method(
#         bluetooth=False, fantomusb=True, fantombt=False, usb=False))

from ev3dev.auto import OUTPUT_D, LargeMotor, OUTPUT_A, MediumMotor, OUTPUT_C
import time

mup = LargeMotor(OUTPUT_D)
mgrab = MediumMotor(OUTPUT_A)
mright = LargeMotor(OUTPUT_C)
#print(m)
# m.run_forever(speed_sp = 360)
# time.sleep(1)
# m.stop()
# print('Hooray')

# mright.run_timed(speed_sp = -360, time_sp = 200)

mup.run_timed(speed_sp = -360, time_sp = 500)
# mright.run_timed(speed_sp = -360, time_sp = 200)
mgrab.run_timed(speed_sp = -360, time_sp = 400)

# #move the arm down
# mup.run_timed(speed_sp = 360, time_sp = 500)
# #grab the object
# time.sleep(1)
# mgrab.run_timed(speed_sp = 360, time_sp = 400)
# # move arm back up
# time.sleep(1)
# mup.run_timed(speed_sp = -360, time_sp = 500)
# #move arm right
# time.sleep(1)
# mright.run_timed(speed_sp = 360, time_sp = 300)
# #move arm down at target location
# time.sleep(1)
# mup.run_timed(speed_sp = 360, time_sp = 705)
# #release the object
# time.sleep(1)
# mgrab.run_timed(speed_sp = -360, time_sp = 400)
# #reset position
# time.sleep(1)
# mup.run_timed(speed_sp = -360, time_sp = 705)
# time.sleep(1)
# mright.run_timed(speed_sp =-360, time_sp = 300)