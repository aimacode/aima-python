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
#print(m)
# m.run_forever(speed_sp = 360)
# time.sleep(1)
# m.stop()
# print('Hooray')

#move the arm down
# mup.run_timed(speed_sp = -360, time_sp = 705)
mgrab = MediumMotor(OUTPUT_A)
# #grab the object
mgrab.run_timed(speed_sp = 360, time_sp = 450)
#move arm back up
# mup.run_timed(speed_sp = -40, time_sp = 705)
# mright = LargeMotor(OUTPUT_C)
# #move arm right
# mright.run_timed(speed_sp = 40, time_sp = 1000)
# #move arm down at target location
# mup.run_timed(speed_sp = 40, time_sp = 705)
# #release the object
# mgrab.run_timed(speed_sp = -10, time_sp = 300)
# #reset position
# mup.run_timed(speed_sp = -40, time_sp = 705)
# mright.run_timed(speed_sp =-40, time_sp = 1000)