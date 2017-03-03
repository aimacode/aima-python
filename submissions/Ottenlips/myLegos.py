
# import nxt
#
# brick = nxt.locator.find_one_brick()
# b = nxt.locator.find_one_brick(
#     name="NXT", strict=True,
#     method=nxt.locator.Method(
#         bluetooth=True, fantomusb=True, fantombt=False, usb=False))
#
from ev3dev.auto import INPUT_1, INPUT_AUTO, INPUT_4, INPUT_2, INPUT_3, OUTPUT_A, OUTPUT_D,OUTPUT_C
import ev3dev.ev3 as ev3
from ev3dev.auto import *
from ev3dev.core import LargeMotor,InfraredSensor, MediumMotor, ColorSensor
import time


left = LargeMotor(OUTPUT_D)
right = LargeMotor(OUTPUT_C)

claw = MediumMotor(OUTPUT_A)
# ir = InfraredSensor(address="in4")
# ir = ev3.InfraredSensor()
# ir.mode = 'IR-PROX'

# assert ir.connected

cs = ev3.ColorSensor()



def spin(deg, speed):
    left.run_to_rel_pos(position_sp=deg/2, speed_sp=speed)
    right.run_to_rel_pos(position_sp=-deg/2, speed_sp=speed)

def drive(dist, speed):
    turns = dist * 360 / 10
    left.run_to_rel_pos(position_sp=turns, speed_sp=speed)
    right.run_to_rel_pos(position_sp=turns, speed_sp=speed)

def close_claw():
    claw.run_to_abs_pos(position_sp=900, speed_sp=300)

def open_claw():
    claw.run_to_abs_pos(position_sp=-900, speed_sp=300)

# drive(10,200)



item_found = False
item_reached = False
target_found = False
target_reached = False
open_claw()
c = cs.color()
while not item_found:
    spin(-5,10)
    c = cs.color()
    print(c)
    if(c==6):
        time.sleep(1)
        drive(10,200)
        time.sleep(1)
        close_claw()
        item_found = True

time.sleep(1)
spin(-180, 900)
time.sleep(1)
drive(20,200)
time.sleep(1)

# drive(20,200)
while not target_found:
    drive(5,200)
    c = cs.color()
    print(c)
    if(c==6):
        open_claw()
        drive(-30,200)

        target_found = True


