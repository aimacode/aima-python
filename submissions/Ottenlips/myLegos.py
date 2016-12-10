
import nxt

# brick = nxt.locator.find_one_brick(name = 'Hooper')
b = nxt.locator.find_one_brick(
    name="NXT", strict=True,
    method=nxt.locator.Method(
        bluetooth=False, fantomusb=True, fantombt=False, usb=False))

from ev3dev.auto import OUTPUT_A, OUTPUT_C, OUTPUT_D, LargeMotor, INPUT_1, INPUT_4, InfraredSensor, MediumMotor, ColorSensor
import time

left = LargeMotor(OUTPUT_D)
right = LargeMotor(OUTPUT_C)
claw = MediumMotor(OUTPUT_A)
ir = InfraredSensor(INPUT_4)
cs = ColorSensor(INPUT_1)

def spin(deg, speed):
    left.run_to_rel_pos(position_sp=deg/2, speed_sp=speed)
    right.run_to_rel_pos(position_sp=-deg/2, speed_sp=speed)

def drive(dist, speed):
    turns = dist * 360 / 10
    left.run_to_rel_pos(position_sp=turns, speed_sp=speed)
    right.run_to_rel_pos(position_sp=turns, speed_sp=speed)

def close_claw():
    claw.run_to_abs_pos(position_sp=0, speed_sp=300)

def open_claw():
    claw.run_to_abs_pos(position_sp=90, speed_sp=300)


item_found = False
item_reached = False
target_found = False
target_reached = False

while not item_found:
    spin(10, 400)
    if ir.proximity < 100:
        item_found = True

while not item_reached:
    drive(5, 400)
    if ir.proximity < 10:
        item_reached = True

while not target_found:
    spin(10, 400)
    if cs.color == 6: # white
        target_found = True

while not target_reached:
    drive(5, 400)
    if cs.reflected_light_intensity < 10:
        target_reached = True

open_claw()
spin(90, 400)
drive(40, 800)

close_claw()

