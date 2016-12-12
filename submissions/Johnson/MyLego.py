#!/usr/bin/env python3
from ev3dev.ev3 import *
import ev3dev.ev3 as ev3
import time

def open(claw):
    claw.run_timed(time_sp=500, speed_sp=-350)

def close(claw):
    claw.run_timed(time_sp=500, speed_sp=1000)

def main():
    claw = ev3.MediumMotor('outA')
    arm = ev3.LargeMotor('outB')
    swing = ev3.LargeMotor('outC')

    open(claw)
    time.sleep(1)
    arm.run_timed(time_sp=500, speed_sp=300)
    time.sleep(1)
    close(claw)
    time.sleep(1)
    arm.run_timed(time_sp=500, speed_sp=-400)
    time.sleep(1)
    swing.run_timed(time_sp=500, speed_sp=1000)
    time.sleep(1)
    arm.run_timed(time_sp=500, speed_sp=300)
    time.sleep(1)
    claw.run_timed(time_sp=500, speed_sp=-250)
    time.sleep(1)
    arm.run_timed(time_sp=500, speed_sp=-400)

if __name__ == '__main__':
    main()