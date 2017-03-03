from ev3dev.auto import OUTPUT_D, LargeMotor, OUTPUT_A, MediumMotor, OUTPUT_C
import time

right = LargeMotor(OUTPUT_D) and LargeMotor(OUTPUT_A)

right.run_timed(speed_sp = 360, time_sp = 600)
time.sleep(1)


