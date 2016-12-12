from ev3dev.auto import OUTPUT_D, LargeMotor, OUTPUT_A, MediumMotor, OUTPUT_C
import time

mup = LargeMotor(OUTPUT_D)
mgrab = MediumMotor(OUTPUT_A)
mright = LargeMotor(OUTPUT_C)

#move the arm down
mup.run_timed(speed_sp = 360, time_sp = 600)
#grab the object
time.sleep(1)
mgrab.run_timed(speed_sp = 720, time_sp = 500)
# move arm back up
time.sleep(1)
mup.run_timed(speed_sp = -360, time_sp = 600)
#move arm right
time.sleep(1)
mright.run_timed(speed_sp = 360, time_sp = 600)
#move arm down at target location
time.sleep(1)
mup.run_timed(speed_sp = 360, time_sp = 705)
#release the object
time.sleep(1)
mgrab.run_timed(speed_sp = -360, time_sp = 400)
#reset position
time.sleep(1)
mup.run_timed(speed_sp = -360, time_sp = 600)
time.sleep(1)
mright.run_timed(speed_sp =-360, time_sp = 400)