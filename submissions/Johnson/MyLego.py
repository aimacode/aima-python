import ev3dev.ev3 as ev3

claw = ev3.MediumMotor('outA')
arm = ev3.LargeMotor('outB')
swing = ev3.LargeMotor('outC')

claw.run_timed(time_sp=3000, speed_sp=500)