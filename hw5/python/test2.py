# collect
trajectory_trap_vel(waypoints, times, frequency, duty_cycle)

collect wps

1) init->pick1, pick1
2) pick1->put1, put
3) ut1->pick2, pick2
4) pick2->put2, put2
5) put2->pick3, pick3
6) pick3->put3, put3

def traj(wp1, wp2, end_action="pick"):
    
    ee_loc_1 = fk(wp1)
    ee_loc_2 = fk(wp2)

    interpolate in xyz space timesteps 
    

    solve ik at each timestep to form joint angle matrix 

    command_traj(robot, trajectory, frequency=10Hz)
    
    if end_action == "pick":
        pick()
    else:
        put()
    pause(delay);

    




