
# fk
'''  ******* 2. COLLECT WPs *******'''

teach_waypoints
%   Returns a matrix of joint angle waypoints, where each column represents a
%   single waypoint (set of joint angles)

WPs =  (num_joints x num_waypoints)

''' ******* 3. FIGURE OUT WORKSPACE TRAJ (X,Y) *******'''
num_joints = size(collected_waypoints, 1);
num_wps = size(collected_waypoints, 2);
# forward kinematics from joint angles 
ee = zeros(3, num_wps)
for i= 1:num_wps:
    wp = collected_waypoints(:,i)
    ee_loc(:;i) = robot.fk(wp) 
end


'''  ******* 3.5 DO LINEAR INTERPOLATION IN XYZ SPACE  *******'''
interpolated = 


'''  ******* 4. DO IK OUTSIDE LOOP  *******'''
# solve for joint configs needed to achieve EE pos 
# goal po

num_interp_wps = size(interpolated,2)
thetas = zeros(joint_angles, num_interp_wps)
for i=1:num_interp_wps:
    goal_ee_pos = interpolated(:,i)
    th = inverse_kinematics_analytical(robot, goal_ee_pos)
    thetas(:, num_interp_wps) = th
end 

'''  ******* 5. GET JOINTS IN TRAJECTORY   *******'''

# get all joints in trajectory



# command
'''  ******* 6. SEND COMMANDS  *******'''

command_trajectory(robot, trajectory, frequency);
pick(gripper);
pause(0.75);

trajectory = trajectory_spline([pick_position put_position], [0, 1], frequency);
command_trajectory(robot, trajectory, frequency);
put(gripper);
pause(0.75);

fbk = robot.getNextFeedback()
initial_thetas = fbk.position'; # transpose turns feedback into column vector

pick(gripper); # pick object at position 1, do not pause more than length of command lifetime above
pause(0.75);

# move 
trajectory = trajectory_spline([position_1 position_1_approach], [0, 1], frequency);
# given matrix traj: (joint_angle x timesteps ), cmd set pos/vel, wait 1/freq
command_trajectory(robot, trajectory, frequency);


# Place the object
place(gripper);
pause(0.75);






''' REFERENCE

trajectory_spline(waypoints, times, frequency)
---------------------------------------------
    Returns a matrix of joint angles, where each column represents a single
    %   timestamp. These joint angles form constant velocity trajectory segments,
    %   hitting waypoints(:,i) at times(i).

    'waypoints' is a matrix of waypoints; each column represents a single
    %   waypoint in joint space, and each row represents a particular joint.
    %
    %   'times' is a row vector that indicates the time each of the waypoints should
    %   be reached.  The number of columns should equal the number of waypoints, and
    %   should be monotonically increasing.  The first number is technically
    %   arbitrary, as it serves only as a reference for the remaining timestamps,
    %   but '0' is a good default.
    %
    %   'frequency' is the control frequency which this trajectory should be played
    %   at, and therefore the number of columns per second of playback.
'''