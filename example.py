import numpy as np
from skyc_maker import Trajectory, write_skyc, XYZYaw, select_file, evaluate_pickle


def fig8(t, x_start, y_start, x_max, y_max, z, yaw_max=0.0):
    period = 5
    rho = np.pi / period * t
    x = list(x_start + np.sin(rho) * x_max)
    y = list(y_start + np.sin(2*rho) * y_max)
    z = [z for _ in x]
    yaw = list(np.sin(1.5*rho) * yaw_max)
    return t, x, y, z, yaw


traj = Trajectory("POLY4D")
traj.set_start(XYZYaw(0, 0, 0, 0))
traj.add_goto(XYZYaw(1, 1, 1, 270), 5)
traj.add_goto(XYZYaw(0, 2, 1, -45), 5)
# t = np.linspace(0, 25, 6000)
# t_x_y_z_yaw = fig8(t, 0, -0.5, 0.5, 0.2, 0.5, np.pi)
pickle_filename = select_file("pickle")
t_x_y_z_yaw, parameters = evaluate_pickle(pickle_filename)
traj.add_goto(XYZYaw(x=t_x_y_z_yaw[1][0],
                     y=t_x_y_z_yaw[2][0],
                     z=t_x_y_z_yaw[3][0],
                     yaw=t_x_y_z_yaw[4][0]), 5)
traj.add_interpolated_traj(t_x_y_z_yaw, 30)
traj.add_goto(XYZYaw(0, 0, 0, 0), 5)
write_skyc([traj])
