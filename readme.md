# Skyc utilities package 
# Installation
Clone this repository, then change into the cloned directory.
## On Linux
1. python3 -m venv venv
2. source venv/bin/activate
3. pip install -e .
## On Windows
1. python -m venv venv
2. cd into the folder venv/Scripts
3. activate
4. navitage back to the root directory of the project
5. pip install -e .
# Skyc inspector
This script will prompt you to select a skyc file, and will evaluate it, providing animation
and diagrams.
# Skyc maker
With the Trajectory, XYZYaw classes and the write_skyc function, you can craft a trajectory
and then write it to a skyc file. Take a look at the provided example.py for an example. 
To make a trajectory, you must initialize it by providing its type ("POLY4D" or "COMPRESSED").
By default POLY4D trajectories are of degree 5, and COMPRESSED ones are of degree 3.
Then set its starting point, and add whatever gotos you would like using `add_goto`
A goto consists of an XYZYaw
object, each dimension of which may be an iterable. In this iterable, you can specify the desired
position, velocity and acceleration at the end of the goto. Take care to provide constraints that
make sense mathematically. For example:
- if your Trajectory is degree 3, you can't give acceleration constraints, since to match the 
position, velocity, and acceleration both at the start and end of the goto segment, you would
need 6 variables; a degree 3 Trajectory can only have 4.
- if your goto has velocity a constraint, give continuity of at least 2 (default is 3). Obviously,
with a continuity of 1, the resulting goto segment will only be continuous in position, and will
therefore violate your constraints. If you're unsure what continuity to use, leave it unfilled, or 
set to 2.

In summary, make sure that the continuity of your goto segments (1 to 3) is less than half of the
degree of your Trajectory (POLY4D->5, COMPRESSED->3), and set your goto constraints accordingly.

If you want a more complex Trajectory, provide a list of t-x-y-z-heading data, where each coordinate
is a list of equal length. The function `add_interpolated_traj` will fit bezier segments to this
data. If you want your trajectory to transition smoothly into this part of the trajectory, then add
a goto with the trajectory's starting conditions before adding the interpolated trajectory. 

Once you're satisfied with your Trajectory, generate it with `write_skyc`. This function expects a 
list of Trajectory objects, since you may want to include several drones in your skyc file. If you
only have one, provide a list with only one element.