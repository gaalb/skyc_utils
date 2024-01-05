import json
from scipy import interpolate as interpolate
import matplotlib.pyplot as plt
import sys
import numpy as np
import pickle
import os
import shutil
import zipfile
import tkinter as tk
from tkinter import filedialog
from typing import Union, List, Optional, Tuple, Any
from functools import partial as partial
from dataclasses import dataclass


@dataclass
class XYZYaw:
    """
    A way to encapsulate data that is 4-dimensional. For example, we may store a spline each for x, y, z and yaw.
    Using this class, we can reach these both by indexing, iterating and directly addressing like .x, .y, etc.
    """
    x: Any
    y: Any
    z: Any
    yaw: Any

    def __getitem__(self, idx):
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.y
        elif idx == 2:
            return self.z
        elif idx == 3:
            return self.yaw
        else:
            return None

    def __len__(self):
        return 4

    def __iter__(self):
        for attr in (self.x, self.y, self.z, self.yaw):
            yield attr


def cleanup(files: List[str], folders: List[str]):
    # function meant for deleting unnecessary files
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted {file}")
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted {folder} folder")


def write_trajectory(Data):
    traj_type = SETTINGS.get("type")
    # this is the format that a TrajectorySpecification requires:
    assert traj_type == "POLY4D" or traj_type == "COMPRESSED"
    json_dict = {
        "version": 1,
        "points": Data,
        "takeoffTime": Data[0][0],
        "landingTime": Data[-1][0],
        "type": traj_type
    }
    json_object = json.dumps(json_dict, indent=2)
    with open("trajectory.json", "w") as f:
        f.write(json_object)


def write_to_skyc(Skyc_Data):
    # delete every file that we can generate that might have been left over from previous sessions
    name = sys.argv[0][:-3]
    cleanup(files=["show.json",
                   "cues.json",
                   f"{name}.zip",
                   f"{name}.skyc",
                   "trajectory.json"],
            folders=["drones"])
    # Create the 'drones' folder if it doesn't already exist
    os.makedirs('drones', exist_ok=True)
    drones = []
    for index, Data_Params in enumerate(Skyc_Data):
        Data, parameters = Data_Params
        # The trajectory is saved to a json file with the data below
        write_trajectory(Data)
        drone_settings = {
            "trajectory": {"$ref": f"./drones/drone_{index}/trajectory.json#"},
            "home": Data[0][1][0:3],
            "startYaw": Data[0][1][-1],
            "landAt": Data[-1][1][0:3],
            "name": f"drone_{index}",
        }
        if parameters is not None:
            drone_settings["parameters"] = parameters
        drones.append({
            "type": "generic",
            "settings": drone_settings
        })

        # Create the 'drone_1' folder if it doesn't already exist
        drone_folder = os.path.join('drones', f'drone_{index}')
        os.makedirs(drone_folder, exist_ok=True)
        shutil.move('trajectory.json', drone_folder)
    # This wall of text below is just overhead that is required to make a skyc file.
    ########################################CUES.JSON########################################
    items = [{"time": Skyc_Data[0][0][0],
              "name": "start"}]
    cues = {
        "version": 1,
        "items": items
    }
    json_object = json.dumps(cues, indent=2)
    with open("cues.json", "w") as f:
        f.write(json_object)
    #######################################SHOW.JSON###########################################
    validation = {
        "maxAltitude": 2.0,
        "maxVelocityXY": 2.0,
        "maxVelocityZ": 1.5,
        "minDistance": 0.8
    }
    cues = {
        "$ref": "./cues.json"
    }
    settings = {
        "cues": cues,
        "validation": validation
    }
    meta = {
        "id": f"{name}.py",
        "inputs": [f"{name}.py"]
    }
    show = {
        "version": 1,
        "settings": settings,
        "swarm": {"drones": drones},
        "environment": {"type": "indoor"},
        "meta": meta,
        "media": {}
    }
    json_object = json.dumps(show, indent=2)
    with open("show.json", "w") as f:
        f.write(json_object)

    # Create a new zip file
    with zipfile.ZipFile(f"{name}.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the first file to the zip
        zipf.write("show.json")

        # Add the second file to the zip
        zipf.write("cues.json")

        # Recursively add files from the specified folder and its sub-folders
        for root, _, files in os.walk("drones"):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path)

    print('Compression complete. The files and folder have been zipped as demo.zip.')

    os.rename(f'{name}.zip', f'{name}.skyc')
    # Delete everything that's not 'trajectory.skyc'
    cleanup(files=["show.json",
                   "cues.json",
                   f"{name}.zip",
                   "trajectory.json"],
            folders=["drones"])
    print("Skyc file ready!")


def determine_knots(time_vector, N):
    '''returns knot vector for the BSplines according to the incoming timestamps and the desired number of knots'''
    # # Problems start to arise when part_length becomes way smaller than N, so generally keep them longer :)
    # part_length = len(time_vector) // N
    # result = time_vector[::part_length][:N]
    # result.append(time_vector[-1])
    start = min(time_vector)
    end = max(time_vector)
    result = list(np.linspace(start, end, N))
    return result


def get_bezier_segments(t_x_y_z_yaw: Tuple[List[float], List[float], List[float], List[float], List[float]],
                        degree, number_of_bezier_segments, start_pose):
    t, x, y, z, yaw = t_x_y_z_yaw
    knots = determine_knots(t, number_of_bezier_segments)[1:-1]
    # We need to give the splrep inside knots. I think [0] and [-1] should also technically be inside knots, but apparently
    # not. I seem to remember that the first k-1 and last k-1 knots are the outside knots. Anyway, slprep seems to add k
    # knots both at the end and at the beginning, instead of k-1 knots which is what would make sense to me. How it decides
    # what those knots should be is a mystery to me, but upon checking them, they are the exact first and last knots that I
    # would've added, so it works out kind of.
    w = [1] * len(t)  # if the fit is particularly bad a certain point in the path, we can adjust it here
    if SETTINGS.get("method", "scipy") != "scipy":
        raise NotImplementedError
    splines = XYZYaw(x=interpolate.splrep(t, x, w, k=degree, task=-1, t=knots),
                     y=interpolate.splrep(t, y, w, k=degree, task=-1, t=knots),
                     z=interpolate.splrep(t, z, w, k=degree, task=-1, t=knots),
                     yaw=interpolate.splrep(t, yaw, w, k=degree, task=-1, t=knots))
    c = SETTINGS.get("continuity", 2)
    assert degree >= 2*c-1
    start_conditions = XYZYaw(x=[0.0]*c, y=[0.0]*c, z=[0.0]*c, yaw=[0.0]*c)  # contains pose and its derivatives at t=0
    end_conditions = XYZYaw(x=[0.0]*c, y=[0.0]*c, z=[0.0]*c, yaw=[0.0]*c)  # contains pose and its derivatives at end
    for i in range(len(start_conditions)):  # for each dimension (x, y, z, yaw)
        for j in range(c):  # for each derivative number
            start_conditions[i][j] = float(interpolate.splev(t[0], splines[i], der=j))
            end_conditions[i][j] = float(interpolate.splev(t[-1], splines[i], der=j))
    if start_pose is None:  # meaning we just take off in place
        start_pose = XYZYaw(x=start_conditions.x[0], y=start_conditions.y[0], z=0.0, yaw=start_conditions.yaw[0])
    else:  # meaning we take off at a different position from the start
        start_pose = XYZYaw(x=start_pose[0], y=start_pose[1], z=0.0, yaw=start_pose[2])
    Bezier_Data = [[t[0],
                    [start_pose.x, start_pose.y, start_pose.z, start_pose.yaw],
                    []]]
    takeoff = 5
    takeoff_seg = goto_segment(XYZYaw(x=[0.0]*c, y=[0.0]*c, z=[0.0]*c, yaw=[0.0]*c), start_conditions, takeoff)
    Bezier_Data.append([takeoff, takeoff_seg[-1], takeoff_seg[1:-1]])
    # BPoly can be constructed from PPoly but not from BSpline. PPoly can be constructed from BSPline. BSpline can
    # be fitted to points. So Points->PPoly->BPoly. The coeffs of the BPoly representation are the control points.
    ppolys = XYZYaw(*[interpolate.PPoly.from_spline(spline) for spline in splines])
    # shift the time because of the goto segment
    for ppoly in ppolys:
        ppoly.x = ppoly.x + takeoff
    bpolys = XYZYaw(*[interpolate.BPoly.from_power_basis(ppoly) for ppoly in ppolys])
    # These two lines below seem complicated but all they do is pack the data above into a convenient form: a list
    # of lists where each element looks like this: [t, (x,y,z), (x,y,z), (x,y,z)].
    bpoly_pts = list(zip(list(bpolys.x.x)[degree+1:-degree],
                     *[list(bpoly.c.transpose())[degree:-degree] for bpoly in bpolys]))
    # at this point bpoly_pts contains the control points for the segments, but that's not exactly what we need in
    # the skyc file: we need the last point, and the inside points
    BPoly = [[element[0]] + list(zip(*list(element[1:]))) for element in bpoly_pts]
    for Bezier_Curve in BPoly:
        curve_to_append = [Bezier_Curve[0],
                           Bezier_Curve[-1],
                           Bezier_Curve[2:-1]]
        Bezier_Data.append(curve_to_append)
    return Bezier_Data, end_conditions


def determine_shorter_deg(start: float, end: float):
    start_norm = start % 360
    end_norm = end % 360
    delta = end_norm - start_norm
    if delta > 180:
        delta = delta - 360
    elif delta < -180:
        delta = delta + 360
    return start + delta


def goto_segment(goto_start, goto_end, dt):
    t0 = 0
    T = t0 + dt
    #    c0, c1, c2,    c3,    c4,    c5
    A = [[1, t0, t0**2, t0**3, t0**4, t0**5],
         [1, T, T**2, T**3, T**4, T**5],
         [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
         [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
         [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
         [0, 0, 2, 6*T, 12*T**2, 20*T**3]]
    A = np.array([lst[:len(goto_start[0])*2] for lst in A[:len(goto_start[0])*2]])
    # these coefficients solve the equation above, thereby ensuring a takeoff that's c2 smooth
    b = np.row_stack((np.column_stack(goto_start), np.column_stack(goto_end)))
    c = np.linalg.solve(A, b)
    coeffs = XYZYaw(*[col.reshape(len(col), 1) for col in np.transpose(np.flip(c, axis=0))])
    ppolys = XYZYaw(*[interpolate.PPoly(cs, np.array([t0, T])) for cs in coeffs])
    bpolys = XYZYaw(*[interpolate.BPoly.from_power_basis(ppoly) for ppoly in ppolys])
    bezier_curve = np.concatenate([poly.c for poly in bpolys], axis=1).tolist()
    return bezier_curve


def get_skyc_data(TXYZHeading: List[Tuple[List[float], List[float], List[float], List[float], List[float]]],
                  degree, number_of_bezier_segments=200,
                  parameters=None):
    c = SETTINGS.get("continuity", 2)
    gotos_after = SETTINGS.get("gotos_after")
    start_poses = SETTINGS.get("start_poses", None)
    Skyc_Data = []
    for i, t_x_y_z_yaw in enumerate(TXYZHeading):
        params = parameters[i] if parameters is not None else None
        takeoff_pose = None if start_poses is None else start_poses[i]
        Bezier_Data, end_conditions = get_bezier_segments(t_x_y_z_yaw, degree, number_of_bezier_segments, takeoff_pose)
        for goto in gotos_after[i]:
            goto[1].yaw = determine_shorter_deg(end_conditions.yaw[0], goto[1].yaw)
            new_end_conditions = XYZYaw(*[[float(pos)] + (c-1)*[0.0] for pos in goto[1]])
            segment = goto_segment(end_conditions, new_end_conditions, goto[0])
            Bezier_Data.append([Bezier_Data[-1][0],
                                segment[-1],
                                segment[1:-1]])
            end_conditions = new_end_conditions
        Skyc_Data.append([Bezier_Data, params])
    return Skyc_Data


def open_file_dialog(file_path_var: List[Optional[str]], root: tk.Tk, filetype: str) -> None:
    file_path = filedialog.askopenfilename(initialdir=os.path.dirname(__file__),
                                           title=f"Select a {filetype} file!",
                                           filetypes=[(f"{filetype} files", f"*.{filetype}"), ("all files", "*.*")])
    if file_path:
        print(f"Selected file: {file_path}")
        file_path_var[0] = file_path
        root.destroy()


def select_file(filetype: str) -> Union[None, str]:
    """ Function that prompts the user to select a skyc file, and returns the file's name if successful.
    Else returns None"""
    selecter = tk.Tk()
    selecter.title(f"Select {filetype} file!")
    selecter.geometry("300x100")
    # using a list to mimic a mutable object (I hate python so much why are you forcing me to
    # do this, just let me pass this by reference, horrible toy language...)
    selected_file = [None]
    button_func = partial(open_file_dialog, selected_file, selecter, filetype)
    button = tk.Button(selecter, text=f"Select {filetype} file!", command=button_func, width=20, height=4)
    button.pack(pady=20)
    selecter.mainloop()
    return selected_file[0]


def evaluate_pickle(pickle_name: str, *, x_offset=0.0, y_offset=0.0, z_offset=0.0, yaw_offset=0.0):
    with open(pickle_name, "rb") as file:
        data = pickle.load(file)
        parameters = data.get("parameters", None)
        segments = data["traj"]
        contains_yaw = len(segments) == 4
        # unpack the data so that the outer index is the slice index, and the inner index switches between x-y-z-yaw
        segments = list(zip(*segments))
        x, y, z, t, yaw = [], [], [], [], []
        for segment in segments:
            # for each segment, segment[0] is the spline for x, [1] is for y, [2] is for z, [3] is for yaw. Within
            # those, [0] is the knot vector, [1] is the coeffs, [2] is the degree of the spline
            granularity = 1000
            # technically, in the pickle, x, y, z and yaw could have different knots, but for practical reasons, they
            # obviously don't. We make use of this fact here to simplify the code, however, if the pickle's composition
            # was changed, then this would have to change as well. TODO: discuss this with Peti
            # also, exclude [0] to avoid duplicate timestamps
            eval_time = np.linspace(segment[0][0][0], segment[0][0][-1], granularity+1)[1:]
            x = x + list(interpolate.splev(eval_time, segment[0]))
            y = y + list(interpolate.splev(eval_time, segment[1]))
            z = z + list(interpolate.splev(eval_time, segment[2]))
            yaw = yaw + list(interpolate.splev(eval_time, segment[3])) if contains_yaw else None
            eval_time = eval_time + t[-1] if len(t) > 1 else eval_time  # shift relative time to absolute
            t = t + list(eval_time)
        x = [element + x_offset for element in x]
        y = [element + y_offset for element in y]
        z = [element + z_offset for element in z]
        yaw = [np.rad2deg(element) + yaw_offset for element in yaw] if contains_yaw else None
    return ((t, x, y, z, yaw), parameters) if contains_yaw else ((t, x, y, z), parameters)


def fig8(t, x_start, y_start, x_max, y_max, z, yaw_max=0.0):
    period = 5
    rho = np.pi / period * t
    x = list(x_start + np.sin(rho) * x_max)
    y = list(y_start + np.sin(2*rho) * y_max)
    z = [z for e in x]
    yaw = list(np.sin(1.5*rho) * yaw_max)
    return t, x, y, z, yaw


SETTINGS = {
    "input": "pickle",  # TXYZ: input is t-x-y-z-yaw samples, pickle: input is pickle of splines
    "method": "scipy",  # spline->bezier conversion method, either scipy or recursive
    "type": "POLY4D",  # trajectory representation in crazyflie memory: POLY4D or COMPRESSED
    "start_poses": [[0.0, 0.0, 0.0]],  # if not None, then this means an extra goto
    "gotos_after": [[[5, XYZYaw(0, 0, 1, 0)], [3, XYZYaw(0, 0, 0, 0)]]],  # goto segments after the end of the trajectory
    "continuity": 3  # the degree of continuity between gotos
}

degree = -1
number_of_segments = 0
if SETTINGS.get("type", "COMPRESSED") == "COMPRESSED":
    degree = 3
    number_of_segments = 200
if SETTINGS.get("type", "COMPRESSED") == "POLY4D":
    degree = 5
    number_of_segments = 55

# TODO: figure out how many segments we can have and determine allowed number of segments based on gotos (each adds 1)
assert (7 >= degree >= 0)  # this also makes sure that the trajectory type was valid
TXYZ = []
if SETTINGS.get("input", "TXYZ") == "pickle":
    # pickle_filename = select_file("pickle")
    pickle_filename = "traj_new.pickle"
    t_x_y_z_yaw, parameters = evaluate_pickle(pickle_filename)
    TXYZ.append(t_x_y_z_yaw)
else:
    t = np.linspace(0, 25, 6000)
    TXYZ.append(fig8(t, 0, -0.5, 0.5, 0.2, 0.5, np.pi))
    TXYZ.append(fig8(t, 0, 0.5, 0.4, 0.4, 1, ))
Skyc_Data = get_skyc_data(TXYZ, degree=degree, number_of_bezier_segments=number_of_segments)
write_to_skyc(Skyc_Data)



