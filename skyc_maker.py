import json
from scipy import interpolate as interpolate  # !
import sys
import numpy as np  # !
import os
import shutil
import zipfile
from typing import Union, List, Optional, Any
from dataclasses import dataclass
from functools import partial as partial
import tkinter as tk
from tkinter import filedialog
import pickle


def determine_shorter_deg(start: float, end: float):
    """If we did a turn from 350 to 10 degrees, the goto segment planner would turn the long way around, going
    from 350->340->330...20->10, instead of going the short way, which is 350->360->370=10. This function gives
    a target angle that yields turning the shortest way."""
    start_norm = start % 360
    end_norm = end % 360
    delta = end_norm - start_norm
    if delta > 180:
        delta = delta - 360
    elif delta < -180:
        delta = delta + 360
    return start + delta


@dataclass
class XYZYaw:
    """
    A way to encapsulate data that is 4-dimensional. For example, we may store a spline each for x, y, z and yaw.
    Using this class, we can reach these both by indexing, iterating and directly addressing like .x, .y, etc.
    A simple way of making a XYZYaw object from an iterable of length 4, is like so: obj = XYZYaw(*iter_obj)
    An XYZYaw is also iterable, therefore *obj is fine, and also, obj[0]=obj.x, obj[3]=obj.yaw, etc.
    Note that there is currently no protection in place to ensure that x, y, z and yaw are the same type. TODO
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


def determine_knots(time_vector, N):
    """
    returns knot vector for the BSplines according to the incoming timestamps and the desired number of knots
    Problems start to arise when part_length becomes way smaller than N, so generally keep them longer :)
    Knots may either be spaced equally in time, or by index, currently we do it by index
    """
    # part_length = len(time_vector) // N
    # result = time_vector[::part_length][:N]
    # result.append(time_vector[-1])
    start = min(time_vector)
    end = max(time_vector)
    result = list(np.linspace(start, end, N))
    return result


class Trajectory:
    """
    A class representing all the information regarding a trajectory.
    """
    def __init__(self, traj_type: str):
        self.parameters = None  # parameters that we set during flight
        # www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/trajectory_formats/
        assert traj_type == "POLY4D" or traj_type == "COMPRESSED"
        self.type = traj_type
        self.degree = 5 if traj_type == "POLY4D" else 3
        # regardless of whether the trajectory gets interpreted as poly4d or compressed, it will be packaged as bezier
        # segments in the skyc file for reasons discussed in the wiki of the skybrush server:
        # github.com/AIMotionLab-SZTAKI/skybrush-server/wiki/Changes-from-stock-Skybrush#adding-poly4d-trajectory-representation
        self.bezier_repr: Optional[List] = None

    def set_start(self, start: XYZYaw):
        """
        Sets the starting point of the bezier representation, which is different from the rest, as it may not
        include any control points.
        """
        assert self.bezier_repr is None
        assert isinstance(start.x, (int, float)) and isinstance(start.y, (int, float)) and \
               isinstance(start.z, (int, float)) and isinstance(start.yaw, (int, float))
        self.bezier_repr = [[0.0, [start.x, start.y, start.z, start.yaw], []]]

    @property
    def end_condition(self) -> XYZYaw:
        """
        Looks at the last two segments of the bezier representation and calculates the ending derivatives.
        The end condition is an XYZYaw object where each dimension is a numpy array, in which the nth element (up to 2)
        is the nth derivative in that dimension.
        """
        if self.bezier_repr is None:
            raise NotImplementedError  # if the starting point isn't initialized, don't look at this property!
        if len(self.bezier_repr) == 1:  # if we have a starting point, but nothing else, then the derivatives are 0
            return XYZYaw(x=np.array([self.bezier_repr[0][1][0], 0.0, 0.0]),
                          y=np.array([self.bezier_repr[0][1][1], 0.0, 0.0]),
                          z=np.array([self.bezier_repr[0][1][2], 0.0, 0.0]),
                          yaw=np.array([self.bezier_repr[0][1][3], 0.0, 0.0]))
        else:  # we have at least one bezier curve
            # in the skyc file, the points of a bezier curve are the previous end, the control points, and the end point
            lst_coeffs = [self.bezier_repr[-2][1]] + self.bezier_repr[-1][2] + [self.bezier_repr[-1][1]]
            lst_coeffs = XYZYaw(*list(zip(*lst_coeffs)))  # organize them by dimension
            t = self.bezier_repr[-1][0] - self.bezier_repr[-2][0]
            # make them into a bernstein polynomial, which can be evaluated at the end
            bpolys = XYZYaw(*[interpolate.BPoly([[coeff] for coeff in coeffs], [0, t]) for coeffs in lst_coeffs])
            # return the XYZYaw object which consists of the evaluated derivatives for each dimension, 0 through 2
            return XYZYaw(*[np.array([bpoly(t, nu=nu) for nu in range(3)]) for bpoly in bpolys])

    def add_goto(self, goto: XYZYaw, dt: float, continuity: int = 3):
        """Modifies the bezier representation to add a goto segment, with a given continuity (1: continuous in
        position, 2: continuous in accelaration). Accordingly, in the goto you may give velocity and acceleration
        constraints.
        """
        assert 1 <= continuity <= 3
        goto_start = np.array(self.end_condition)[:, :continuity]
        goto_end = np.array([np.append(e, np.zeros(continuity-len(e))) for e in [np.atleast_1d(x) for x in goto]])
        goto_end[3, 0] = determine_shorter_deg(self.end_condition.yaw[0], goto_end[3][0])
        t0 = 0
        T = t0 + dt
        #              c0, c1, c2,      c3,      c4,      c5
        A1 = np.array([[1, t0, t0 ** 2, t0 ** 3, t0 ** 4, t0 ** 5],
                       [0, 1, 2 * t0, 3 * t0 ** 2, 4 * t0 ** 3, 5 * t0 ** 4],
                       [0, 0, 2, 6 * t0, 12 * t0 ** 2, 20 * t0 ** 3]])
        A2 = np.array([[1, T, T ** 2, T ** 3, T ** 4, T ** 5],
                       [0, 1, 2 * T, 3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                       [0, 0, 2, 6 * T, 12 * T ** 2, 20 * T ** 3]])
        A = np.vstack((A1[:continuity, :2*continuity], A2[:continuity, :2*continuity]))
        # the derivatives we want to end with:
        b = np.row_stack((np.column_stack(goto_start), np.column_stack(goto_end)))
        # these coefficients solve the equations above:
        c = np.linalg.solve(A, b)
        # reshape them to fit the ordering needed by a PPoly object
        coeffs = XYZYaw(*[col.reshape(len(col), 1) for col in np.transpose(np.flip(c, axis=0))])
        ppolys = XYZYaw(*[interpolate.PPoly(cs, np.array([t0, T])) for cs in coeffs])
        # BPoly objects can be made from PPoly objects. We could also calculate them by hand TODO
        bpolys = XYZYaw(*[interpolate.BPoly.from_power_basis(ppoly) for ppoly in ppolys])
        # the points of the resulting bezier curve, TODO: understand the connection between bpolys.c and the points
        bezier_curve = np.concatenate([poly.c for poly in bpolys], axis=1).tolist()
        # the resulting bezier representation looks like this: [prev_seg, [t_end, endpoint, [ctrl_points], next_seg]
        goto_segment = [self.bezier_repr[-1][0]+dt, bezier_curve[-1], bezier_curve[1:-1]]
        self.bezier_repr.append(goto_segment)

    def add_interpolated_traj(self, t_x_y_z_yaw, number_of_segments, method="scipy"):
        """
        t_x_y_z_yaw is the raw interpolated data that we need to organize into number_of_segments bezier segments,
        we then add these segments to the bezier representation. This function doesn't care about continuity in the
        trajectory, if the user wishes c2 continuity, it is up to them to provide that with the correct goto segment.
        """
        if self.type == "POLY4D":
            assert number_of_segments < 60
        t, x, y, z, yaw = t_x_y_z_yaw
        t = [x+self.bezier_repr[-1][0] for x in t]  # timeshift
        # make sure that we don't make an unsafe trajectory, there can't be a break in the positions
        assert abs(x[0]-self.end_condition.x[0]) < 0.01
        assert abs(y[0] - self.end_condition.y[0]) < 0.01
        assert abs(z[0] - self.end_condition.z[0]) < 0.01
        assert abs(yaw[0] - self.end_condition.yaw[0]) < 10
        # We need to give the splrep inside knots. I think [0] and [-1] should also technically be inside knots, but
        # apparently not. I seem to remember that the first k-1 and last k-1 knots are the outside knots. Anyway, slprep
        # seems to add k knots both at the end and at the beginning, instead of k-1 knots which is what would make sense
        # to me. How it decides what those knots should be is a mystery to me, but upon checking them, they are the
        # exact first and last knots that I would've added, so it works out kind of.
        knots = determine_knots(t, number_of_segments)[1:-1]
        w = [1] * len(t)  # if the fit is particularly bad a certain point in the path, we can adjust weights here
        if method != "scipy":  # TODO
            raise NotImplementedError
        splines = XYZYaw(x=interpolate.splrep(t, x, w, k=self.degree, task=-1, t=knots),
                         y=interpolate.splrep(t, y, w, k=self.degree, task=-1, t=knots),
                         z=interpolate.splrep(t, z, w, k=self.degree, task=-1, t=knots),
                         yaw=interpolate.splrep(t, yaw, w, k=self.degree, task=-1, t=knots))
        # BPoly can be constructed from PPoly but not from BSpline. PPoly can be constructed from BSPline. BSpline can
        # be fitted to points. So Points->PPoly->BPoly. The coeffs of the BPoly representation are the control points.
        ppolys = XYZYaw(*[interpolate.PPoly.from_spline(spline) for spline in splines])
        bpolys = XYZYaw(*[interpolate.BPoly.from_power_basis(ppoly) for ppoly in ppolys])
        # These two lines below seem complicated but all they do is pack the data above into a convenient form: a list
        # of lists where each element looks like this: [t, (x,y,z), (x,y,z), (x,y,z)].
        bpoly_pts = list(zip(list(bpolys.x.x)[self.degree + 1:-self.degree],
                             *[list(bpoly.c.transpose())[self.degree:-self.degree] for bpoly in bpolys]))
        # at this point bpoly_pts contains the control points for the segments, but that's not exactly what we need in
        # the skyc file: we need the last point, and the inside points
        bezier_curves = [[element[0]] + list(zip(*list(element[1:]))) for element in bpoly_pts]
        for bezier_curve in bezier_curves:
            curve_to_append = [bezier_curve[0],
                               bezier_curve[-1],
                               bezier_curve[2:-1]]
            self.bezier_repr.append(curve_to_append)

    def export_json(self, write_file: bool = True):
        """
        Returns the json formatted string of the bezier representation, and also writes it to a file if we wish.
        """
        # this is the format that a TrajectorySpecification requires:
        json_dict = {
            "version": 1,
            "points": self.bezier_repr,
            "takeoffTime": self.bezier_repr[0][0],
            "landingTime": self.bezier_repr[-1][0],
            "type": self.type
        }
        json_object = json.dumps(json_dict, indent=2)
        if write_file:
            with open("trajectory.json", "w") as f:
                f.write(json_object)
        return json_object


def cleanup(files: List[str], folders: List[str]):
    """
    function meant for deleting unnecessary files
    """
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted {file}")
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Deleted {folder} folder")


def write_skyc(trajectories: List[Trajectory], name=sys.argv[0][:-3]):
    """
    Constructs a skyc file from the provided trajectory, with the given name.
    """
    cleanup(files=["show.json",
                   "cues.json",
                   f"{name}.zip",
                   f"{name}.skyc",
                   "trajectory.json"],
            folders=["drones"])
    # Create the 'drones' folder if it doesn't already exist
    os.makedirs('drones', exist_ok=True)
    drones = []
    for index, traj in enumerate(trajectories):
        Data = traj.bezier_repr
        parameters = traj.parameters
        # The trajectory is saved to a json file with the data below
        traj.export_json()
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
        items = [{"time": 0.0,
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

        print('Compression complete. The files and folder have been zipped.')

        os.rename(f'{name}.zip', f'{name}.skyc')
        # Delete everything that's not 'trajectory.skyc'
        cleanup(files=["show.json",
                       "cues.json",
                       f"{name}.zip",
                       "trajectory.json"],
                folders=["drones"])
        print(f"{name}.skyc ready!")


def open_file_dialog(file_path_var: List[Optional[str]], root: tk.Tk, filetype: str) -> None:
    """
    Helper function for select_file, which handles the selection window.
    """
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
    # do this, just let me pass this by reference, horrible toy language for children and phds...)
    selected_file = [None]
    button_func = partial(open_file_dialog, selected_file, selecter, filetype)
    button = tk.Button(selecter, text=f"Select {filetype} file!", command=button_func, width=20, height=4)
    button.pack(pady=20)
    selecter.mainloop()
    return selected_file[0]


def evaluate_pickle(pickle_name: str, *, x_offset=0.0, y_offset=0.0, z_offset=0.0, yaw_offset=0.0):
    """
    Helper function to evaluate pickles made by Dr. Prof. PhD Antal Péter
    """
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