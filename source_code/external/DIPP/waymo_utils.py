"""
This file is taken from https://github.com/MCZhi/DIPP
"""
from typing import Optional, Tuple, Dict

from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate

import math
import numpy as np
import bisect


class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        """
        :param x: keys of the data points (e.g. time or longitudinal position)
        :param y: values of data points (which will be interpolated by the spline)
        """
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x) + 1e-3

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t: float) -> Optional[float]:
        """
        Evaluate spline for value t.
        if t is outside of the input x, return None
        :param t: value at which spline is evaluated
        :return value at t
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t) -> Optional[float]:
        """
        Calc first derivative at value t
        if t is outside of the input x, return None
        :param t: value at which spline is evaluated
        :return first derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0

        return result

    def calcdd(self, t) -> Optional[float]:
        """
        Calc second derivative
        :param t: value at which spline is evaluated
        :return second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx

        return result

    def __search_index(self, x) -> int:
        """
        search data segment index
        :param x: values to search for
        :return index lower than x
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h) -> np.ndarray:
        """
        calc matrix A for spline coefficient c
        :param h: coefficients
        :return A: matrix
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0

        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0

        return A

    def __calc_B(self, h) -> np.ndarray:
        """
        calc matrix B for spline coefficient c
        :param h coefficients
        :return B matrix
        """
        B = np.zeros(self.nx)

        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]

        return B


class Spline2D:
    """
    2D Cubic Spline class where each dimension is represented by a 1D spline mapping the arc length to the
    respective coordinate.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        :param x: x coordinates of a 2D path
        :param y: y coordinates of a 2D path
        """
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y) -> list:
        """
        Calculate arc length s at each point.
        :param x: x coordinates of a 2D path
        :param y: y coordinates of a 2D path
        :return: arc length s
        """
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))

        return s

    def calc_position(self, s: float) -> Tuple[float, float]:
        """
        Return position at arc length
        :param s: arc length
        :return: position tuple
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s: float) -> float:
        """
        Return curvature at arc length
        :param s: arc length
        :return: curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))

        return k

    def calc_yaw(self, s: float) -> float:
        """
        Return yaw at arc length
        :param s: arc length
        :return: yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)

        return yaw


def calc_spline_course(x: np.ndarray, y: np.ndarray, ds=0.1) -> tuple:
    """
    Calculate states consisting of [x, y, yaw and arc length] for a path using a spline approximation.
    :param x: x coordinates of path
    :param y: y coordinates of path
    :param ds: step size arc length
    :return: one list for each state variable in [x, y, yaw and arc length]
    """
    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []

    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s


def wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    """
    Wrap angle to interval [-pi, pi]
    :param theta: angle
    :return: wrapped angle
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


def compute_direction_diff(ego_theta: float, target_theta: float) -> np.ndarray:
    """
    Difference between two angles wrapped between +-PI.
    :param ego_theta: first angle
    :param target_theta: second angle
    :return: ego_theta - target_theta
    """
    delta = np.abs(ego_theta - target_theta)
    delta = np.where(delta > np.pi, 2 * np.pi - delta, delta)

    return delta


def depth_first_search(cur_lane, lanes, dist=0, threshold=100, visited_lanes=None):
    """
    Perform depth first search over lane graph up to the threshold.
    :param cur_lane: Starting lane_id
    :param lanes: raw lane data
    :param dist: Distance of the current path
    :param threshold: Threshold after which to stop the search
    :return lanes_to_return (list of list of integers): List of sequence of lane ids
    """
    visited_lanes = visited_lanes if visited_lanes is not None else set()
    if dist > threshold:
        return [[cur_lane]]
    else:
        child_lanes = set(lanes[cur_lane].exit_lanes)  # | get_neighbors(lanes[cur_lane])
        child_lanes -= visited_lanes
        traversed_lanes = []
        if child_lanes:
            for child in child_lanes:
                centerline = np.array([(map_point.x, map_point.y, map_point.z) for map_point in lanes[child].polyline])
                cl_length = centerline.shape[0]
                curr_lane_ids = depth_first_search(child,
                                                   lanes,
                                                   dist + cl_length, threshold,
                                                   visited_lanes=visited_lanes)
                traversed_lanes.extend(curr_lane_ids)
                for lane_ids in curr_lane_ids:
                    visited_lanes = visited_lanes | set(lane_ids)

        if len(traversed_lanes) == 0:
            return [[cur_lane]]

        lanes_to_return = []

        for lane_seq in traversed_lanes:
            lanes_to_return.append([cur_lane] + lane_seq)

        return lanes_to_return


def polygon_completion(polygon) -> np.ndarray:
    """
    Create polygon represented by polylines from waymo's polyline objects.
    :param polygon: parsed waymo protobufs
    :return: polylines
    """
    polyline_x = []
    polyline_y = []

    for i in range(len(polygon)):
        if i + 1 < len(polygon):
            next = i + 1
        else:
            next = 0

        dist_x = polygon[next].x - polygon[i].x
        dist_y = polygon[next].y - polygon[i].y
        dist = np.linalg.norm([dist_x, dist_y])
        interp_num = np.ceil(dist) * 2
        interp_index = np.arange(2 + interp_num)
        point_x = np.interp(interp_index, [0, interp_index[-1]], [polygon[i].x, polygon[next].x]).tolist()
        point_y = np.interp(interp_index, [0, interp_index[-1]], [polygon[i].y, polygon[next].y]).tolist()
        polyline_x.extend(point_x[:-1])
        polyline_y.extend(point_y[:-1])

    polyline_x, polyline_y = np.array(polyline_x), np.array(polyline_y)
    polyline_heading = wrap_to_pi(np.arctan2(polyline_y[1:] - polyline_y[:-1], polyline_x[1:] - polyline_x[:-1]))
    polyline_heading = np.insert(polyline_heading, -1, polyline_heading[-1])

    return np.stack([polyline_x, polyline_y, polyline_heading], axis=1)


def get_polylines(lines) -> Dict["Line", np.ndarray]:
    """
    Get poylines from line objects
    :param lines: polylines in waymo protobuf format
    :return: maps line objects to polylines
    """
    polylines = {}

    for line in lines.keys():
        polyline = np.array([(map_point.x, map_point.y) for map_point in lines[line].polyline])
        if len(polyline) > 1:
            direction = wrap_to_pi(np.arctan2(polyline[1:, 1] - polyline[:-1, 1], polyline[1:, 0] - polyline[:-1, 0]))
            direction = np.insert(direction, -1, direction[-1])[:, np.newaxis]
        else:
            direction = np.array([0])[:, np.newaxis]
        polylines[line] = np.concatenate([polyline, direction], axis=-1)

    return polylines


def find_reference_lanes(agent_type, agent_traj, lanes, distance_threshold, visited_lane_ids=None) -> dict:
    """
    Find reference lane for an agent
    :param agent_type: type of agent
    :param agent_traj: trajectory of agent
    :param lanes: vectorized lanes
    :param distance_threshold: max. distance for adding lanes
    :param visited_lane_ids: keeps already added lanes that are ignored in this search
    :return: dict maps lane ID to closest index of that lane's center line
    """
    curr_lane_ids = {}
    visited_lane_ids = visited_lane_ids or {}
    if agent_type == 2:
        distance_threshold = distance_threshold
        for lane in lanes.keys():
            if lane not in visited_lane_ids and lanes[lane].shape[0] > 1:
                distance_to_agent = LineString(lanes[lane][:, :2]).distance(Point(agent_traj[-1, :2]))
                if distance_to_agent < distance_threshold:
                    curr_lane_ids[lane] = 0

    else:
        distance_threshold = 3.5
        direction_threshold = 10

        while len(curr_lane_ids) < 1:
            for lane in lanes.keys():
                if lane not in visited_lane_ids:
                    distance_to_ego = np.linalg.norm(agent_traj[-1, :2] - lanes[lane][:, :2], axis=-1)
                    direction_to_ego = compute_direction_diff(agent_traj[-1, 2], lanes[lane][:, -1])
                    for i, j, k in zip(distance_to_ego, direction_to_ego, range(distance_to_ego.shape[0])):
                        if i <= distance_threshold:  # and j <= np.radians(direction_threshold):
                            curr_lane_ids[lane] = k
                            break

            distance_threshold += 3.5
            direction_threshold += 10

    return curr_lane_ids


def find_neighbor_lanes(curr_lane_ids, traj, lanes, lane_polylines) -> dict:
    """
    Find neighbouring lanes of a lane at the location of the last state of a trajectory.
    :param curr_lane_ids: lanes for which neighbours are searched
    :param traj: trajectory that is considered in the search
    :param lanes: lanes from which neighbors can be selected
    :param lane_polylines: dict with centerlines of lanes, {lane_id: polyline}
    :return: dictionary with lane IDs mapping to the closest point in the neighbouring lane
    """
    neighbor_lane_ids = {}

    for curr_lane, start in curr_lane_ids.items():
        left_lanes = lanes[curr_lane].left_neighbors
        right_lanes = lanes[curr_lane].right_neighbors
        left_lane = None
        right_lane = None
        curr_index = start

        for l_lane in left_lanes:
            if l_lane.self_start_index <= curr_index <= l_lane.self_end_index and not l_lane.feature_id in curr_lane_ids:
                left_lane = l_lane

        for r_lane in right_lanes:
            if r_lane.self_start_index <= curr_index <= r_lane.self_end_index and not r_lane.feature_id in curr_lane_ids:
                right_lane = r_lane

        if left_lane is not None:
            left_polyline = lane_polylines[left_lane.feature_id]
            start = np.argmin(np.linalg.norm(traj[-1, :2] - left_polyline[:, :2], axis=-1))
            neighbor_lane_ids[left_lane.feature_id] = start

        if right_lane is not None:
            right_polyline = lane_polylines[right_lane.feature_id]
            start = np.argmin(np.linalg.norm(traj[-1, :2] - right_polyline[:, :2], axis=-1))
            neighbor_lane_ids[right_lane.feature_id] = start

    return neighbor_lane_ids


def find_neareast_point(curr_point, line) -> np.ndarray:
    """
    Find closest point in a given polyline
    :param curr_point: a single point
    :param line: polyline from which closest point is selected
    :return: nearest point
    """
    distance_to_curr_point = np.linalg.norm(curr_point[np.newaxis, :2] - line[:, :2], axis=-1)
    neareast_point = line[np.argmin(distance_to_curr_point)]

    return neareast_point


def generate_target_course(x, y) -> tuple:
    """
    Create reference path by evenly resampling a path provided by data points.
    :param x: x coordinates of path
    :param y: y coordinates of path
    :return: resampled x values, resampled y values, yaw values, curvature values, spline object
    """
    csp = Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


direction_threshold = math.radians(10)


def find_map_waypoint(pos: np.ndarray, polylines: Dict[int, np.ndarray]) -> tuple:
    """
    Find waypoints in lanes that is closest to a provided position.
    :param pos: position for which the closest waypoint is searched
    :param polylines: center lines of lanes
    :return: lane ID and index of the closest waypoint
    """
    waypoint = [-1, -1, 1e9, 1e9]

    for id, polyline in polylines.items():
        distance_to_gt = np.linalg.norm(pos[np.newaxis, :2] - polyline[:, :2], axis=-1)
        direction_to_gt = compute_direction_diff(pos[np.newaxis, 2], polyline[:, 2])
        for i, j, k in zip(range(polyline.shape[0]), distance_to_gt, direction_to_gt):
            if j < waypoint[2] and k <= direction_threshold - k:
                waypoint = [id, i, j, k]

    lane_id = waypoint[0]
    waypoint_id = waypoint[1]

    if lane_id > 0:
        return lane_id, waypoint_id
    else:
        return None, None


def find_route(traj: np.ndarray, timestep: int, cur_pos: np.ndarray, map_lanes,
               map_crosswalks, map_signals) -> np.ndarray:
    """
    Find route composed of center lines of connected lanes that reflect a reference path for a given ground truth
    trajectory.
    :param traj: trajectory for which reference path is computed
    :param timestep: time step of initial trajectory state
    :param cur_pos: initial position of trajectory
    :param map_lanes: waymo map container
    :param map_crosswalks: waymo crosswalk container
    :param map_signals: traffic signals
    :return: reference path
    """
    lane_polylines = get_polylines(map_lanes)
    end_lane, end_point = find_map_waypoint(np.array((traj[-1].center_x, traj[-1].center_y, traj[-1].heading)),
                                            lane_polylines)
    start_lane, start_point = find_map_waypoint(np.array((traj[0].center_x, traj[0].center_y, traj[0].heading)),
                                                lane_polylines)
    cur_lane, _ = find_map_waypoint(cur_pos, lane_polylines)

    path_waypoints = []
    for t in range(0, len(traj), 10):
        lane, point = find_map_waypoint(np.array((traj[t].center_x, traj[t].center_y, traj[t].heading)), lane_polylines)
        path_waypoints.append(lane_polylines[lane][point])

    before_waypoints = []
    if start_point < 40:
        if map_lanes[start_lane].entry_lanes:
            lane = map_lanes[start_lane].entry_lanes[0]
            for waypoint in lane_polylines[lane]:
                before_waypoints.append(waypoint)
    for waypoint in lane_polylines[start_lane][:start_point]:
        before_waypoints.append(waypoint)

    after_waypoints = []
    for waypoint in lane_polylines[end_lane][end_point:]:
        after_waypoints.append(waypoint)
    if len(after_waypoints) < 40:
        if map_lanes[end_lane].exit_lanes:
            lane = map_lanes[end_lane].exit_lanes[0]
            for waypoint in lane_polylines[lane]:
                after_waypoints.append(waypoint)

    waypoints = np.concatenate([before_waypoints[::5], path_waypoints, after_waypoints[::5]], axis=0)

    # generate smooth route
    tx, ty, tyaw, tc, _ = generate_target_course(waypoints[:, 0], waypoints[:, 1])
    ref_line = np.column_stack([tx, ty, tyaw, tc])

    # get reference path at current timestep
    current_location = np.argmin(np.linalg.norm(ref_line[:, :2] - cur_pos[np.newaxis, :2], axis=-1))
    start_index = np.max([current_location - 200, 0])
    ref_line = ref_line[start_index:start_index + 1200]

    # add speed limit, crosswalk, and traffic signal info to ref route
    line_info = np.zeros(shape=(ref_line.shape[0], 1))
    speed_limit = map_lanes[cur_lane].speed_limit_mph / 2.237
    ref_line = np.concatenate([ref_line, line_info], axis=-1)
    if map_crosswalks is not None:
        crosswalks = [Polygon([(point.x, point.y) for point in crosswalk.polygon])
                      for _, crosswalk in map_crosswalks.items()]
    else:
        crosswalks = []

    signals = [Point([signal.stop_point.x, signal.stop_point.y])
               for signal in map_signals[timestep].lane_states if signal.state in [1, 4, 7]]

    for i in range(ref_line.shape[0]):
        if any([Point(ref_line[i, :2]).distance(signal) < 0.2 for signal in signals]):
            ref_line[i, 4] = 0  # red light
        elif any([crosswalk.contains(Point(ref_line[i, :2])) for crosswalk in crosswalks]):
            ref_line[i, 4] = 1  # crosswalk
        else:
            ref_line[i, 4] = speed_limit

    return ref_line


def imputer(traj: np.ndarray) -> np.ndarray:
    """
    Extrapolate short trajectories where x=0.
    :param traj: trajectory to be extrapolated
    :return: extrapolated trajectory
    """
    x, y, v_x, v_y, theta = traj[:, 0], traj[:, 1], traj[:, 3], traj[:, 4], traj[:, 2]

    if np.any(x == 0):
        for i in reversed(range(traj.shape[0])):
            if x[i] == 0:
                v_x[i] = v_x[i + 1]
                v_y[i] = v_y[i + 1]
                x[i] = x[i + 1] - v_x[i] * 0.1
                y[i] = y[i + 1] - v_y[i] * 0.1
                theta[i] = theta[i + 1]
        return np.column_stack((x, y, theta, v_x, v_y))
    else:
        return np.column_stack((x, y, theta, v_x, v_y))


def agent_norm(traj, center, angle, impute=False) -> np.ndarray:
    """
    Transform trajectory with respect to 2D reference pose.
    :param traj: trajectory to be transformed
    :param center: center of reference pose
    :param angle: orientation of reference pose
    :param impute: extrapolate short trajectories
    :return: transformed map element
    """
    if impute:
        traj = imputer(traj[:, :5])

    line = LineString(traj[:, :2])
    line_offset = affine_transform(line, [1, 0, 0, 1, -center[0], -center[1]])
    line_rotate = rotate(line_offset, -angle, origin=(0, 0), use_radians=True)
    line_rotate = np.array(line_rotate.coords)
    line_rotate[traj[:, :2] == 0] = 0
    heading = wrap_to_pi(traj[:, 2] - angle)
    heading[traj[:, 2] == 0] = 0

    if traj.shape[-1] > 3:
        velocity_x = traj[:, 3] * np.cos(angle) + traj[:, 4] * np.sin(angle)
        velocity_x[traj[:, 3] == 0] = 0
        velocity_y = traj[:, 4] * np.cos(angle) - traj[:, 3] * np.sin(angle)
        velocity_y[traj[:, 4] == 0] = 0
        return np.column_stack((line_rotate, heading, velocity_x, velocity_y))
    else:
        return np.column_stack((line_rotate, heading))


def map_norm(map_line, center, angle) -> np.ndarray:
    """
    Transform map element with respect to 2D reference pose.
    :param map_line: polyline representing a map element
    :param center: center of reference pose
    :param angle: orientation of reference pose
    :return: transformed map element
    """
    self_line = LineString(map_line[:, 0:2])
    self_line = affine_transform(self_line, [1, 0, 0, 1, -center[0], -center[1]])
    self_line = rotate(self_line, -angle, origin=(0, 0), use_radians=True)
    self_line = np.array(self_line.coords)
    self_line[map_line[:, 0:2] == 0] = 0
    self_heading = wrap_to_pi(map_line[:, 2] - angle)

    if map_line.shape[1] > 3:
        left_line = LineString(map_line[:, 3:5])
        left_line = affine_transform(left_line, [1, 0, 0, 1, -center[0], -center[1]])
        left_line = rotate(left_line, -angle, origin=(0, 0), use_radians=True)
        left_line = np.array(left_line.coords)
        left_line[map_line[:, 3:5] == 0] = 0
        left_heading = wrap_to_pi(map_line[:, 5] - angle)
        left_heading[map_line[:, 5] == 0] = 0

        right_line = LineString(map_line[:, 6:8])
        right_line = affine_transform(right_line, [1, 0, 0, 1, -center[0], -center[1]])
        right_line = rotate(right_line, -angle, origin=(0, 0), use_radians=True)
        right_line = np.array(right_line.coords)
        right_line[map_line[:, 6:8] == 0] = 0
        right_heading = wrap_to_pi(map_line[:, 8] - angle)
        right_heading[map_line[:, 8] == 0] = 0

        return np.column_stack((self_line, self_heading, left_line, left_heading, right_line, right_heading))
    else:
        return np.column_stack((self_line, self_heading))


def ref_line_norm(ref_line: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    """
    Transform reference path with respect to 2D reference pose.
    :param ref_line: reference path to be transformed
    :param center: position of reference pose
    :param angle: orientation of reference pose
    :return: transformed reference path
    """
    xy = LineString(ref_line[:, 0:2])
    xy = affine_transform(xy, [1, 0, 0, 1, -center[0], -center[1]])
    xy = rotate(xy, -angle, origin=(0, 0), use_radians=True)
    yaw = wrap_to_pi(ref_line[:, 2] - angle)
    c = ref_line[:, 3]
    info = ref_line[:, 4]

    return np.column_stack((xy.coords, yaw, c, info))
