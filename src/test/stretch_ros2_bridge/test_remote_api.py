import pathlib
import sys

import pytest

np = pytest.importorskip("numpy", reason="numpy is required for stretch imports")

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from stretch.core.robot import ControlMode
from stretch.motion import conversions
from stretch.motion.kinematics import HelloStretchIdx, HelloStretchKinematics
from stretch_ros2_bridge.remote.api import StretchClient


ROUND_DECIMALS = 4


def _round_vec(values, decimals=ROUND_DECIMALS):
    return tuple(np.round(values, decimals=decimals).tolist())


class FakeManip:
    def __init__(self, pose_to_joints=None):
        self.pose_to_joints = pose_to_joints or {}
        self.last_joint_positions = None
        self.goto_joint_positions_calls = []
        self.goto_ee_pose_calls = []

    def goto_joint_positions(
        self, joint_positions, gripper=None, head_pan=None, head_tilt=None, blocking=False, debug=False
    ):
        self.last_joint_positions = list(joint_positions)
        self.goto_joint_positions_calls.append(
            {
                "joint_positions": list(joint_positions),
                "gripper": gripper,
                "head_pan": head_pan,
                "head_tilt": head_tilt,
                "blocking": blocking,
                "debug": debug,
            }
        )

    def get_joint_positions(self):
        return self.last_joint_positions

    def goto_ee_pose(self, pos, quat=None, relative=False, blocking=True):
        self.goto_ee_pose_calls.append(
            {
                "pos": list(pos),
                "quat": None if quat is None else list(quat),
                "relative": relative,
                "blocking": blocking,
            }
        )
        key = (_round_vec(pos), None if quat is None else _round_vec(quat), relative)
        if key not in self.pose_to_joints:
            raise RuntimeError("No IK solution")
        self.goto_joint_positions(self.pose_to_joints[key], blocking=blocking)
        return True


@pytest.fixture
def make_client():
    def _factory(manip):
        client = StretchClient.__new__(StretchClient)
        client.manip = manip
        client._base_control_mode = ControlMode.IDLE
        return client

    return _factory


@pytest.fixture
def joint_pose_pairs():
    rng = np.random.default_rng(0)
    model = HelloStretchKinematics()
    pairs = []

    for _ in range(5):
        q = np.zeros(model.dof)
        q[HelloStretchIdx.BASE_X] = rng.uniform(-0.15, 0.15)
        q[HelloStretchIdx.BASE_Y] = rng.uniform(-0.1, 0.1)
        q[HelloStretchIdx.BASE_THETA] = rng.uniform(-np.pi, np.pi)
        q[HelloStretchIdx.LIFT] = rng.uniform(0.1, 0.8)
        q[HelloStretchIdx.ARM] = rng.uniform(0.0, 0.5)
        q[HelloStretchIdx.GRIPPER] = rng.uniform(-0.2, 0.4)
        q[HelloStretchIdx.WRIST_ROLL] = rng.uniform(-1.0, 1.0)
        q[HelloStretchIdx.WRIST_PITCH] = rng.uniform(-1.0, 1.0)
        q[HelloStretchIdx.WRIST_YAW] = rng.uniform(-1.0, 1.0)
        q[HelloStretchIdx.HEAD_PAN] = 0.0
        q[HelloStretchIdx.HEAD_TILT] = 0.0

        pos, quat = model.manip_fk(q)
        pairs.append(
            {
                "joint_goal": conversions.config_to_manip_command(q),
                "pose_key": (_round_vec(pos), _round_vec(quat)),
                "pose_args": (pos.tolist(), quat.tolist()),
            }
        )

    return pairs


def test_arm_to_ee_pose_matches_joint_command(make_client):
    q_target = [0.2, 0.1, -0.05, 0.3, -0.2, 0.1]
    pos = (0.5, -0.1, 0.8)
    quat = (0.0, 0.0, 0.0, 1.0)
    manip_arm = FakeManip()
    client_arm = make_client(manip_arm)

    client_arm.arm_to(q_target, gripper=0.25, head_pan=0.05, head_tilt=-0.1, blocking=True, timeout=0.2)
    joint_goal_from_arm = manip_arm.last_joint_positions

    manip_pose = FakeManip({(pos, quat, False): q_target})
    client_pose = make_client(manip_pose)

    success = client_pose.arm_to_ee_pose(list(pos), list(quat), relative=False, blocking=True)

    assert success is True
    assert manip_pose.last_joint_positions == joint_goal_from_arm
    assert manip_pose.goto_ee_pose_calls[-1]["blocking"] is True


def test_arm_to_ee_pose_returns_false_on_error(make_client):
    manip_pose = FakeManip()
    client_pose = make_client(manip_pose)

    result = client_pose.arm_to_ee_pose([0.0, 0.0, 0.0], quat=None, blocking=True)

    assert result is False
    assert manip_pose.goto_ee_pose_calls[-1]["blocking"] is True


def test_arm_to_ee_pose_replays_saved_joint_pose_pairs(make_client, joint_pose_pairs):
    pose_to_joint = {
        (pos, quat, False): data["joint_goal"] for data in joint_pose_pairs for pos, quat in [data["pose_key"]]
    }
    manip_pose = FakeManip(pose_to_joint)
    client_pose = make_client(manip_pose)

    for pair in joint_pose_pairs:
        manip_arm = FakeManip()
        client_arm = make_client(manip_arm)

        client_arm.arm_to(pair["joint_goal"], blocking=True)
        expected_joint_goal = manip_arm.last_joint_positions

        success = client_pose.arm_to_ee_pose(*pair["pose_args"], relative=False, blocking=True)

        assert success is True
        assert manip_pose.last_joint_positions == expected_joint_goal

    assert len(manip_pose.goto_ee_pose_calls) == len(joint_pose_pairs)
