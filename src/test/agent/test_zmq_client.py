import time

import pytest

np = pytest.importorskip("numpy")

from stretch.agent.zmq_client import HomeRobotZmqClient


@pytest.fixture
def fast_sleep(monkeypatch):
    """Speed up tests that rely on sleep calls."""

    monkeypatch.setattr("time.sleep", lambda *_, **__: None)


@pytest.fixture
def make_client(monkeypatch):
    def _factory(ee_pose: np.ndarray, *, manipulation_mode: bool = True):
        client = HomeRobotZmqClient.__new__(HomeRobotZmqClient)
        client._ee_pos_tolerance = 0.01
        client._finish = False
        client._iter = 0
        client._seq_id = 0
        client._moving_threshold = 0.01
        client._angle_threshold = 0.01
        client._min_steps_not_moving = 1

        client.in_manipulation_mode = lambda: manipulation_mode

        actions = []

        def send_action(action, reliable=True, **_):
            actions.append(action)
            return {"step": 0}

        client.send_action = send_action
        client.get_ee_pose2 = lambda: ee_pose

        return client, actions

    return _factory


def test_arm_to_ee_pose_sends_action_and_succeeds(monkeypatch, fast_sleep, make_client):
    target_pos = np.array([0.1, -0.2, 0.3])
    target_quat = np.array([0.0, 0.0, 0.0, 1.0])
    ee_pose = np.eye(4, dtype=float)
    ee_pose[:3, 3] = target_pos

    client, actions = make_client(ee_pose)

    result = client.arm_to_ee_pose(
        pos=target_pos,
        quat=target_quat,
        gripper=0.25,
        relative=True,
        blocking=True,
        timeout=0.1,
    )

    assert result is True
    assert len(actions) == 1
    action = actions[0]
    assert np.allclose(action["ee_pose"]["pos"], target_pos)
    assert np.allclose(action["ee_pose"]["quat"], target_quat)
    assert action["gripper"] == 0.25
    assert action["relative"] is True
    assert action["manip_blocking"] is True


def test_arm_to_ee_pose_times_out(monkeypatch, fast_sleep, make_client):
    target_pos = np.array([0.0, 0.0, 0.0])
    ee_pose = np.eye(4, dtype=float)
    ee_pose[:3, 3] = [1.0, 1.0, 1.0]

    client, _ = make_client(ee_pose)

    result = client.arm_to_ee_pose(
        pos=target_pos,
        blocking=True,
        timeout=0.01,
    )

    assert result is False


def test_arm_to_ee_pose_requires_manipulation_mode(make_client):
    target_pos = np.array([0.1, 0.0, 0.0])
    ee_pose = np.eye(4, dtype=float)
    ee_pose[:3, 3] = target_pos

    client, _ = make_client(ee_pose, manipulation_mode=False)

    with pytest.raises(ValueError):
        client.arm_to_ee_pose(pos=target_pos)
