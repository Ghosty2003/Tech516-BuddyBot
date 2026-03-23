# TurtleBot3 Human-Following Robot

> **YOLO · MediaPipe · Finite State Machine · ROS 2**

![System Overview](assets/Photo.jpg)

---

## Table of Contents

- [Project Overview](#1-project-overview)
- [System Architecture](#2-system-architecture)
- [Module Descriptions](#3-module-descriptions)
  - [BiggestPersonTracker](#31-biggestpersontracker--trackerpy)
  - [HumanActionPublisher](#32-humanactionpublisher--action_publisherpy)
  - [FollowerFSMV2](#33-followerfsm v2--follower_fsm_v2py)
  - [PlatformHeightController](#34-platformheightcontroller--platform_ctrlpy)
- [Installation & Setup](#4-installation--setup)
- [Usage Guide](#5-usage-guide)
- [Configuration Parameters](#6-configuration-parameters)
- [Known Limitations & Future Work](#7-known-limitations--future-work)
- [Licence](#8-licence)

---

## 1. Project Overview

This project implements a **gesture-driven human-following robot** on the TurtleBot3 platform using ROS 2. The robot uses a **camera-only perception pipeline** — no LiDAR — to detect and track a target person, interpret posture and gesture commands in real time, and adjust its movement and an onboard height-adjustable platform accordingly.

### Key Features

- **Camera-only perception**: YOLO v11n for person detection, MediaPipe Pose for skeleton-based action recognition
- **Gesture command interface**: wave to start/stop service, reach/sit/stand to control approach distance
- **Spin-search recovery**: automatically rotates to re-acquire a lost target
- **Adaptive platform height**: raises/lowers a motorised platform based on the human's posture
- **Finite-State Machine (FSM)** architecture for robust, deterministic behaviour

---

## 2. System Architecture

The system is composed of four ROS 2 nodes that communicate over topics:


| Node | File | Subscribes | Publishes |
|---|---|---|---|
| `BiggestPersonTracker` | `tracker.py` | `/image_raw/compressed` | `/target_person_pos` |
| `HumanActionPublisher` | `action_publisher.py` | `/image_raw/compressed` | `/human_action` |
| `FollowerFSMV2` | `follower_fsm_v2.py` | `/image_raw/compressed`, `/human_action` | `/cmd_vel` |
| `PlatformHeightController` | `platform_ctrl.py` | `/human_action` | `/gix_controller/joint_trajectory` |

---

## 3. Module Descriptions

### 3.1 BiggestPersonTracker (`tracker.py`)

Subscribes to the raw compressed camera feed, runs YOLO v11n inference on every frame to detect all persons (class 0), and selects the bounding box with the **largest pixel area** as the primary tracking target. The centre coordinates and area are published as a `Pose2D` message so that other nodes can use area as a proxy for distance.

**Published topic: `/target_person_pos` (`geometry_msgs/Pose2D`)**

| Field | Meaning |
|---|---|
| `x` | Horizontal pixel centre of the bounding box |
| `y` | Vertical pixel centre of the bounding box |
| `theta` | Bounding-box pixel area (distance proxy) |

---

### 3.2 HumanActionPublisher (`action_publisher.py`)

Uses **MediaPipe Pose** to extract 33 skeleton landmarks from each camera frame at **5 Hz**. Joint angles are computed between hip–knee–ankle (leg angle) and shoulder–elbow–wrist (arm angle) to classify the human's current posture and gesture.


**Recognition Logic**

| Condition | Label Published |
|---|---|
| Wrist above shoulder | `Wave` |
| Arm angle < 45° | `Reach Out` |
| Both leg angles < 120° | `Sitting` |
| Both leg angles > 160° | `Standing` |

**Stability features:**
- **Debouncing**: requires 3 consecutive identical detections before confirming a label
- **Cooldown**: 3-second cooldown prevents repeated `Sitting` / `Standing` publications

---

### 3.3 FollowerFSMV2 (`follower_fsm_v2.py`)

The main motion controller. Consumes both the raw camera feed (for real-time bounding-box width feedback) and `/human_action` (for state transitions). A **Finite State Machine** drives velocity commands on `/cmd_vel`.

![FSM Diagram](assets/FSM.png)
FSM state transition diagram*

**States & Transitions**

| State | Entry Trigger | Robot Behaviour |
|---|---|---|
| `IDLE` | Default / 2nd wave | Stationary; ignores detections |
| `FOLLOW` | 1st wave / walk posture | Moves forward if `bw < 300 px`; faces human |
| `APPROACH_60` | stand posture | Approaches until `bw ≥ 300 px`, then stops |
| `APPROACH_30` | sit/squat, or reach+stand | Approaches until `bw ≥ 450 px`, then stops |
| `APPROACH_20` | reach + sit | Approaches until `bw ≥ 550 px`, then stops |
| `RETREAT_60` | sit → stand transition | Reverses until `bw ≤ 300 px`, then → `FOLLOW` |

**Spin-Search Recovery**

If no person is detected for `lost_threshold = 5` consecutive frames and the robot has not yet achieved its current target, it rotates in place at `0.5 rad/s` until the person re-enters the frame.

**Heading Control**

When the service is active the robot continuously corrects its yaw to keep the human centred. A ±40 px deadband suppresses micro-oscillations. Angular velocity is proportional to pixel drift:

```
ω = −0.002 × drift
```

**Safety Stop**

If bounding-box width exceeds `600 px` at any time, linear motion stops immediately regardless of FSM state.

---

### 3.4 PlatformHeightController (`platform_ctrl.py`)

Controls a motorised height-adjustable platform via `JointTrajectory` messages. The platform height mirrors the human's posture so that a seated user's eye level is matched by the robot's onboard interface.


| Posture | Platform Height | Joint Angle |
|---|---|---|
| sit / squat | 40 cm | 0 rad (0°) |
| stand / walk | 90 cm | ~6.28 rad (360°) |

The system is activated by the **first wave only**; all subsequent waves are ignored by this node.

---

## 4. Installation & Setup

### 4.1 Prerequisites

- Ubuntu 22.04 + ROS 2 Humble
- Python 3.10+
- TurtleBot3 Waffle Pi (or Gazebo simulation)
- USB / onboard camera publishing to `/image_raw/compressed`

### 4.2 Python Dependencies

```bash
pip install ultralytics mediapipe opencv-python numpy
```

### 4.3 Model Weights

The YOLO v11 nano weights (`yolo11n.pt`) are downloaded automatically by `ultralytics` on first run. Ensure internet access, or pre-download and place the file in the working directory.

### 4.4 Running the Nodes

Open one terminal with ros2 after build:

```bash
ros2 launch buddybot launch.launch.py
```

---

## 5. Usage Guide

### 5.1 Gesture Command Reference

![Design](assets/design.png)

| Gesture | Posture | Robot Response | Platform |
|---|---|---|---|
| Wave (1st) | Any | Start following service | — |
| Wave (2nd) | Any | Stop service, hold position | — |
| None | Walk | Follow at ~60 cm | 90 cm |
| None | Stand | Approach to ~60 cm | 90 cm |
| None | Sit / Squat | Approach to ~30 cm | 40 cm |
| Reach | Stand / Walk | Approach to ~30 cm | 90 cm |
| Reach | Sit | Approach to ~20 cm | 40 cm |
| — | Sit → Stand | Retreat to ~60 cm | 90 cm |

### 5.2 Typical Interaction Flow

1. Stand in front of the robot within camera range (~2–3 m)
2. **Wave** → robot starts following
3. Walk freely; robot maintains ~60 cm following distance
4. **Sit down** → robot approaches to ~30 cm, platform lowers to 40 cm
5. **Reach** while seated → robot closes to ~20 cm
6. **Stand up** → robot retreats to ~60 cm, platform raises to 90 cm
7. **Wave again** → service stops, robot holds position

---

## 6. Configuration Parameters

All key parameters are defined at the top of `follower_fsm_v2.py`:

| Parameter | Default | Description |
|---|---|---|
| `lost_threshold` | `5` | Frames without detection before spin-search activates |
| `spin_speed` | `0.5 rad/s` | Angular speed during spin-search |
| `deadband` | `40 px` | Heading deadband; suppresses micro-oscillations |
| `k_ang` | `0.002` | Proportional gain for heading control |
| `bw_20 / 30 / 60` | `550 / 450 / 300 px` | Bounding-box width targets for approach states |
| `bw_stop` | `600 px` | Safety stop threshold |

---

## 7. Known Limitations & Future Work

**Current limitations:**

- **Single-person assumption**: the largest bounding box is tracked, which may not always be the intended user when multiple people are present
- **Area-based distance**: pixel area is affected by body orientation; a depth sensor would be more accurate
- **2-D skeleton only**: MediaPipe without camera calibration loses accuracy at steep viewing angles
- **Fixed bounding-box thresholds**: `bw` values are calibrated for a specific camera FOV and require re-tuning for a different lens

**Planned improvements:**

- Integrate LiDAR for metric distance control and obstacle avoidance
- Add person re-identification (Re-ID) to persist tracking across occlusions
- Extend gesture vocabulary with hand-landmark tracking (MediaPipe Hands)
- Package all nodes into a ROS 2 launch file for single-command startup

---

## 8. Licence

This project is released under the MIT Licence. See [LICENSE](LICENSE) for details.
