# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:05:04 2026

@author: Michael3080
"""

import sys, time, threading
import numpy as np
from PyQt5 import QtWidgets, QtCore
import os, json

import pybullet as p

# from ur5_22_2FandScoop import DualUR5EEGuiIK
from ur5_22_2FandScoop_0318 import DualUR5EEGuiIK
#%%

def get_arm_revolute_joints(rid, target_dofs=6):
    joints = []
    for j in range(p.getNumJoints(rid)):
        if p.getJointInfo(rid, j)[2] == p.JOINT_REVOLUTE:
            joints.append(j)
    return joints[:target_dofs]

def find_link(rid, candidates):
    name_to_idx = {}
    for j in range(p.getNumJoints(rid)):
        link_name = p.getJointInfo(rid, j)[12].decode("utf-8")
        name_to_idx[link_name] = j
    for c in candidates:
        if c in name_to_idx:
            return name_to_idx[c]
    return p.getNumJoints(rid) - 1

def get_joint_limit(rid, jid, default=(-2*np.pi, 2*np.pi)):
    info = p.getJointInfo(rid, jid)
    lo, hi = float(info[8]), float(info[9])
    if hi <= lo or abs(hi) > 1e8 or abs(lo) > 1e8:
        return default
    return (lo, hi)

def hide_body(body_id):
    nJ = p.getNumJoints(body_id)
    try:
        p.changeVisualShape(body_id, -1, rgbaColor=[1, 1, 1, 0])
    except:
        pass
    p.setCollisionFilterGroupMask(body_id, -1, 0, 0)
    for j in range(nJ):
        try:
            p.changeVisualShape(body_id, j, rgbaColor=[1, 1, 1, 0])
        except:
            pass
        p.setCollisionFilterGroupMask(body_id, j, 0, 0)

class ProbeFK:
    def __init__(self, urdf_path, base_pos, base_orn_euler):
        self.rid = p.loadURDF(
            urdf_path,
            basePosition=base_pos,
            baseOrientation=p.getQuaternionFromEuler(base_orn_euler),
            useFixedBase=True
        )
        hide_body(self.rid)
        self.jids = get_arm_revolute_joints(self.rid, 6)
        self.ee = find_link(self.rid, ["ee_link", "tool0", "wrist_3_link"])

    def fk(self, q6):
        for jid, qi in zip(self.jids, q6):
            p.resetJointState(self.rid, jid, float(qi))
        ls = p.getLinkState(self.rid, self.ee, computeForwardKinematics=True)
        pos = list(ls[4])
        orn = ls[5]
        rpy = list(p.getEulerFromQuaternion(orn))
        return np.array(pos + rpy, dtype=np.float32)

class SimThread(QtCore.QThread):
    sig_state = QtCore.pyqtSignal(dict)  # UI로 상태 전달

    def __init__(self, sim: DualUR5EEGuiIK, lock: threading.Lock, shared: dict, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.lock = lock
        self.shared = shared
        self._running = True
        self._cnt = 0

    def stop(self):
        self._running = False

    def _collect_arm_debug(self, rid, jids, ee_idx):
        joint_xyz = []
        for jid in jids:
            ls = p.getLinkState(rid, jid, computeForwardKinematics=True)
            joint_xyz.append([float(ls[4][0]), float(ls[4][1]), float(ls[4][2])])

        j6 = jids[-1]
        far1, far2, near_pt, _ = self.sim._get_gripper_extreme_points(rid, j6, ee_idx)
        yz_ang = self.sim._yz_angle_from_xy_plane_deg(far1, near_pt) if hasattr(self.sim, "_yz_angle_from_xy_plane_deg") else None
        return {
            "joint_xyz": joint_xyz,
            "far1": [float(x) for x in far1],
            "far2": [float(x) for x in far2],
            "near": [float(x) for x in near_pt],
            "yz_angle_deg": None if yz_ang is None else float(yz_ang),
        }

    def _collect_sack_debug(self):
        if not hasattr(self.sim, "_get_sack_state"):
            return None
        state = self.sim._get_sack_state()
        if state is None:
            return None
        return {
            "center": [float(x) for x in state["center"]],
            "rpy_deg": [float(x) for x in np.degrees(state["rpy"])],
            "size": [float(x) for x in state["size"]],
        }

    def run(self):
        try:
            fixed_dt = float(p.getPhysicsEngineParameters().get("fixedTimeStep", 1.0/240.0))
        except Exception:
            fixed_dt = 1.0/240.0
    
        while self._running and p.isConnected():
            t0 = time.perf_counter()
    
            with self.lock:
                qL = self.shared["L_q"].copy()
                qR = self.shared["R_q"].copy()
                sleep_dt = float(self.shared.get("sleep_dt", 0.0))
    
            self.sim._apply_q(self.sim.urL, self.sim.jL, list(qL), self.sim.maxF_L)
            self.sim._apply_q(self.sim.urR, self.sim.jR, list(qR), self.sim.maxF_R)
            # print("ur5py_134")
            p.stepSimulation()

            if hasattr(self.sim, "_update_sack_debug"):
                self.sim._update_sack_debug()

            if hasattr(self.sim, "_update_robot_realtime_debug"):
                self.sim._update_robot_realtime_debug()

            self._cnt += 1
    
            if self._cnt % 30 == 0:
                qL_now = [p.getJointState(self.sim.urL, jid)[0] for jid in self.sim.jL]
                qR_now = [p.getJointState(self.sim.urR, jid)[0] for jid in self.sim.jR]
                (pL, rL) = self.sim._get_ee_pose(self.sim.urL, self.sim.eeL)
                (pR, rR) = self.sim._get_ee_pose(self.sim.urR, self.sim.eeR)
                armL_dbg = self._collect_arm_debug(self.sim.urL, self.sim.jL, self.sim.eeL)
                armR_dbg = self._collect_arm_debug(self.sim.urR, self.sim.jR, self.sim.eeR)
                sack_dbg = self._collect_sack_debug()
                self.sig_state.emit({
                    "qL": qL_now, "qR": qR_now,
                    "eeL": pL+rL, "eeR": pR+rR,
                    "armL_dbg": armL_dbg, "armR_dbg": armR_dbg,
                    "sack_dbg": sack_dbg,
                })
    
            dt = sleep_dt if sleep_dt > 0 else fixed_dt
            elapsed = time.perf_counter() - t0
            remain = dt - elapsed
    
            if remain > 0:
                time.sleep(remain)
            else:
                time.sleep(0.001)

class Pendant(QtWidgets.QWidget):
    def __init__(self, sim: DualUR5EEGuiIK, left_urdf_path: str, right_urdf_path: str):
        super().__init__()
        self.setWindowTitle("UR5 Dual Pendant (PyQt) - EE <-> Joint Coupled")

        self.sim = sim
        self.lock = threading.Lock()

        self.shared = {
            "L_mode": 0,  # 0=EE, 1=Joint
            "R_mode": 0,
            "L_ee": np.array(self.sim._get_ee_pose(self.sim.urL, self.sim.eeL)[0] +
                             self.sim._get_ee_pose(self.sim.urL, self.sim.eeL)[1], dtype=np.float32),
            "R_ee": np.array(self.sim._get_ee_pose(self.sim.urR, self.sim.eeR)[0] +
                             self.sim._get_ee_pose(self.sim.urR, self.sim.eeR)[1], dtype=np.float32),
            "L_q": np.array([p.getJointState(self.sim.urL, jid)[0] for jid in self.sim.jL], dtype=np.float32),
            "R_q": np.array([p.getJointState(self.sim.urR, jid)[0] for jid in self.sim.jR], dtype=np.float32),
            "sleep_dt": 1.0/240.0,  # 0이면 가능한 빠르게(권장). 필요하면 1/240 등
        }
        
        self.ctrl_mode = "single"  # "single" or "dual"
        
        self.pending = {
            "L_q": self.shared["L_q"].copy(),
            "R_q": self.shared["R_q"].copy(),
            "L_ee": self.shared.get("L_ee", None),
            "R_ee": self.shared.get("R_ee", None),
        }
        
        self.log_left  = "configs/ur5L_motions.jsonl"
        self.log_right = "configs/ur5R_motions.jsonl"
        self._save_counts = {"L": 0, "R": 0}

        self.motionsL = []
        self.motionsR = []
        self.selected_motion = {"L": None, "R": None}

        self.left_gripper_ui = None
        self.left_gripper_range = (0.0, 0.70)

        baseL_pos, baseL_orn = p.getBasePositionAndOrientation(self.sim.urL)
        baseR_pos, baseR_orn = p.getBasePositionAndOrientation(self.sim.urR)
        baseL_eul = list(p.getEulerFromQuaternion(baseL_orn))
        baseR_eul = list(p.getEulerFromQuaternion(baseR_orn))

        self.probeL = ProbeFK(left_urdf_path, baseL_pos, baseL_eul)
        self.probeR = ProbeFK(right_urdf_path, baseR_pos, baseR_eul)

        self._build_ui()

        self.th = SimThread(self.sim, self.lock, self.shared)
        self.th.sig_state.connect(self._on_state)
        self.th.start()

        self._apply_from_ee("L")
        self._apply_from_ee("R")

    def _is_probably_deg(self, angle_values):
        arr = np.asarray(angle_values, dtype=np.float32)
        if arr.size == 0:
            return False
        return np.max(np.abs(arr)) > (2.0 * np.pi + 0.5)

    def _sim_ee_to_ui(self, ee6_rad):
        ee = np.array(ee6_rad, dtype=np.float32).copy()
        ee[3:] = np.degrees(ee[3:])
        return ee

    def _ui_ee_to_sim(self, ee6_ui):
        ee = np.array(ee6_ui, dtype=np.float32).copy()
        ee[3:] = np.radians(ee[3:])
        return ee

    def _sim_q_to_ui(self, q6_rad):
        return np.degrees(np.array(q6_rad, dtype=np.float32))

    def _ui_q_to_sim(self, q6_ui):
        return np.radians(np.array(q6_ui, dtype=np.float32))

    def _motion_ee_to_sim(self, ee6):
        arr = np.array(ee6[:6], dtype=np.float32)
        if self._is_probably_deg(arr[3:]):
            arr[3:] = np.radians(arr[3:])
        return arr

    def _motion_q_to_sim(self, q6):
        arr = np.array(q6[:6], dtype=np.float32)
        if self._is_probably_deg(arr):
            arr = np.radians(arr)
        return arr
    
    def _load_jsonl(self, path):
        items = []
        if not os.path.exists(path):
            return items
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
        return items
    
    def _populate_motion_table(self, arm):
        if arm == "L":
            table = self.tblL
            motions = self.motionsL
        else:
            table = self.tblR
            motions = self.motionsR
    
        table.setRowCount(len(motions))
    
        for r, rec in enumerate(motions):
            ts = rec.get("ts", "")
            ee = rec.get("ee_target", None)  # [x,y,z,r,p,y]
            q  = rec.get("q_target", None)   # [j1..j6]
    
            if ee and len(ee) >= 3:
                xyz = f"{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}"
            else:
                xyz = "-"
    
            if q and len(q) >= 6:
                q_sim = self._motion_q_to_sim(q)
                q_deg = self._sim_q_to_ui(q_sim)
                qtxt = f"{q_deg[0]:+.1f}, {q_deg[1]:+.1f}, {q_deg[2]:+.1f}, {q_deg[3]:+.1f}, {q_deg[4]:+.1f}, {q_deg[5]:+.1f}"
            else:
                qtxt = "-"
    
            def set_item(c, s):
                it = QtWidgets.QTableWidgetItem(str(s))
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                table.setItem(r, c, it)
    
            set_item(0, r)
            set_item(1, ts)
            set_item(2, xyz)
            set_item(3, qtxt)
    
        table.resizeColumnsToContents()

    def _stage_motion_from_table(self, arm, row):
        if arm == "L":
            motions = self.motionsL
        else:
            motions = self.motionsR
    
        if row < 0 or row >= len(motions):
            return
    
        rec = motions[row]
        q = rec.get("q_target", None)
        ee = rec.get("ee_target", None)
    
        if q is not None and len(q) >= 6:
            q6 = self._motion_q_to_sim(q)
    
            ee6 = self.probeL.fk(q6) if arm == "L" else self.probeR.fk(q6)
    
            self._write_ui_q(arm, q6)
            self._write_ui_ee(arm, ee6)
    
            if arm == "L":
                self.pending["L_q"]  = q6.copy()
                self.pending["L_ee"] = ee6.copy()
                self.selected_motion["L"] = rec
                if hasattr(self, "labSelL"):
                    self.labSelL.setText(f"Selected L: #{row}")
            else:
                self.pending["R_q"]  = q6.copy()
                self.pending["R_ee"] = ee6.copy()
                self.selected_motion["R"] = rec
                if hasattr(self, "labSelR"):
                    self.labSelR.setText(f"Selected R: #{row}")
    
        else:
            if ee is None or len(ee) < 6:
                return
            ee6 = self._motion_ee_to_sim(ee)
            self._write_ui_ee(arm, ee6)
            if arm == "L":
                q6 = np.array(self.sim._ik_to_joints(self.sim.urL, self.sim.eeL, self.sim.jL, ee6, self.sim.homeL), dtype=np.float32)
                self._write_ui_q("L", q6)
                self.pending["L_q"], self.pending["L_ee"] = q6.copy(), ee6.copy()
                self.selected_motion["L"] = rec
            else:
                q6 = np.array(self.sim._ik_to_joints(self.sim.urR, self.sim.eeR, self.sim.jR, ee6, self.sim.homeR), dtype=np.float32)
                self._write_ui_q("R", q6)
                self.pending["R_q"], self.pending["R_ee"] = q6.copy(), ee6.copy()
                self.selected_motion["R"] = rec

    
    def _commit_pending_to_shared(self):
        """pending -> shared (시뮬에 반영되는 값 업데이트)"""
        with self.lock:
            self.shared["L_q"] = self.pending["L_q"].copy()
            self.shared["R_q"] = self.pending["R_q"].copy()

    def _commit_if_single(self):
        if self.ctrl_mode == "single":
            self._commit_pending_to_shared()

    
    def _make_slider_spin(self, lo, hi, val, step=0.001, decimals=4):
        """
        float 범위(lo~hi)를 QSlider(int)로 매핑 + QDoubleSpinBox 동기화
        슬라이더 움직이면 즉시 valueChanged가 발생해서 로봇이 바로 움직이게 함.
        """
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(step)
        sp.setValue(val)
    
        sp.setLocale(QtCore.QLocale("C"))
    
        sp.setKeyboardTracking(True)
    
        sp.setFixedWidth(110)
    
        sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    
        n = int(round((hi - lo) / step))
        n = max(n, 1)
        sl.setRange(0, n)
    
        sl.setTracking(True)
    
        def x_to_i(x):
            i = int(round((float(x) - lo) / step))
            return max(0, min(n, i))
    
        def i_to_x(i):
            return float(lo + float(i) * step)
    
        sl.setValue(x_to_i(val))
    
        def on_slider(i):
            x = i_to_x(i)
            if abs(sp.value() - x) > (step * 0.25):
                sp.setValue(x)  # ✅ 여기서 sp.valueChanged가 발생 -> 너의 _on_ee_changed/_on_j_changed가 바로 호출됨
    
        def on_spin(x):
            with QtCore.QSignalBlocker(sl):
                sl.setValue(x_to_i(x))
    
        sl.valueChanged.connect(on_slider)
        sp.valueChanged.connect(on_spin)
        sp.editingFinished.connect(lambda: on_spin(sp.value()))
        
        sp._paired_slider = sl
        sp._x_to_i = x_to_i
        

    
        return sl, sp

    def _set_pair_value(self, sp, x):
        """프로그램이 값을 바꿀 때 spin + slider 위치를 같이 맞춘다."""
        x = float(x)
    
        with QtCore.QSignalBlocker(sp):
            sp.setValue(x)
    
        sl = getattr(sp, "_paired_slider", None)
        x_to_i = getattr(sp, "_x_to_i", None)
        if sl is not None and x_to_i is not None:
            with QtCore.QSignalBlocker(sl):
                sl.setValue(int(x_to_i(x)))

    
    def _row_widget(self, spin, slider):
        """FormLayout 한 줄에 (SpinBox + Slider) 같이 넣기 위한 row 위젯"""
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(spin)
        h.addWidget(slider)
        return w

    def _on_left_gripper_changed(self):
        if self.left_gripper_ui is None:
            return
        opening = float(self.left_gripper_ui.value())
        with self.lock:
            if hasattr(self.sim, "set_left_gripper_opening"):
                self.sim.set_left_gripper_opening(opening)

    def _build_left_gripper_panel(self):
        if getattr(self.sim, "left_gripper_joint", None) is None:
            return None
        lo, hi = self.left_gripper_range
        init = float(getattr(self.sim, "left_gripper_target", hi))
        sl, sp = self._make_slider_spin(lo, hi, init, step=0.005, decimals=3)
        sp.valueChanged.connect(lambda _=None: self._on_left_gripper_changed())
        self.left_gripper_ui = sp

        box = QtWidgets.QGroupBox("L Robotiq140 Gripper")
        form = QtWidgets.QFormLayout(box)
        form.addRow("Opening [rad]", self._row_widget(sp, sl))

        btn_row = QtWidgets.QHBoxLayout()
        btn_open = QtWidgets.QPushButton("Open")
        btn_close = QtWidgets.QPushButton("Close")
        btn_open.clicked.connect(lambda: self._set_pair_value(sp, hi))
        btn_close.clicked.connect(lambda: self._set_pair_value(sp, lo))
        btn_row.addWidget(btn_open)
        btn_row.addWidget(btn_close)
        btn_row.addStretch(1)
        form.addRow(btn_row)
        return box

    def _build_arm_panel(self, arm="L"):
        if arm == "L":
            x_lo, x_hi = 0.15, 0.85
            y_lo, y_hi = -0.80, 0.20
        else:
            x_lo, x_hi = 0.15, 0.85
            y_lo, y_hi = -0.20, 0.80
        z_lo, z_hi = 0.02, 0.60
        a_lo, a_hi = -180.0, 180.0
    
        ee0_rad = self.shared["L_ee"] if arm=="L" else self.shared["R_ee"]
        ee0 = self._sim_ee_to_ui(ee0_rad)
        sx_sl, sx = self._make_slider_spin(x_lo, x_hi, float(ee0[0]), step=0.001, decimals=4)
        sy_sl, sy = self._make_slider_spin(y_lo, y_hi, float(ee0[1]), step=0.001, decimals=4)
        sz_sl, sz = self._make_slider_spin(z_lo, z_hi, float(ee0[2]), step=0.001, decimals=4)
        sR_sl, sR = self._make_slider_spin(a_lo, a_hi, float(ee0[3]), step=0.1, decimals=2)
        sP_sl, sP = self._make_slider_spin(a_lo, a_hi, float(ee0[4]), step=0.1, decimals=2)
        sY_sl, sY = self._make_slider_spin(a_lo, a_hi, float(ee0[5]), step=0.1, decimals=2)
    
        g_ee = QtWidgets.QGroupBox(f"{arm} End-Effector (target)")
        f1 = QtWidgets.QFormLayout(g_ee)
        f1.addRow("X [m]", self._row_widget(sx, sx_sl))
        f1.addRow("Y [m]", self._row_widget(sy, sy_sl))
        f1.addRow("Z [m]", self._row_widget(sz, sz_sl))
        f1.addRow("Roll [deg]",  self._row_widget(sR, sR_sl))
        f1.addRow("Pitch [deg]", self._row_widget(sP, sP_sl))
        f1.addRow("Yaw [deg]",   self._row_widget(sY, sY_sl))
    
        rid_probe = self.probeL.rid if arm=="L" else self.probeR.rid
        jids_probe = self.probeL.jids if arm=="L" else self.probeR.jids
        q0 = self.shared["L_q"] if arm=="L" else self.shared["R_q"]
    
        j_pairs = []
        for k, jid in enumerate(jids_probe):
            lo, hi = get_joint_limit(rid_probe, jid)
            lo_deg, hi_deg = np.degrees(lo), np.degrees(hi)
            q0_ui = self._sim_q_to_ui(q0)
            j_sl, j_sp = self._make_slider_spin(lo_deg, hi_deg, float(q0_ui[k]), step=0.1, decimals=2)
            j_pairs.append((j_sp, j_sl))
    
        g_j = QtWidgets.QGroupBox(f"{arm} Joints (target)")
        f2 = QtWidgets.QFormLayout(g_j)
        for i, (j_sp, j_sl) in enumerate(j_pairs):
            f2.addRow(f"J{i+1} [deg]", self._row_widget(j_sp, j_sl))
    
        lab_ee = QtWidgets.QLabel("EE(actual): -")
        lab_q  = QtWidgets.QLabel("Q(actual): -")
        lab_joint_xyz = QtWidgets.QLabel("Joint XYZ(actual): -")
        lab_pts = QtWidgets.QLabel("far1/far2/near(actual): -")

        for lb in (lab_joint_xyz, lab_pts):
            lb.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            lb.setWordWrap(True)
    
        g_act = QtWidgets.QGroupBox(f"{arm} Readback (actual)")
        vact = QtWidgets.QVBoxLayout(g_act)
        vact.addWidget(lab_ee)
        vact.addWidget(lab_q)
        vact.addWidget(lab_joint_xyz)
        vact.addWidget(lab_pts)
    
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
    
        lay.addWidget(QtWidgets.QLabel(f"{arm} Arm Panel"))
    
        lay.addWidget(g_ee)
        lay.addWidget(g_j)
        lay.addWidget(g_act)
    
        btn_save = QtWidgets.QPushButton(f"{arm} motion save")
        btn_save.clicked.connect(lambda _=None, a=arm: self._save_motion(a))
    
        lab_save = QtWidgets.QLabel("Saved: 0")
        rowS = QtWidgets.QHBoxLayout()
        rowS.addWidget(btn_save)
        rowS.addWidget(lab_save)
        rowS.addStretch(1)
        lay.addLayout(rowS)
    
        lay.addStretch(1)
    
        pack = {
            "ee": [sx, sy, sz, sR, sP, sY],
            "j":  [pair[0] for pair in j_pairs],   # spin만
            "lab_ee": lab_ee,
            "lab_q": lab_q,
            "lab_joint_xyz": lab_joint_xyz,
            "lab_pts": lab_pts,
            "lab_save": lab_save,
        }
    
        for sp in pack["ee"]:
            sp.valueChanged.connect(lambda _=None, a=arm: self._on_ee_changed(a))
    
        for sp in pack["j"]:
            sp.valueChanged.connect(lambda _=None, a=arm: self._on_j_changed(a))
    
        return w, pack

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)

        leftW, self.left = self._build_arm_panel("L")
        rightW, self.right = self._build_arm_panel("R")

        root.addWidget(leftW)
        root.addWidget(rightW)

        g_left_gripper = self._build_left_gripper_panel()
        if g_left_gripper is not None:
            root.addWidget(g_left_gripper)

        bar = QtWidgets.QVBoxLayout()
        btn_stop = QtWidgets.QPushButton("Stop Simulation Thread")
        btn_stop.clicked.connect(self._stop_thread)
        bar.addWidget(btn_stop)

        btn_force_far = QtWidgets.QPushButton("Set far1/far2 from sack")
        btn_force_far.clicked.connect(self._force_far_from_sack)
        bar.addWidget(btn_force_far)
        root.addLayout(bar)

        self.labSack = QtWidgets.QLabel("SACK(actual): -")
        self.labSack.setWordWrap(True)
        self.labSack.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        bar.addWidget(self.labSack)
        
        self.btnLoadL = QtWidgets.QPushButton("Load JSONL (L)")
        self.btnLoadR = QtWidgets.QPushButton("Load JSONL (R)")
        
        self.btnLoadL.clicked.connect(lambda: self._load_jsonl_dialog("L"))
        self.btnLoadR.clicked.connect(lambda: self._load_jsonl_dialog("R"))
        
        self.labFileL = QtWidgets.QLabel("L file: (default)")
        self.labFileR = QtWidgets.QLabel("R file: (default)")
        
        bar.addSpacing(12)
        bar.addWidget(self.btnLoadL)
        bar.addWidget(self.labFileL)
        bar.addSpacing(6)
        bar.addWidget(self.btnLoadR)
        bar.addWidget(self.labFileR)
        
        bar.addStretch(1)
        
        
        
        
        self.cmb_ctrl = QtWidgets.QComboBox()
        self.cmb_ctrl.addItems(["Single control (live)", "Dual control (staged)"])
        self.cmb_ctrl.setCurrentIndex(0)
        
        self.btn_exec = QtWidgets.QPushButton("EXECUTE (move both robots)")
        self.btn_exec.setEnabled(False)
        
        self.lab_mode = QtWidgets.QLabel("Mode: Single (live)")
        
        def on_ctrl_changed(idx):
            self.ctrl_mode = "single" if idx == 0 else "dual"
            if self.ctrl_mode == "single":
                self.lab_mode.setText("Mode: Single (live)")
                self.btn_exec.setEnabled(False)
                self._commit_pending_to_shared()
            else:
                self.lab_mode.setText("Mode: Dual (staged) - press EXECUTE")
                self.btn_exec.setEnabled(True)
        
        self.cmb_ctrl.currentIndexChanged.connect(on_ctrl_changed)
        
        def on_execute():
            self._commit_pending_to_shared()
        
        self.btn_exec.clicked.connect(on_execute)
        
        bar.addWidget(self.lab_mode)
        bar.addWidget(self.cmb_ctrl)
        bar.addWidget(self.btn_exec)
        
        
        
        gM = QtWidgets.QGroupBox("Saved Motions (double-click to stage)")
        mLay = QtWidgets.QVBoxLayout(gM)
        
        def make_table():
            tbl = QtWidgets.QTableWidget()
            tbl.setColumnCount(4)
            tbl.setHorizontalHeaderLabels(["idx", "ts", "EE xyz", "q(6)"])
            tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            tbl.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            tbl.verticalHeader().setVisible(False)
            tbl.setMinimumWidth(520)
            return tbl
        
        boxL = QtWidgets.QGroupBox("UR5 Left motions")
        vL = QtWidgets.QVBoxLayout(boxL)
        self.tblL = make_table()
        self.tblL.cellDoubleClicked.connect(lambda r, c: self._stage_motion_from_table("L", r))
        vL.addWidget(self.tblL)
        
        boxR = QtWidgets.QGroupBox("UR5 Right motions")
        vR = QtWidgets.QVBoxLayout(boxR)
        self.tblR = make_table()
        self.tblR.cellDoubleClicked.connect(lambda r, c: self._stage_motion_from_table("R", r))
        vR.addWidget(self.tblR)
        
        mLay.addWidget(boxL)
        mLay.addWidget(boxR)
        
        root.addWidget(gM)
        
        if os.path.exists(self.log_left):
            self._load_jsonl_for_arm("L", self.log_left)
        if os.path.exists(self.log_right):
            self._load_jsonl_for_arm("R", self.log_right)

    
    def _load_jsonl_dialog(self, arm):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"Select {arm} motion JSONL",
            os.getcwd(),
            "JSONL files (*.jsonl);;All files (*.*)"
        )
        if not path:
            return
        self._load_jsonl_for_arm(arm, path)
    
    def _load_jsonl_for_arm(self, arm, path):
        path = os.path.abspath(path)
        motions = self._load_jsonl(path)
    
        if arm == "L":
            self.log_left = path
            self.motionsL = motions
            if hasattr(self, "tblL"):
                self._populate_motion_table("L")
            if hasattr(self, "labFileL"):
                self.labFileL.setText(f"L file: {os.path.basename(path)}")
        else:
            self.log_right = path
            self.motionsR = motions
            if hasattr(self, "tblR"):
                self._populate_motion_table("R")
            if hasattr(self, "labFileR"):
                self.labFileR.setText(f"R file: {os.path.basename(path)}")
    
    
            

    def _read_ui_ee(self, arm):
        pack = self.left if arm=="L" else self.right
        ee_ui = np.array([sp.value() for sp in pack["ee"]], dtype=np.float32)
        return self._ui_ee_to_sim(ee_ui)

    def _read_ui_q(self, arm):
        pack = self.left if arm=="L" else self.right
        q_ui = np.array([sp.value() for sp in pack["j"]], dtype=np.float32)
        return self._ui_q_to_sim(q_ui)

    def _write_ui_q(self, arm, q6):
        pack = self.left if arm=="L" else self.right
        q_ui = self._sim_q_to_ui(q6)
        for sp, v in zip(pack["j"], q_ui):
            self._set_pair_value(sp, float(v))
    
    def _write_ui_ee(self, arm, ee6):
        pack = self.left if arm=="L" else self.right
        ee_ui = self._sim_ee_to_ui(ee6)
        for sp, v in zip(pack["ee"], ee_ui):
            self._set_pair_value(sp, float(v))

    def _apply_from_ee(self, arm):
        ee6 = self._read_ui_ee(arm)
        with self.lock:  # ✅ IK가 bullet 호출
            if arm == "L":
                q = self.sim._ik_to_joints(self.sim.urL, self.sim.eeL, self.sim.jL, ee6, self.sim.homeL)
                self.pending["L_q"] = np.array(q, dtype=np.float32)
                self.pending["L_ee"] = ee6.copy()
            else:
                q = self.sim._ik_to_joints(self.sim.urR, self.sim.eeR, self.sim.jR, ee6, self.sim.homeR)
                self.pending["R_q"] = np.array(q, dtype=np.float32)
                self.pending["R_ee"] = ee6.copy()
    
        self._write_ui_q(arm, q)
    
        self._commit_if_single()

    def _apply_from_joint(self, arm):
        q6 = self._read_ui_q(arm)
    
        ee6 = self.probeL.fk(q6) if arm == "L" else self.probeR.fk(q6)
    
        self._write_ui_ee(arm, ee6)
    
        if arm == "L":
            self.pending["L_q"] = q6.copy()
            self.pending["L_ee"] = ee6.copy()
        else:
            self.pending["R_q"] = q6.copy()
            self.pending["R_ee"] = ee6.copy()
    
        self._commit_if_single()

    def _on_mode(self, arm):
        pack = self.left if arm=="L" else self.right
        self._set_shared_mode(arm, pack["mode"].currentIndex())

    def _on_ee_changed(self, arm):
        self._apply_from_ee(arm)

    def _on_j_changed(self, arm):
        self._apply_from_joint(arm)

    def _fmt_joint_xyz(self, joint_xyz):
        if not joint_xyz:
            return "Joint XYZ(actual): -"
        parts = [f"J{i+1}=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})" for i, p in enumerate(joint_xyz)]
        return "Joint XYZ(actual): " + " | ".join(parts)

    def _fmt_pts(self, dbg):
        if not dbg:
            return "far1/far2/near(actual): -"
        f1 = dbg.get("far1", [0,0,0]); f2 = dbg.get("far2", [0,0,0]); nr = dbg.get("near", [0,0,0])
        ang = dbg.get("yz_angle_deg", None)
        ang_txt = "-" if ang is None else f"{ang:.2f}deg"
        return (
            f"far1=({f1[0]:+.3f},{f1[1]:+.3f},{f1[2]:+.3f}) "
            f"far2=({f2[0]:+.3f},{f2[1]:+.3f},{f2[2]:+.3f}) "
            f"near=({nr[0]:+.3f},{nr[1]:+.3f},{nr[2]:+.3f}) yz={ang_txt}"
        )

    def _fmt_sack(self, sack):
        if not sack:
            return "SACK(actual): -"
        c = sack.get("center", [0,0,0])
        r = sack.get("rpy_deg", [0,0,0])
        sz = sack.get("size", [0,0,0])
        return (
            f"SACK(actual): xyz=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) "
            f"rpy_deg=({r[0]:+.1f},{r[1]:+.1f},{r[2]:+.1f}) "
            f"LWH=({sz[0]:.3f},{sz[1]:.3f},{sz[2]:.3f})"
        )

    def _on_state(self, st: dict):
        eeL = st["eeL"]; eeR = st["eeR"]
        qL = st["qL"]; qR = st["qR"]

        eeL_ui = self._sim_ee_to_ui(eeL)
        eeR_ui = self._sim_ee_to_ui(eeR)
        qL_ui = self._sim_q_to_ui(qL)
        qR_ui = self._sim_q_to_ui(qR)

        self.left["lab_ee"].setText(f"EE(actual): x={eeL_ui[0]:.3f}, y={eeL_ui[1]:.3f}, z={eeL_ui[2]:.3f}, rpy_deg=({eeL_ui[3]:+.1f},{eeL_ui[4]:+.1f},{eeL_ui[5]:+.1f})")
        self.left["lab_q"].setText("Q(actual,deg): " + " ".join([f"{x:+.1f}" for x in qL_ui]))

        self.right["lab_ee"].setText(f"EE(actual): x={eeR_ui[0]:.3f}, y={eeR_ui[1]:.3f}, z={eeR_ui[2]:.3f}, rpy_deg=({eeR_ui[3]:+.1f},{eeR_ui[4]:+.1f},{eeR_ui[5]:+.1f})")
        self.right["lab_q"].setText("Q(actual,deg): " + " ".join([f"{x:+.1f}" for x in qR_ui]))

        armL_dbg = st.get("armL_dbg", None)
        armR_dbg = st.get("armR_dbg", None)
        self.left["lab_joint_xyz"].setText(self._fmt_joint_xyz(None if armL_dbg is None else armL_dbg.get("joint_xyz")))
        self.right["lab_joint_xyz"].setText(self._fmt_joint_xyz(None if armR_dbg is None else armR_dbg.get("joint_xyz")))
        self.left["lab_pts"].setText(self._fmt_pts(armL_dbg))
        self.right["lab_pts"].setText(self._fmt_pts(armR_dbg))

        if hasattr(self, "labSack"):
            self.labSack.setText(self._fmt_sack(st.get("sack_dbg", None)))

    def _ensure_dir(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    def _append_jsonl(self, path, rec: dict):
        self._ensure_dir(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    def _save_motion(self, arm: str):
        ee_t = self._sim_ee_to_ui(self._read_ui_ee(arm)).tolist()   # [x,y,z,r,p,y] in deg
        q_t  = self._sim_q_to_ui(self._read_ui_q(arm)).tolist()      # [j1..j6] in deg
    
        if arm == "L":
            q_a = self._sim_q_to_ui([float(p.getJointState(self.sim.urL, jid)[0]) for jid in self.sim.jL]).tolist()
            (pos, rpy) = self.sim._get_ee_pose(self.sim.urL, self.sim.eeL)
            ee_a = self._sim_ee_to_ui(np.array(pos + rpy, dtype=np.float32)).tolist()
            out_path = self.log_left
            lab = self.left.get("lab_save", None)
        else:
            q_a = self._sim_q_to_ui([float(p.getJointState(self.sim.urR, jid)[0]) for jid in self.sim.jR]).tolist()
            (pos, rpy) = self.sim._get_ee_pose(self.sim.urR, self.sim.eeR)
            ee_a = self._sim_ee_to_ui(np.array(pos + rpy, dtype=np.float32)).tolist()
            out_path = self.log_right
            lab = self.right.get("lab_save", None)
    
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "t_unix": time.time(),
            "arm": arm,
            "ee_target": ee_t,
            "q_target": q_t,
            "ee_actual": ee_a,
            "q_actual": q_a,
        }
    
        self._append_jsonl(out_path, rec)
    
        self._save_counts[arm] += 1
        if lab is not None:
            print(f"Saved: {self._save_counts[arm]}  -> {out_path}")
    

    def _force_far_from_sack(self):
        """
        버튼 클릭 동작:
        - sim에 sack 기반 far override를 설정
        - RobotL은 즉시 move하지 않고(UI/pending만 갱신)
        """
        with self.lock:
            if not hasattr(self.sim, "set_forced_far_from_sack"):
                return
            target = self.sim.set_forced_far_from_sack(True)
            if target is None:
                return

            pos, rpy = self.sim._get_ee_pose(self.sim.urL, self.sim.eeL)
            ee6 = np.array(pos + rpy, dtype=np.float32)
            ee6[:3] = np.array(target, dtype=np.float32)

            q6 = np.array(
                self.sim._ik_to_joints(self.sim.urL, self.sim.eeL, self.sim.jL, ee6, self.sim.homeL),
                dtype=np.float32,
            )

            self.pending["L_q"] = q6.copy()
            self.pending["L_ee"] = ee6.copy()

        self._write_ui_q("L", q6)
        self._write_ui_ee("L", ee6)

    def _stop_thread(self):
        if self.th.isRunning():
            self.th.stop()
            self.th.wait(1000)

    def closeEvent(self, ev):
        self._stop_thread()
        try:
            if p.isConnected():
                p.disconnect()
        except:
            pass
        super().closeEvent(ev)

def main():
    URDF_L = r'D:\Michael\2025\01.Research\01.Parceldetection\16.Pybullet\tutorial\urdf\ur_description\urdf\ur5_2F\urdf/'
    URDF_R = r'D:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/tutorial/urdf/ur_description/urdf/'
    LOG_PATH = "configs/teleop_log_pyqt.jsonl"
    LEFT_URDF = URDF_L+"ur5_robotiq_140.urdf"
    RIGHT_URDF = URDF_R+"ur5_with_tool0_scoop4.urdf"
    # LEFT_URDF = URDF+"ur5_with_tool0_scoop4.urdf"

    sim = DualUR5EEGuiIK(
        gui=True,
        urdf_path_left=LEFT_URDF,
        urdf_path_right=RIGHT_URDF,
        logpath=LOG_PATH,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = Pendant(sim, LEFT_URDF, RIGHT_URDF)
    w.resize(1300, 760)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
