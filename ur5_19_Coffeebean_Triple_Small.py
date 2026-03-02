# dual_ur5_ee_gui_ik.py
# -*- coding: utf-8 -*-
import time
import numpy as np
import os, json, time
import math

import pybullet as p
import pybullet_data
# from ur5_10_jointmodified import DualUR5EEGuiIK

#%%

"""
ts:로그 확인
t_unix: UNIX epoch time (초)/속도/가속도 추정 시 사용 가능
Left_X, Left_Y, Left_Z, Right_X, Right_Y, Right_Z: Left / Right EE 목표값
Left_Roll, Left_Pitch, Left_Yaw, Right_Roll, Right_Pitch, Right_Yaw: 엔드이펙터의 롤/피치/요 [rad]
qL, qR: 실제 로봇 관절 각도/강화학습에서 가장 좋은 “행동” 형태
eeL, eeR: 실제 도달한 EE 포즈 (결과)

20260109
충돌 제한 기구학 기반 캘리브레이션으로 확인
"""



#%%
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

class TeleopLogger:
    def __init__(self, log_path="configs/teleop_log.jsonl"):
        self.log_path = log_path
        ensure_dir(self.log_path)
        self.prev_save = 0

    def append_log(self, payload: dict):
        ensure_dir(self.log_path)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class DualUR5EEGuiIK:
    def __init__(self,
                 gui=True,
                 urdf_path_left="ur5/ur5.urdf",
                 urdf_path_right="ur5/ur5.urdf",
                 logpath='./configs/teleop_log.jsonl',
                 Clothobj="./cloth_z_up.obj",
                 base_pos_left=(0.0, -0.3, 0.0),
                 base_pos_right=(0.0,  0.3, 0.0),
                 base_ori_left=(0, 0, 0),
                 base_ori_right=(0, 0, 0),
                 time_step=1/240.0):

        self.gui = gui
        self.dt = time_step
        
        timei1=time.time()

        p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)

        p.loadURDF("plane.urdf")

        # 시뮬 안정성 (너의 deformable 세팅과 같이 써도 됨)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt,
            numSubSteps=2, # 4 -> 1  (체감 제일 큼)#ur5_16
            numSolverIterations=80, # 150 -> 80#ur5_16
            deterministicOverlappingPairs=1
        )

        # Robots
        self.urL = p.loadURDF(
            urdf_path_left,
            basePosition=base_pos_left,
            baseOrientation=p.getQuaternionFromEuler(base_ori_left),
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        )
        self.urR = p.loadURDF(
            urdf_path_right,
            basePosition=base_pos_right,
            baseOrientation=p.getQuaternionFromEuler(base_ori_right),
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        )

        # arm joint ids (6DOF)
        self.jL = self._get_arm_revolute_joints(self.urL, 6)
        self.jR = self._get_arm_revolute_joints(self.urR, 6)

        # EE link index 찾기 (pybullet_data UR5는 ee_link가 있는 경우가 많음)
        self.eeL = self._find_link(self.urL, ["ee_link", "tool0", "wrist_3_link"])
        self.eeR = self._find_link(self.urR, ["ee_link", "tool0", "wrist_3_link"])


        # --- after loading robots and finding eeL/eeR ---
        
        plateL = self._find_link(self.urL, ["plate_link"])
        plateR = self._find_link(self.urR, ["plate_link"])
        
        # ✅ 디버그: 실제로 plate_link를 찾았는지 확인
        nameL = p.getJointInfo(self.urL, plateL)[12].decode("utf-8")
        nameR = p.getJointInfo(self.urR, plateR)[12].decode("utf-8")
        print("[DEBUG] plateL idx:", plateL, "name:", nameL)
        print("[DEBUG] plateR idx:", plateR, "name:", nameR)
        
        if nameL != "plate_link":
            print("[WARN] Left plate_link not found! check URDF link name.")
        if nameR != "plate_link":
            print("[WARN] Right plate_link not found! check URDF link name.")
        
        # ✅ 마찰 세팅
        for rid, link in [(self.urL, plateL), (self.urR, plateR)]:
            try:
                p.changeDynamics(rid, link,
                    lateralFriction=2.0,      # 1.5~4.0 테스트
                    spinningFriction=0.02,
                    rollingFriction=0.0,
                    restitution=0.0,
                    frictionAnchor=1
                )
            except TypeError:
                # 어떤 빌드에서는 frictionAnchor 인자 없을 수 있음
                p.changeDynamics(rid, link,
                    lateralFriction=2.0,
                    spinningFriction=0.02,
                    rollingFriction=0.0,
                    restitution=0.0
                )



        # 토크/힘 제한 (진동 심하면 더 낮춰도 됨)
        self.maxF_L = [120, 120, 120, 20, 20, 20]
        self.maxF_R = [120, 120, 120, 20, 20, 20]
        
        # self.maxF_L = [520, 520, 520, 520, 520, 520]
        # self.maxF_R = [520, 520, 520, 520, 520, 520]
        
        # spec limit (monitoring 기준)
        self.torque_limit_L = [150,150,150,28,28,28]
        self.torque_limit_R = [150,150,150,28,28,28]

        # ---- 진동 억제 핵심 파라미터들 ----
        # 1) 모터 gain 낮추기
        self.kp = 0.35        # positionGain (0.05~0.25 사이 추천)
        self.kd = 0.8         # velocityGain 느낌 (PyBullet setJointMotorControl2에서 velocityGain으로 사용)
        self.maxVel = 2.5     # 관절 최대 속도 제한(너무 크면 튐)

        # 2) 관절 댐핑(특히 IK 안정화에 도움)
        self.joint_damping = [0.08]*6  # 0.02~0.2 추천

        # 3) EE 목표값 필터링(EMA) : 0.0이면 매우 느림, 1.0이면 필터 없음
        self.alpha = 0.15     # 0.1~0.3 추천 (작을수록 덜 튐)

        # 홈 포즈(초기)
        self.homeL = [-1.57, -1.54,  1.34, -1.37, -1.57, 0.001]
        self.homeR = [ 1.57, -1.54,  1.34, -1.37, -1.57, 0.001]
        self._reset_arm(self.urL, self.jL, self.homeL)
        self._reset_arm(self.urR, self.jR, self.homeR)

        # EE 현재 위치를 슬라이더 초기값으로 사용
        self.targetL = self._get_ee_pose(self.urL, self.eeL)  # (pos[3], rpy[3])
        self.targetR = self._get_ee_pose(self.urR, self.eeR)

        # 내부 필터 상태
        self.filtL = np.array(self.targetL[0] + self.targetL[1], dtype=np.float32)
        self.filtR = np.array(self.targetR[0] + self.targetR[1], dtype=np.float32)
        
        #prev 상태(변화 감지용) 초기화 prev slider states (for change detection)
        self.prev_L_ee = np.array(self.filtL, dtype=np.float32)
        self.prev_R_ee = np.array(self.filtR, dtype=np.float32)
        self.prev_L_q  = np.array([p.getJointState(self.urL, jid)[0] for jid in self.jL], dtype=np.float32)
        self.prev_R_q  = np.array([p.getJointState(self.urR, jid)[0] for jid in self.jR], dtype=np.float32)
        
        # joint smoothing(선택): 조인트 슬라이더도 부드럽게
        self.alpha_q = 0.25
        self.filtqL = self.prev_L_q.copy()
        self.filtqR = self.prev_R_q.copy()
        
        # change detect threshold
        self.ctrl_thresh = 3e-3

        timei2=time.time()
        # ---- 파지물체 ----
        self.scale = 0.15
        self.mass_each = 0.20  # sheet mass (tune)
        
        timei3=time.time()

        shperepath="sphere_small.urdf"
        cubepath = "cube_small.urdf"
        coffeebeanpath = "D:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master (1)/bullet3-master/examples/pybullet/examples/DeformableTest/object/coffeebean/coffee_bean.obj"
        coffeebeanmeshpath = "D:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master (1)/bullet3-master/examples/pybullet/examples/DeformableTest/object/coffeebean/coffee_bean_mesh.obj"
        # ---- sack softbody (Blender obj) ----
        
        SACK_OBJ = r"D:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master (1)/bullet3-master/examples/pybullet/examples/DeformableTest/object/sack.obj"
        # SACK_OBJ = r"D:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master (1)/bullet3-master/examples/pybullet/examples/DeformableTest/object/sack3.obj"
        # SACK_OBJ = r"D:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master (1)/bullet3-master/examples/pybullet/examples/DeformableTest/object/sack5.obj"

        # ✅ OBJ 경로를 pybullet이 못 찾는 경우가 많으니, 폴더를 search path에 추가
        p.setAdditionalSearchPath(os.path.dirname(SACK_OBJ))
        
        # (권장) softbody 파라미터: 처음은 안정적으로
        sack_scale = 0.05       # Blender에서 만든 스케일에 따라 0.1~5.0 사이로 조절
        sack_mass  = 0.1        # 겉천 총 질량 (원하는 느낌에 맞게 조절)
        # BaseLoaction=[0.50, 0.35, 0.20]
        BaseLoaction=[0.50, 0.17, 0.20]#scale 0.05
      
        self.sack_id = p.loadSoftBody(
            fileName=SACK_OBJ,
            basePosition=BaseLoaction,     # 원하는 위치로
            baseOrientation=p.getQuaternionFromEuler([0, 1.57, 0]),
            scale=sack_scale,
            mass=sack_mass,
            useNeoHookean=0,                 # 0=mass-spring(주름/천 느낌 좋음)
            useMassSpring=1,
            useBendingSprings=1, #접힘/굽힘 저항 on off
            springElasticStiffness=50,      # 50~400 튜닝/ 낮을수록 잘 늘어남
            springDampingStiffness=0.35,     # 0.1~0.5 튜닝
            springDampingAllDirections=1, #진동 억제 on off
            frictionCoeff=0.8,#마찰력
            useSelfCollision=1,
            useFaceContact=1, 
            collisionMargin=0.0005            # ✅ 입자 빠져나감 줄이려면 0.003~0.008
        )
        
        # # 시각화 (선택)
        # p.changeVisualShape(self.sack_id, -1, rgbaColor=[0.9, 0.85, 0.7, 0.8])
        p.changeVisualShape(self.sack_id, -1, rgbaColor=[0.9, 0.85, 0.7, 0.99])
        p.changeVisualShape(self.sack_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
        
        
        # verts = self.get_soft_verts(self.sack_id)[0]  # (N,3) float32
        # print(f'verts:{verts}')


        s=2
        mass0=0.003#kg단위
        bean_ids = self.spawn_clump_grid(ClumpType=3,
            r=0.003*s, vis_scale=0.003*s,mass=mass0*(s**3),
            # xs=(0.44,0.46,0.48,0.50,0.52,0.54,0.56),
            # ys=(-0.1,-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.1),
            # zs=(0.16,0.18,0.2,0.22,0.24),
            
            # xs=(0.47,0.48,0.49,0.50,0.51,0.52,0.53), #7
            # ys=(-0.05,-0.04,-0.03,-0.02,-0.01,0,0.01,0.02,0.03,0.04,0.05), #11
            # zs=(0.2-0.007*2,0.2-0.007,0.2,0.2+0.007), #4
            
            xs=(0.48,0.50,0.52), #7
            ys=(-0.02,0,0.02), #11
            zs=(0.185,0.2,0.215), #4
            )

        

        timei4=time.time()
        # --------- Pybullet GUI ---------
        # UI 슬라이더 생성
        self.ui = {}
        # self._create_ee_sliders()
        #EE 슬라이더 만든 뒤에 조인트 슬라이더도 만들기
        # self._create_joint_sliders()
        
        self._prev_btn = {
            "L_COPY_EE2J": 0, "L_COPY_J2EE": 0,
            "R_COPY_EE2J": 0, "R_COPY_J2EE": 0
        }
        
        # readback text id
        self._readback_id_L = None
        self._readback_id_R = None

        
        # logger
        self.logger = TeleopLogger(log_path=logpath)

        # SAVE 버튼(토글)      
        # self.btn_save = p.addUserDebugParameter("SAVE (append)", 0, 1, 0)
        # self.btn_save_reset = p.addUserDebugParameter("SAVE_RESET (set SAVE=0)", 0, 1, 0)
        self._prev_save = 0
        self._prev_reset = 0
        
        # 토크 텍스트 표시용 debug item id 저장
        self.torque_text_ids_L = [None]*len(self.jL)
        self.torque_text_ids_R = [None]*len(self.jR)

        # 스펙 토크 리밋(모니터링 기준)
        self.torque_limit_L = [150,150,150,28,28,28]
        self.torque_limit_R = [150,150,150,28,28,28]

        # print 과다 방지 (초당 너무 많이 찍히면 또 느려짐)
        self._torque_last_print_t = 0.0
        self._torque_print_interval = 0.25  # 0.25초에 1번만 출력
        self._torque_ratio_th = 1.0         # 1.0이면 리밋 초과만, 0.8이면 80% 경고
        
        self.sat_count_L = 0
        self.sat_count_R = 0
        self.sat_warn_frames = 30   # 240Hz 기준 0.125초 (원하면 60=0.25초)
        
        timei5=time.time()
        self.cmd = {
            "L_mode": 0,  # 0=EE,1=J
            "R_mode": 0,
            "L_ee": np.array(self.filtL, dtype=np.float32),  # 6
            "R_ee": np.array(self.filtR, dtype=np.float32),
            "L_q":  np.array(self.homeL, dtype=np.float32),  # 6
            "R_q":  np.array(self.homeR, dtype=np.float32),
            "sleep": self.dt,
        }
        time1=timei2-timei1;print(f'time1:{time1}')
        time2=timei3-timei2;print(f'time2:{time2}')
        time3=timei4-timei3;print(f'time3:{time3}')
        time4=timei5-timei4;print(f'time4:{time4}')

        
    def _save_if_pressed(self, qL, qR):
        save_val = int(p.readUserDebugParameter(self.btn_save))
    
        # SAVE rising edge에서만 저장
        if save_val == 1 and self._prev_save == 0:
            L_raw, R_raw, _, _ = self._read_targets()
    
            rec = {
                "ts": now_ts(),
                "t_unix": time.time(),
    
                "Left_X": float(L_raw[0]), "Left_Y": float(L_raw[1]), "Left_Z": float(L_raw[2]),
                "Left_Roll": float(L_raw[3]), "Left_Pitch": float(L_raw[4]), "Left_Yaw": float(L_raw[5]),
    
                "Right_X": float(R_raw[0]), "Right_Y": float(R_raw[1]), "Right_Z": float(R_raw[2]),
                "Right_Roll": float(R_raw[3]), "Right_Pitch": float(R_raw[4]), "Right_Yaw": float(R_raw[5]),
    
                "qL": [float(x) for x in qL],
                "qR": [float(x) for x in qR],
    
                "eeL": self._get_ee_pose(self.urL, self.eeL),
                "eeR": self._get_ee_pose(self.urR, self.eeR),
            }
    
            self.logger.append_log(rec)
            print(f"[SAVED] -> {self.logger.log_path}")
            p.addUserDebugText("Saved!", [0,0,1], lifeTime=0.8)
    
        self._prev_save = save_val
    
        # ---- 수동 리셋 버튼 처리 ----
        reset_val = int(p.readUserDebugParameter(self.btn_save_reset))
        if reset_val == 1 and self._prev_reset == 0:
            print("[RESET] Please manually set SAVE slider back to 0.")
            p.addUserDebugText("Set SAVE back to 0", [0,0,1], lifeTime=0.8)
        self._prev_reset = reset_val

    
############################20260211
    def create_peanut_collision(self,r=0.003, d=0.0035):
        # d: 두 구 중심 간 거리(겹치게 하려면 d < 2r)
        col = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_SPHERE, p.GEOM_SPHERE],
            radii=[r, r],
            collisionFramePositions=[[0, 0, -d/2], [0, 0, +d/2]],
        )
        return col

    def create_bean_visual(self,mesh_path, s=0.003):
        vis = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=[s, s, s],
        )
        return vis
    
    
    def create_clump_collision(self, r, offsets):
        """
        offsets: [[x,y,z], ...] 각 구의 중심 오프셋 (m)
        r: 각 구 반지름 (m)
        """
        n = len(offsets)
        col = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_SPHERE]*n,
            radii=[r]*n,
            collisionFramePositions=offsets,
        )
        return col

    def create_clump_visual(self, r, offsets, rgba=(0.4,0.3,0.2,1.0)):
        n = len(offsets)
        vis = p.createVisualShapeArray(
            shapeTypes=[p.GEOM_SPHERE]*n,
            radii=[r]*n,
            visualFramePositions=offsets,
            rgbaColors=[rgba]*n
        )
        return vis

    
    def offsets_tri(self, d):
        # 정삼각형 배치 (z=0 평면)
        return [
            [0.0,     0.0,   0.0],
            [d,       0.0,   0.0],
            [0.5*d, 0.866*d, 0.0],
        ]
    
    def offsets_tetra(self, d):
        # 정사면체 배치 (대략)
        return [
            [0.0,     0.0,    0.0],
            [d,       0.0,    0.0],
            [0.5*d, 0.866*d,  0.0],
            [0.5*d, 0.2887*d, 0.816*d],
        ]
    
    def spawn_clump_grid(self, xs, ys, zs,
                     r=0.003, d=None, ClumpType=3,
                     mass=0.003, dyn=None,
                     use_mesh_visual=False, mesh_path=None, vis_scale=0.003):

        if d is None:
            d = 1.5 * r  # 시작값
    
        if ClumpType == 3:
            offsets = self.offsets_tri(d)
        elif ClumpType == 4:
            offsets = self.offsets_tetra(d)
        else:
            raise ValueError("n must be 3 or 4")
    
        col = self.create_clump_collision(r, offsets)
    
        if use_mesh_visual and mesh_path is not None:
            vis = self.create_bean_visual(mesh_path, vis_scale)  # 네 기존 mesh visual
        else:
            vis = self.create_clump_visual(r, offsets)
    
        if dyn is None:
            dyn = dict(
                lateralFriction=3.0,
                spinningFriction=0.05,
                rollingFriction=0.08,     # ✅ 0.02 -> 0.05~0.15 올려봐 (쌓임에 도움)
                restitution=0.0,
                linearDamping=0.9,
                angularDamping=0.9,
                # 가능하면 sleep도 켜기(버전 지원 시)
                # linearSleepingThreshold=0.02,
                # angularSleepingThreshold=0.02,
            )
    
        ids=[]
        for x in xs:
            for y in ys:
                for z in zs:
                    orn = p.getQuaternionFromEuler(np.random.uniform(-np.pi, np.pi, 3))
                    bid = p.createMultiBody(
                        baseMass=mass,
                        baseCollisionShapeIndex=col,
                        baseVisualShapeIndex=vis,
                        basePosition=[x,y,z],
                        baseOrientation=orn
                    )
                    p.changeDynamics(bid, -1, **dyn)
                    ids.append(bid)
        return ids

    
    def spawn_peanut_grid(self,mesh_path, xs, ys, zs,
                          r=0.003, d=0.0035, vis_scale=0.003,
                          mass=0.003, dyn=None):
        col = self.create_peanut_collision(r, d)
        vis = self.create_bean_visual(mesh_path, vis_scale)
    
        if dyn is None:
            # dyn = dict(
            #     lateralFriction=1.5,
            #     spinningFriction=0.05,
            #     rollingFriction=0.02,     # ✅ 핵심: 구슬-유체화 방지
            #     restitution=0.0,
            #     linearDamping=0.6,
            #     angularDamping=0.6,
            # )
            
            dyn = dict(
                lateralFriction=3.0,
                spinningFriction=0.02,
                rollingFriction=0.02,     # ✅ 핵심: 구슬-유체화 방지
                restitution=0.0,
                linearDamping=0.9,
                angularDamping=0.9,
            )
    
        ids=[]
        for x in xs:
            for y in ys:
                for z in zs:
                    orn = p.getQuaternionFromEuler(np.random.uniform(-np.pi, np.pi, 3))
                    bid = p.createMultiBody(
                        baseMass=mass,
                        baseCollisionShapeIndex=col,
                        baseVisualShapeIndex=vis,
                        basePosition=[x,y,z],
                        baseOrientation=orn
                    )
                    p.changeDynamics(bid, -1, **dyn)
                    ids.append(bid)
        return ids

############################20260211




##############################20260302 sack 위위쪽


    def _get_arm_revolute_joints(self, rid, target_dofs=6):
        joints = []
        for j in range(p.getNumJoints(rid)):
            if p.getJointInfo(rid, j)[2] == p.JOINT_REVOLUTE:
                joints.append(j)
        return joints[:target_dofs]

    def _find_link(self, rid, candidates):
        name_to_idx = {}
        for j in range(p.getNumJoints(rid)):
            link_name = p.getJointInfo(rid, j)[12].decode("utf-8")
            name_to_idx[link_name] = j
        for c in candidates:
            if c in name_to_idx:
                return name_to_idx[c]
        return p.getNumJoints(rid) - 1

    def _reset_arm(self, rid, joint_ids, q):
        for jid, qi in zip(joint_ids, q):
            p.resetJointState(rid, jid, qi)
        for jid in joint_ids:
            p.setJointMotorControl2(rid, jid, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    def _get_ee_pose(self, rid, ee_idx):
        ls = p.getLinkState(rid, ee_idx)
        pos = list(ls[4])  # worldLinkFramePosition
        orn = ls[5]
        rpy = list(p.getEulerFromQuaternion(orn))
        return pos, rpy

    def contact_force_sum(self, bodyA, bodyB=None):
        pts = p.getContactPoints(bodyA=bodyA, bodyB=bodyB) if bodyB is not None else p.getContactPoints(bodyA=bodyA)
        return sum(cp[9] for cp in pts)  # normal force


    def _create_ee_sliders(self):
        # 범위는 네 작업공간에 맞게 조절해. (대충 UR이 닿을만한 범위로)
        (pL, rL) = self.targetL
        (pR, rR) = self.targetR

        # Left
        self.ui["Left_X"] = p.addUserDebugParameter("Left_X", 0.15, 0.85, pL[0])
        self.ui["Left_Y"] = p.addUserDebugParameter("Left_Y", -0.80, 0.20, pL[1])
        self.ui["Left_Z"] = p.addUserDebugParameter("Left_Z", 0.02, 0.60, pL[2])
        self.ui["Left_Roll"]  = p.addUserDebugParameter("Left_Roll",  -3.14, 3.14, rL[0])
        self.ui["Left_Pitch"] = p.addUserDebugParameter("Left_Pitch", -3.14, 3.14, rL[1])
        self.ui["Left_Yaw"]   = p.addUserDebugParameter("Left_Yaw",   -3.14, 3.14, rL[2])

        # Right
        self.ui["Right_X"] = p.addUserDebugParameter("Right_X", 0.15, 0.85, pR[0])
        self.ui["Right_Y"] = p.addUserDebugParameter("Right_Y", -0.20, 0.80, pR[1])
        self.ui["Right_Z"] = p.addUserDebugParameter("Right_Z", 0.02, 0.60, pR[2])
        self.ui["Right_Roll"]  = p.addUserDebugParameter("Right_Roll",  -3.14, 3.14, rR[0])
        self.ui["Right_Pitch"] = p.addUserDebugParameter("Right_Pitch", -3.14, 3.14, rR[1])
        self.ui["Right_Yaw"]   = p.addUserDebugParameter("Right_Yaw",   -3.14, 3.14, rR[2])

        # 튜닝 슬라이더 (진동 줄이기용)
        self.ui["KP"]     = p.addUserDebugParameter("KP", 0.02, 0.35, self.kp)
        self.ui["ALPHA"]  = p.addUserDebugParameter("ALPHA(EE smooth)", 0.05, 0.6, self.alpha)
        # self.ui["ALPHA"]  = p.addUserDebugParameter("ALPHA(EE smooth)", 0.05, 1.6, self.alpha)
        self.ui["MAX_VEL"]= p.addUserDebugParameter("MAX_VEL", 0.2, 3.0, self.maxVel)
        # self.ui["MAX_VEL"]= p.addUserDebugParameter("MAX_VEL", 0.2, 10.0, self.maxVel)
        self.ui["SLEEP"]  = p.addUserDebugParameter("SLEEP(sec)", 0.0, 0.02, self.dt)
        self.ui["HOME(0/1)"] = p.addUserDebugParameter("HOME(0/1)", 0, 1, 0)
        
        # 모드(0=EE, 1=JOINT)
        self.ui["L_MODE"] = p.addUserDebugParameter("L_MODE(0=EE,1=J)", 0, 1, 0)
        self.ui["R_MODE"] = p.addUserDebugParameter("R_MODE(0=EE,1=J)", 0, 1, 0)
        
        # 복사 버튼 (rising edge로 처리)
        self.ui["L_COPY_EE2J"] = p.addUserDebugParameter("L_COPY_EE->J(0/1)", 0, 1, 0)
        self.ui["L_COPY_J2EE"] = p.addUserDebugParameter("L_COPY_J->EE(0/1)", 0, 1, 0)
        self.ui["R_COPY_EE2J"] = p.addUserDebugParameter("R_COPY_EE->J(0/1)", 0, 1, 0)
        self.ui["R_COPY_J2EE"] = p.addUserDebugParameter("R_COPY_J->EE(0/1)", 0, 1, 0)

    
    def _read_targets(self):
        L_ee = np.array([
            p.readUserDebugParameter(self.ui["Left_X"]),
            p.readUserDebugParameter(self.ui["Left_Y"]),
            p.readUserDebugParameter(self.ui["Left_Z"]),
            p.readUserDebugParameter(self.ui["Left_Roll"]),
            p.readUserDebugParameter(self.ui["Left_Pitch"]),
            p.readUserDebugParameter(self.ui["Left_Yaw"]),
        ], dtype=np.float32)
    
        R_ee = np.array([
            p.readUserDebugParameter(self.ui["Right_X"]),
            p.readUserDebugParameter(self.ui["Right_Y"]),
            p.readUserDebugParameter(self.ui["Right_Z"]),
            p.readUserDebugParameter(self.ui["Right_Roll"]),
            p.readUserDebugParameter(self.ui["Right_Pitch"]),
            p.readUserDebugParameter(self.ui["Right_Yaw"]),
        ], dtype=np.float32)
    
        L_q = np.array([p.readUserDebugParameter(pid) for pid in self.ui["LJ"]], dtype=np.float32)
        R_q = np.array([p.readUserDebugParameter(pid) for pid in self.ui["RJ"]], dtype=np.float32)
    
        self.kp = p.readUserDebugParameter(self.ui["KP"])
        self.alpha = p.readUserDebugParameter(self.ui["ALPHA"])
        self.maxVel = p.readUserDebugParameter(self.ui["MAX_VEL"])
        sl = p.readUserDebugParameter(self.ui["SLEEP"])
        home = p.readUserDebugParameter(self.ui["HOME(0/1)"])
    
        # modes/buttons
        L_mode = int(p.readUserDebugParameter(self.ui["L_MODE"]))
        R_mode = int(p.readUserDebugParameter(self.ui["R_MODE"]))
        btns = {k: int(p.readUserDebugParameter(self.ui[k])) for k in
                ["L_COPY_EE2J","L_COPY_J2EE","R_COPY_EE2J","R_COPY_J2EE"]}
    
        return L_ee, R_ee, L_q, R_q, L_mode, R_mode, btns, sl, home

    
    def _update_readback(self):
        # Left actual
        qL_now = [p.getJointState(self.urL, jid)[0] for jid in self.jL]
        (pL, rL) = self._get_ee_pose(self.urL, self.eeL)
        msgL = f"[L] EE: ({pL[0]:.3f},{pL[1]:.3f},{pL[2]:.3f})  RPY: ({rL[0]:+.2f},{rL[1]:+.2f},{rL[2]:+.2f})\n" \
               f"    Q: " + " ".join([f"{x:+.2f}" for x in qL_now])
    
        # Right actual
        qR_now = [p.getJointState(self.urR, jid)[0] for jid in self.jR]
        (pR, rR) = self._get_ee_pose(self.urR, self.eeR)
        msgR = f"[R] EE: ({pR[0]:.3f},{pR[1]:.3f},{pR[2]:.3f})  RPY: ({rR[0]:+.2f},{rR[1]:+.2f},{rR[2]:+.2f})\n" \
               f"    Q: " + " ".join([f"{x:+.2f}" for x in qR_now])
    
        # 화면 상단에 고정 표시(좌표는 적당히)
        posL = [0.1, -0.9, 1.2]
        posR = [0.1, -0.9, 1.0]
    
        if self._readback_id_L is None:
            self._readback_id_L = p.addUserDebugText(msgL, posL, textSize=1.1, lifeTime=0)
        else:
            self._readback_id_L = p.addUserDebugText(msgL, posL, textSize=1.1, lifeTime=0,
                                                     replaceItemUniqueId=self._readback_id_L)
    
        if self._readback_id_R is None:
            self._readback_id_R = p.addUserDebugText(msgR, posR, textSize=1.1, lifeTime=0)
        else:
            self._readback_id_R = p.addUserDebugText(msgR, posR, textSize=1.1, lifeTime=0,
                                                     replaceItemUniqueId=self._readback_id_R)
    
               
            
    def _sync_ee_sliders(self, arm_prefix, ee_vec6):
        #슬라이더 “종속 갱신” 함수들
        if arm_prefix == "L":
            keys = ["Left_X","Left_Y","Left_Z","Left_Roll","Left_Pitch","Left_Yaw"]
        else:
            keys = ["Right_X","Right_Y","Right_Z","Right_Roll","Right_Pitch","Right_Yaw"]
    
        for k, key in enumerate(keys):
            # resetUserDebugParameter가 없으면 그냥 스킵
            self._try_set_debug_param(self.ui[key], float(ee_vec6[k]))
    
    def _sync_joint_sliders(self, arm_prefix, q6):
        #슬라이더 “종속 갱신” 함수들
        pids = self.ui["LJ"] if arm_prefix == "L" else self.ui["RJ"]
        for pid, val in zip(pids, q6):
            self._try_set_debug_param(pid, float(val))

            
    def _can_set_debug_param(self):
        return hasattr(p, "resetUserDebugParameter")
    
    def _try_set_debug_param(self, param_id, value):
        """가능하면 슬라이더 값 강제 변경(너 환경에서는 보통 False)."""
        if self._can_set_debug_param():
            p.resetUserDebugParameter(param_id, float(value))
            return True
        return False

            
    def _who_controls(self, cur_ee, cur_q, prev_ee, prev_q, ee_w=1.0, q_w=1.0, thresh=3e-3):
        #“EE가 움직였나 / Joint가 움직였나” 자동 판정
        """
        변화량 비교로 입력 주도권 판단.
        thresh: 너무 작은 노이즈 변화는 무시
        """
        
        de = float(np.sum(np.abs(cur_ee - prev_ee)))
        dq = float(np.sum(np.abs(cur_q - prev_q))) if len(cur_q)>0 else 0.0
    
        # 스케일 차이 보정(EE는 m+rad, q는 rad)
        score_ee = ee_w * de
        score_q  = q_w  * dq
    
        if max(score_ee, score_q) < thresh:
            return "NONE", de, dq
    
        return ("EE" if score_ee >= score_q else "J"), de, dq



    def _smooth(self, prev, x, alpha):
        # EMA: prev <- (1-a)*prev + a*x
        return (1.0 - alpha) * prev + alpha * x

    def _ik_to_joints(self, rid, ee_idx, joint_ids, target_vec6, rest_pose):
        pos = target_vec6[:3].tolist()
        rpy = target_vec6[3:].tolist()
        orn = p.getQuaternionFromEuler(rpy)
    
        # ---------------------------
        # 1) IK2 (버전별 시그니처 대응)
        # ---------------------------
        if hasattr(p, "calculateInverseKinematics2"):
            try:
                # ✅ 너 빌드: endEffectorLinkIndices (list) 필요
                q = p.calculateInverseKinematics2(
                    bodyUniqueId=rid,
                    endEffectorLinkIndices=[ee_idx],
                    targetPositions=[pos],
                    targetOrientations=[orn],
                    jointIndices=joint_ids,
                    jointDamping=getattr(self, "joint_damping", None),
                    restPoses=rest_pose,
                )
                # IK2는 경우에 따라 (n_end_effectors * n_joints) 형태로 flatten될 수 있음
                # endEffector 1개면 앞쪽 joint 개수만 사용
                return list(q[:len(joint_ids)])
            except TypeError:
                # 다른 빌드 시그니처면 아래 try로 한 번 더
                try:
                    q = p.calculateInverseKinematics2(
                        rid,
                        [ee_idx],
                        [pos],
                        [orn],
                        jointIndices=joint_ids,
                        jointDamping=getattr(self, "joint_damping", None),
                        restPoses=rest_pose,
                    )
                    return list(q[:len(joint_ids)])
                except Exception:
                    pass  # IK1 fallback
    
        # ---------------------------
        # 2) IK1 fallback (항상 있음)
        # ---------------------------
        q = p.calculateInverseKinematics(
            bodyUniqueId=rid,
            endEffectorLinkIndex=ee_idx,
            targetPosition=pos,
            targetOrientation=orn,
            jointDamping=getattr(self, "joint_damping", None),
            restPoses=rest_pose,
        )
    
        # ✅ q는 보통 "제어관절 순서"라서 6개면 그대로, 아니면 앞 6개
        if len(q) >= len(joint_ids):
            return list(q[:len(joint_ids)])
        return list(q)


    


    def _apply_q(self, rid, joint_ids, q, max_forces):
        # 관절별 위치 제어 + 속도 제한(진동 감소에 매우 중요)
        for jid, qi, mf in zip(joint_ids, q, max_forces):
            p.setJointMotorControl2(
                rid, jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=qi,
                force=mf,
                positionGain=self.kp,
                velocityGain=self.kd,
                maxVelocity=self.maxVel
            )
            
    
    def is_sustained_sat(self, sat, tag="L"):
        if tag=="L":
            self.sat_count_L = self.sat_count_L + 1 if sat else 0
            return self.sat_count_L >= self.sat_warn_frames
        else:
            self.sat_count_R = self.sat_count_R + 1 if sat else 0
            return self.sat_count_R >= self.sat_warn_frames

    def _draw_torque_texts(self, rid, joint_ids, taus, limits, text_ids, prefix="L"):
        """관절 근처에 토크 표시. limit 초과면 빨간색."""
        for k, (jid, tau, lim) in enumerate(zip(joint_ids, taus, limits)):
            # joint 근처 좌표: 해당 joint의 link frame 위치를 사용
            # (joint index = link index인 경우가 대부분)
            try:
                ls = p.getLinkState(rid, jid, computeForwardKinematics=True)
                pos = ls[4]  # worldLinkFramePosition
            except:
                # fallback: base pos
                pos, _ = p.getBasePositionAndOrientation(rid)

            ratio = abs(tau) / max(lim, 1e-6)
            over = ratio >= 1.0

            color = [1, 0, 0] if over else [0, 1, 0]   # red if over else green
            txt = f"{prefix}J{k+1}: {tau:+.2f} / {lim:.1f} Nm ({ratio*100:.0f}%)"

            if text_ids[k] is None:
                text_ids[k] = p.addUserDebugText(
                    txt, pos,
                    textColorRGB=color,
                    textSize=1.2,
                    lifeTime=0,  # 0이면 유지됨(우리가 replace로 업데이트)
                )
            else:
                text_ids[k] = p.addUserDebugText(
                    txt, pos,
                    textColorRGB=color,
                    textSize=1.2,
                    lifeTime=0,
                    replaceItemUniqueId=text_ids[k]
                )
                
    def _check_torque_over_and_print(self, robot_id, joint_ids, limits, tag="L", limit="ON"):
        # joint torque: getJointState -> (pos, vel, reactionForces, appliedMotorTorque)
        taus = []
        for jid in joint_ids:
            js = p.getJointState(robot_id, jid)
            taus.append(float(js[3]))  # applied motor torque (approx)
    
        now = time.time()
        if now - self._torque_last_print_t < self._torque_print_interval:
            return taus
    
        # ratio 계산
        ratios = []
        for tau, lim in zip(taus, limits):
            if lim <= 0:
                ratios.append(0.0)
            else:
                ratios.append(abs(tau) / float(lim))
    
        if limit.upper() == "OFF":
            # 항상 출력
            msg = " | ".join([f"J{i+1}:{taus[i]:+.2f}Nm/{limits[i]:.1f}(x{ratios[i]:.2f})"
                              for i in range(len(taus))])
            print(f"[TORQUE {tag}] {msg}")
            self._torque_last_print_t = now
            return taus
    
        # limit == "ON" -> 초과만 출력
        over = [(i, taus[i], limits[i], ratios[i]) for i in range(len(taus)) if ratios[i] >= self._torque_ratio_th]
        if over:
            msg = " | ".join([f"J{i+1}:{tau:+.2f}Nm/{lim:.1f}(x{ratio:.2f})" for (i, tau, lim, ratio) in over])
            print(f"[TORQUE OVER {tag}] {msg}")
            self._torque_last_print_t = now
    
        return taus

    def torque_saturation_ratio(self, robot_id, joint_ids, maxF):
        taus = [float(p.getJointState(robot_id, jid)[3]) for jid in joint_ids]
        ratios = [abs(t)/mf if mf>1e-6 else 0.0 for t, mf in zip(taus, maxF)]
        sat = any(r > 0.95 for r in ratios)  # 95% 이상이면 포화로 판단
        return taus, ratios, sat
    
    
    def border_indices_from_verts(self, verts, edge_band=0):
        """
        verts: p.getMeshData(...)[1]  ([(x,y,z), ...])
        edge_band: 0이면 가장 바깥 1줄, 1이면 2줄 ...
        return: border_indices(list[int]), N_grid(or None)
        """
        numv = len(verts)
        N = int(round(math.sqrt(numv)))

        # 케이스 A: 정사각 그리드로 딱 맞는 경우 (가장 흔함)
        if N * N == numv:
            idxs = []
            for idx in range(numv):
                i = idx // N
                j = idx % N
                if (i <= edge_band) or (j <= edge_band) or (i >= N-1-edge_band) or (j >= N-1-edge_band):
                    idxs.append(idx)
            return idxs, N

        # 케이스 B: 정사각 그리드가 아닌 경우 → bbox 기반 판정 (최후의 안전장치)
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # edge_band를 "거리"로 환산하기 위해 대략적인 셀 간격 추정
        # 여기서는 sqrt(numv)로 거칠게 셀 크기를 추정
        approxN = max(int(round(math.sqrt(numv))), 2)
        dx = (x_max - x_min) / (approxN - 1)
        dy = (y_max - y_min) / (approxN - 1)
        thx = (edge_band + 0.5) * dx + 1e-6
        thy = (edge_band + 0.5) * dy + 1e-6

        idxs = []
        for idx, (x, y, z) in enumerate(verts):
            is_edge = (abs(x - x_min) < thx) or (abs(x - x_max) < thx) or (abs(y - y_min) < thy) or (abs(y - y_max) < thy)
            if is_edge:
                idxs.append(idx)
        return idxs, None
    
    def spawn_mesh_grid(self,
        visual_mesh_path,
        collision_mesh_path,   # ✅ convex hull obj
        mesh_scale=0.003,
        mass=0.001,
        xs=(0.43, 0.50, 0.57),
        ys=(-0.14, -0.07, 0.0, 0.07, 0.14),
        zs=(0.03, 0.1),
        quat=None,
        dyn_kwargs=None,
    ):
        if quat is None:
            quat = p.getQuaternionFromEuler([0, 0, 0])
    
        visual_mesh_path = os.path.abspath(visual_mesh_path)
        collision_mesh_path = os.path.abspath(collision_mesh_path)
    
        p.setAdditionalSearchPath(os.path.dirname(visual_mesh_path))
    
        # ✅ collision은 convex hull 메쉬를 사용 (flags 필요 없음)
        col = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=collision_mesh_path,
            meshScale=[mesh_scale]*3
        )
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=visual_mesh_path,
            meshScale=[mesh_scale]*3
        )
    
        if dyn_kwargs is None:
            dyn_kwargs = dict(
                lateralFriction=2.0,
                spinningFriction=0.2,
                rollingFriction=0.05,
                restitution=0.0,
                linearDamping=0.6,
                angularDamping=0.6,
            )
    
        ids = []
        for x in xs:
            for y in ys:
                for z in zs:
                    bid = p.createMultiBody(
                        baseMass=mass,
                        baseCollisionShapeIndex=col,
                        baseVisualShapeIndex=vis,
                        basePosition=[x, y, z],
                        baseOrientation=quat
                    )
                    p.changeDynamics(bid, -1, **dyn_kwargs)
                    ids.append(bid)
        return ids

    

    
    def spawn_object_grid(self,
        objecturdf="sphere_small.urdf",
        sphere_scale=1.0,
        zs=(0.03, 0.1),
        xs=(0.43, 0.50, 0.57),
        ys=(-0.14, -0.07, 0.0, 0.07, 0.14),
        quat=None,
        dyn_kwargs=None,
    ):
        """
        동일 URDF 구체를 격자 형태로 생성하고, 동역학 파라미터를 일괄 적용.
        반환: sphere_ids(list)
        """
        if quat is None:
            quat = p.getQuaternionFromEuler([0, 0, 0])
    
        if dyn_kwargs is None:
            # dyn_kwargs = dict(
            #     lateralFriction=0.8,
            #     spinningFriction=0.02,
            #     rollingFriction=0.01,
            #     restitution=0.0,
            # )
            dyn_kwargs = dict(
                mass = 0.05,
                lateralFriction=1.5,
                spinningFriction=0.05,
                rollingFriction=0.0,
                restitution=0.0,
                linearDamping=0.8,   
                angularDamping=0.8,
                # 아래 2개는 버전에 따라 지원됨 (되면 매우 효과 큼)
                # linearSleepingThreshold=0.02,
                # angularSleepingThreshold=0.02,
            )
    
        sphere_ids = []
        for x in xs:
            for y in ys:
                for z in zs:
                    sid = p.loadURDF(
                        objecturdf,
                        [x, y, z],
                        quat,
                        globalScaling=sphere_scale
                    )
                    p.changeDynamics(sid, -1, **dyn_kwargs)
                    sphere_ids.append(sid)
    
        return sphere_ids
    
    
    def min_border_distance_xy(self,cloth_bottom, cloth_top, edge_band=1):
        vb, vb_raw = self.get_soft_verts(cloth_bottom)
        vt, vt_raw = self.get_soft_verts(cloth_top)
        bb, _ = self.border_indices_from_verts(vb_raw, edge_band=edge_band)
        bt, _ = self.border_indices_from_verts(vt_raw, edge_band=edge_band)
    
        vb_b = vb[bb][:,:2]
        vt_b = vt[bt][:,:2]
    
        # 빠른 근사: top의 각 border점에 대해 bottom border 중 최근접 거리 계산
        # (KdTree가 더 빠르지만 여기선 간단히)
        # 최솟값만 필요하니 샘플링해도 됨
        min_d = 1e9
        step = max(len(vt_b)//200, 1)  # 너무 많으면 샘플링 (최대 200개 정도)
        for pt in vt_b[::step]:
            d = vb_b - pt
            dd = np.min(np.sum(d*d, axis=1))
            if dd < min_d:
                min_d = dd
        return float(np.sqrt(min_d))
    
    def get_soft_verts(self,soft_id):
        md = p.getMeshData(soft_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        return np.array(md[1], dtype=np.float32), md[1]

    
    def _get_joint_limit(self, rid, jid, default=(-2*np.pi, 2*np.pi)):
        #조인트 리밋 얻기 (가능하면 URDF 리밋 사용)
        info = p.getJointInfo(rid, jid)
        # info[8]=lower, info[9]=upper
        lo, hi = float(info[8]), float(info[9])
        if hi <= lo or abs(hi) > 1e8 or abs(lo) > 1e8:
            return default
        # UR5 같은 경우 +/-2pi 정도로 넉넉히 잡아도 OK
        return (lo, hi)
    
    def _create_joint_sliders(self):
        #조인트 슬라이더 생성
        # Left joints
        self.ui["LJ"] = []
        for k, jid in enumerate(self.jL):
            lo, hi = self._get_joint_limit(self.urL, jid, default=(-2*np.pi, 2*np.pi))
            init_q = p.getJointState(self.urL, jid)[0]
            pid = p.addUserDebugParameter(f"L_J{k+1}", lo, hi, init_q)
            self.ui["LJ"].append(pid)
    
        # Right joints
        self.ui["RJ"] = []
        for k, jid in enumerate(self.jR):
            lo, hi = self._get_joint_limit(self.urR, jid, default=(-2*np.pi, 2*np.pi))
            init_q = p.getJointState(self.urR, jid)[0]
            pid = p.addUserDebugParameter(f"R_J{k+1}", lo, hi, init_q)
            self.ui["RJ"].append(pid)
    
        # (선택) 어떤 입력이 우선인지 보기 위한 디버그 토글/표시도 가능
        # self.ui["SYNC_THRESH"] = p.addUserDebugParameter("SYNC_THRESH", 1e-4, 1e-2, 3e-3)


    
    def check_dual_collision(self,robotA, robotB, safety_dist=0.02):
        # FK/충돌 업데이트
        p.performCollisionDetection()
    
        # 1) 실제 접촉/침투 체크
        contacts = p.getContactPoints(bodyA=robotA, bodyB=robotB)
        if len(contacts) > 0:
            return True, 0.0, contacts  # 충돌
    
        # 2) 근접 거리 체크(사전 충돌 방지)
        near = p.getClosestPoints(bodyA=robotA, bodyB=robotB, distance=safety_dist)
        if len(near) > 0:
            min_d = min([pt[8] for pt in near])  # pt[8] = contact distance
            return False, float(min_d), near     # 충돌은 아니지만 위험

        return False, float("inf"), []

    def step(self):

        L_mode = self.cmd["L_mode"]
        R_mode = self.cmd["R_mode"]
        
        if L_mode == 0:
            self.filtL = self._smooth(self.filtL, self.cmd["L_ee"], self.alpha)
            qL = self._ik_to_joints(self.urL, self.eeL, self.jL, self.filtL, self.homeL)
        else:
            qL = list(self.cmd["L_q"])
        
        if R_mode == 0:
            self.filtR = self._smooth(self.filtR, self.cmd["R_ee"], self.alpha)
            qR = self._ik_to_joints(self.urR, self.eeR, self.jR, self.filtR, self.homeR)
        else:
            qR = list(self.cmd["R_q"])
        
    
        # 모터 적용
        self._apply_q(self.urL, self.jL, qL, self.maxF_L)
        self._apply_q(self.urR, self.jR, qR, self.maxF_R)
    
        # 저장
        self._save_if_pressed(qL, qR)
    
        # 충돌 체크
        is_collide, min_dist, info = self.check_dual_collision(self.urL, self.urR, safety_dist=0.02)
        if is_collide:
            print("collision:", is_collide, "min_dist:", min_dist)
    
        p.stepSimulation()
    
        # ✅ readback 표시(펜던트의 현재값 표시 역할)
        self._update_readback()
    
        # if self.gui:
        #     time.sleep(sleep_sec)



    def run(self):
        while p.isConnected():
            self.step()


if __name__ == "__main__":
    Log_Name='teleop_log1'
    Clothmeshnum=10
    
    LOG_PATH = "configs/"+Log_Name+".jsonl"
    UR5_URDF = r'D:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/tutorial/urdf/ur_description/urdf/ur5_tool0_plate.urdf'
    cloth_obj = "cloth_"+str(Clothmeshnum)+"x"+str(Clothmeshnum)+"_zup.obj"
    
    # sim = DualUR5EEGuiIK(
    #     gui=True,
    #     urdf_path_left=UR5_URDF,
    #     urdf_path_right=UR5_URDF,
    #     logpath=LOG_PATH,
    #     Clothobj=cloth_obj
    # )
    # sim.run()

