from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
import numpy as np

class ScrewDriverController:
    def __init__(self, prim_path: str = "/World/m0609/m0609/tool0/assembly_screw/assembly_screw/tn__Part1_f5/tn__Part1_f5"):
        self.prim_path = prim_path
        self.target_prim = XFormPrim(self.prim_path)
        print(f"[ScrewController] 팁 제어 모듈 연동 완료: {self.prim_path}")

    def rotate_step(self, angle_increment: float = 0.2):
        """
        복수형 메서드(get_local_poses / set_local_poses)를 사용하여 팁 회전
        """
        # 1. 단일 프레임이라도 복수형 함수는 리스트 형태로 반환함 [0]번 인덱스 사용
        positions, quats = self.target_prim.get_local_poses()
        curr_pos = positions[0]
        curr_rot_quat = quats[0]

        # 2. 쿼터니언 -> 오일러 변환
        curr_rot_euler = quat_to_euler_angles(curr_rot_quat)

        # 3. Z축 회전값 누적
        curr_rot_euler[1] += angle_increment

        # 4. 오일러 -> 쿼터니언 변환
        new_rot_quat = euler_angles_to_quat(curr_rot_euler)

        # 5. 설정할 때도 반드시 리스트 형태로 감싸서 전달해야 함
        self.target_prim.set_local_poses(
            translations=np.array([curr_pos]),
            orientations=np.array([new_rot_quat])
        )