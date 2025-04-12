import numpy as np
kf_data = np.loadtxt("KeyFrameTrajectory.txt")
kf_data[:, 0] /= 1e9  # 仅将时间戳列（第一列）从纳秒转换为秒
np.savetxt("KeyFrameTrajectory_seconds.txt", kf_data, fmt="%.6f")  # 保存为6位小数的格式
