import numpy as np

# 读取 ground_truth.txt
gt_data = np.loadtxt("ground_truth.txt")
gt_timestamps = gt_data[:, 0]

# 读取 KeyFrameTrajectory_seconds.txt
kf_data = np.loadtxt("KeyFrameTrajectory_seconds.txt")
kf_timestamps = kf_data[:, 0]

# 定义时间戳匹配的误差范围（例如 0.01 秒）
timestamp_tolerance = 0.01

# 找到时间戳在误差范围内匹配的部分
matched_indices_gt = []
matched_indices_kf = []

for i, gt_t in enumerate(gt_timestamps):
    for j, kf_t in enumerate(kf_timestamps):
        if abs(gt_t - kf_t) <= timestamp_tolerance:
            matched_indices_gt.append(i)
            matched_indices_kf.append(j)
            break

# 保存匹配的部分
gt_matched = gt_data[matched_indices_gt]
kf_matched = kf_data[matched_indices_kf]

# 保存为新的文件
np.savetxt("ground_truth_sync.txt", gt_matched, fmt="%.6f")
np.savetxt("KeyFrameTrajectory_sync.txt", kf_matched, fmt="%.6f")

print(f"Found {len(gt_matched)} matched timestamps.")
