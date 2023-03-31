import glob

from src.optical_flow import read_optical_flow, compute_error, compute_msen, compute_pepn, visualize_error, \
    histogram_error


def main():
    gt_path = "D:/datasets_diskD/data_stereo_flow/training/flow_noc"#'../dataset/data_stereo_flow/training/flow_noc'  # Path of the ground truths
    detections_path = '../dataset/results_opticalflow_kitti/results'  # Path of the detections

    detections = glob.glob(detections_path + "/*.png")
    detections = [det.replace("\\", "/") for det in detections]  # for Windows users

    for det_filename in detections:
        seq_n = det_filename.split("/")[-1].replace("LKflow_", "")

        mask, flow_gt = read_optical_flow(f"{gt_path}/{seq_n}")
        _, flow_det = read_optical_flow(det_filename)
        flow_noc_det = flow_det[mask]
        flow_noc_gt = flow_gt[mask]
        error = compute_error(flow_noc_gt, flow_noc_det)
        msen = compute_msen(error)
        pepn = compute_pepn(error)

        print(f"Image {seq_n}")
        print(f"MSEN: {msen}")
        print(f"PEPN: {pepn}\n")

        visualize_error(error, mask)
        histogram_error(error)


if __name__ == "__main__":
    main()
