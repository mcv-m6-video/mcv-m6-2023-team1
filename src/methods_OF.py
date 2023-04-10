import glob
import cv2
import numpy as np
from .optical_flow import read_optical_flow, report_errors_OF, save_plot_OF, visualize_error, histogram_error
from .optical_flow import OF_block_matching, block_matching, compute_error, compute_pepn, compute_msen
from .enums import OFMethods
from src.metrics import mean_square_error
from PIL import Image
# import pyflow

MethodsFactory = {OFMethods.Farneback:cv2.calcOpticalFlowFarneback,
                    OFMethods.OwnBlockMatching: OF_block_matching}


def of_perceiver(cfg):
    from perceiver.model.vision.optical_flow import convert_config, OpticalFlow
    from perceiver.data.vision.optical_flow import OpticalFlowProcessor
    from transformers import AutoConfig
    import torch

    prev = cv2.imread(cfg["paths"]["frame0"])
    post = cv2.imread(cfg["paths"]["frame1"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained model configuration from the Hugging Face Hub
    config = AutoConfig.from_pretrained("deepmind/optical-flow-perceiver")

    # Convert configuration, instantiate model and load weights
    model = OpticalFlow(convert_config(config)).eval().to(device)

    # Create optical flow processor
    processor = OpticalFlowProcessor(patch_size=tuple(config.train_size))

    frame_pair = (prev, post)

    optical_flow = processor.process(model, image_pairs=[frame_pair], batch_size=1, device=device).numpy()[0]

    return optical_flow


def of_pyflow(prev, post):
    img1 = np.atleast_3d(prev.astype(float) / 255.)
    img2 = np.atleast_3d(post.astype(float) / 255.)
    u, v, im2W = pyflow.coarse2fine_flow(img1, img2, 0.012, 0.75, 20, 7, 1, 30, 1)
    opflow = np.dstack((u, v))


def of_lucas_kanade(prev, post):
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))
    # Track all features of image
    p0 = np.array([[x, y] for y in range(prev.shape[0]) for x in range(prev.shape[1])], dtype=np.float32).reshape(
        (-1, 1, 2))

    p1, st, err = cv2.calcOpticalFlowPyrLK(prev, post, p0, None, **lk_params)

    p0 = p0.reshape((prev.shape[0], prev.shape[1], 2))
    p1 = p1.reshape((prev.shape[0], prev.shape[1], 2))
    st = st.reshape((prev.shape[0], prev.shape[1]))

    opflow = p1 - p0
    opflow[st == 0] = 0

    return opflow


def evaluate_kitti_week_1(cfg):
    gt_path = cfg["paths"]["gt"]
    detections_path = cfg["paths"]["kitti"]
    detections = glob.glob(detections_path + "/*.png")
    if cfg["OS_system"] =="windows":
        detections = [det.replace("\\", "/") for det in detections]  # for Windows users
    else:
        pass
    for det_filename in detections:

        seq_n = det_filename.split("/")[-1].replace("LKflow_", "")
        _, flow_det = read_optical_flow(det_filename)

        mask, error, msen, pepn = report_errors_OF(gt_path,seq_n, flow_det)

        print(f"Image {seq_n}")
        print(f"MSEN: {msen}")
        print(f"PEPN: {pepn}\n")

        visualize_error(error, mask)
        histogram_error(error)

def evaluate_optimized_blockmatching(cfg):
    area_of_search_range = [64]
    block_size_range = [16]
    step_size_range = [1]
    # area_of_search_range = [32]
    # block_size_range = [16]
    # step_size_range = [2]

    best_area = None
    best_block = None
    best_step = None

    best_msen = 10000000
    for area in area_of_search_range:
        for block in block_size_range:
            for step in step_size_range:
                print("area of search range: ", area, "block size: ", block, "step size: ", step)

                gt_path = cfg["paths"]["gt"]  # Path of the ground truths

                image45_10 = cv2.imread(cfg["paths"]["sequence45"] + "/000045_10.png", cv2.IMREAD_GRAYSCALE)
                image45_11 = cv2.imread(cfg["paths"]["sequence45"] + "/000045_11.png", cv2.IMREAD_GRAYSCALE)
                flow_det = block_matching(image45_10, image45_11, block, area, "SSD", step) #optimized block matching
                save_plot_OF(flow_det, cfg["paths"]["block_matching"] + "/flow_blockmatching_000045_10.png")
                save_plot_OF(flow_det, cfg["paths"]["block_matching2"] + f"/flow_blockmatching_000045_10_{area}_{block}_{step}.png")

                # Get the detections (in our case, the optical flow is just from sequence 45)

                detections = glob.glob(cfg["paths"]["block_matching"] + "/*.png")
                detections = [det.replace("\\", "/") for det in detections]  # for Windows users

                # Compute the metrics and plots for the detections
                for det_filename in detections:
                    if not cfg["evaluate_own_OF"]:
                        seq_n = det_filename.split("/")[-1].replace("LKflow_", "")
                        _, flow_det = read_optical_flow(det_filename)
                    else:
                        seq_n = det_filename.split("/")[-1].replace("flow_blockmatching_", "")

                    mask, flow_gt = read_optical_flow(f"{gt_path}/{seq_n}")
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

                    # keep best mse
                    if msen < best_msen:
                        best_msen = msen
                        best_area = area
                        best_block = block
                        best_step = step

    print("best area of search range: ", best_area, "best block size: ", best_block, "best step size: ", best_step)



def evaluate_methods_week4(cfg):

    gt_path = cfg["paths"]["gt"]
    image45_10 = cv2.imread(cfg["paths"]["sequence45"] + "/000045_10.png", cv2.IMREAD_GRAYSCALE)
    image45_11 = cv2.imread(cfg["paths"]["sequence45"] + "/000045_11.png", cv2.IMREAD_GRAYSCALE)

    if cfg["OF_method"]==OFMethods.OwnBlockMatching:
        flow_det = OF_block_matching(image45_10, image45_11, area_of_search=(32,32), block_size=(4,4), step_size=(4,4), error_function=mean_square_error)
    elif cfg["OF_method"]==OFMethods.Farneback:
        flow_det = cv2.calcOpticalFlowFarneback(image45_10, image45_11, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    save_plot_OF(flow_det, cfg["paths"][cfg["OF_method"]] + f"/flow_{cfg['OF_method']}_000045_10.png")

    detections = glob.glob(cfg["paths"][cfg["OF_method"]] + "/*.png")
    if cfg["OS_system"] =="windows":
        detections = [det.replace("\\", "/") for det in detections]  # for Windows users
    else:
        pass

    for det_filename in detections:
        seq_n = det_filename.split("/")[-1].replace(f"flow_{cfg['OF_method']}_", "")

        mask, error, msen, pepn = report_errors_OF(gt_path, seq_n, flow_det)

        print(f"Image {seq_n}")
        print(f"MSEN: {msen}")
        print(f"PEPN: {pepn}\n")

        visualize_error(error, mask)
        histogram_error(error)