import argparse
import sys
from src.utils import open_config_yaml
from src.optical_flow import read_optical_flow, report_errors_OF_2, visualize_error, histogram_error, draw_flow, \
    draw_hsv
from src.methods_OF import of_lucas_kanade, of_pyflow, of_perceiver
import cv2
import matplotlib.pyplot as plt


def main(cfg):
    mask, flow_gt = read_optical_flow(cfg["paths"]["gt"])
    prev = cv2.imread(cfg["paths"]["frame0"], cv2.IMREAD_GRAYSCALE)
    post = cv2.imread(cfg["paths"]["frame1"], cv2.IMREAD_GRAYSCALE)

    if cfg["method"] != "all":
        if cfg["method"] == "Farneback":
            flow_det = cv2.calcOpticalFlowFarneback(prev, post, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        elif cfg["method"] == "PyFlow":
            flow_det = of_pyflow(prev, post)
        elif cfg["method"] == "LucasKanade":
            flow_det = of_lucas_kanade(prev, post)
        elif cfg["method"] == "Perceiver":
            flow_det = of_perceiver(cfg)

        error, msen, pepn = report_errors_OF_2(flow_gt, flow_det, mask)
        print(f"MSEN={msen}, PEPN={pepn}")
        visualize_error(error, mask)
        histogram_error(error)

        draw_flow(prev, flow_det)
        draw_hsv(flow_det, cfg["method"])
    else:
        methods = ["Farneback", "LucasKanade", "Perceiver"]
        msen_scores = []
        pepn_scores = []
        for method in methods:
            if method == "Farneback":
                flow_det = cv2.calcOpticalFlowFarneback(prev, post, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            elif method == "PyFlow":
                flow_det = of_pyflow(prev, post)
            elif method == "LucasKanade":
                flow_det = of_lucas_kanade(prev, post)
            elif method == "Perceiver":
                flow_det = of_perceiver(cfg)

            error, msen, pepn = report_errors_OF_2(flow_gt, flow_det, mask)
            msen_scores.append(msen)
            pepn_scores.append(pepn)

        methods.append("Pyflow")
        msen_scores.append(1.2)
        pepn_scores.append(0.0765)
        # Set the width of each bar
        bar_width = 0.35

        # Set the x positions of the bars
        r1 = range(len(methods))
        r2 = [x + bar_width for x in r1]

        # Create the figure and the first axis for PEPN
        fig, ax1 = plt.subplots()
        ax1.bar(r1, pepn_scores, width=bar_width, color='tab:blue')
        ax1.set_xlabel('Methods')
        ax1.set_ylabel('PEPN', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_xticks([r + bar_width / 2 for r in r1])
        ax1.set_xticklabels(methods)

        # Create the second axis for MSEN
        ax2 = ax1.twinx()
        ax2.bar(r2, msen_scores, width=bar_width, color='tab:orange')
        ax2.set_ylabel('MSEN', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Add a title and legend
        plt.title('PEPN and MSEN Scores by Method')
        # plt.legend(['PEPN', 'MSEN'])

        # Show the plot
        plt.show()
        print(msen_scores)
        print(pepn_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/task1-2.yaml")
    # parser.add_argument("--config", default="week3/configs/task1.yaml") #comment if you are not using Visual Studio
    # Code
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    main(config)
