import glob
import re
from PIL import Image

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def make_gif(frame_folder):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"), key=numericalSort)]
    frame_one = frames[0]
    frame_one.save(frame_folder.split("/")[-1] + ".gif", format="GIF", append_images=frames,
                   save_all=True, duration=50, loop=0)


if __name__ == "__main__":
    make_gif("gif_images/detection_plots")
    make_gif("gif_images/iou_vs_frames_plots")
