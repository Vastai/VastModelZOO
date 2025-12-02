import os
import sys
import argparse


def clear_data():
    os.system("cd kitti_eval_system && rm -rf evaluate_object label results")
    os.system("cd kitti_eval_system/kitti_eval_src && rm -rf build")
    os.system("cd evals && rm -rf kitti")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="asdfadffuck",
        help="output dir",
    )
    args = parser.parse_args()

    output_dir=os.path.join(os.getcwd(), args.out_dir)+"/"

   
    os.path.dirname(os.path.abspath(__file__))
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_file_path)

    clear_data()

    cmd = f"python3 int8_pointpillar_with_detection_eval.py --output_path {output_dir}"

    os.system(
        "cd kitti_eval_system/kitti_eval_src && mkdir -p build && cd build && cmake .. && make && mv evaluate_object ../.."
    )

    os.system("mkdir -p evals/kitti")
    os.system(
        "cd evals/kitti && ln -s /opt/vastai/vaststreamx/data/datasets/kitti_val/calib calib"
    )
    os.system(
        "cd evals/kitti && ln -s /opt/vastai/vaststreamx/data/datasets/kitti_val/kitti_infos_val.pkl   kitti_infos_val.pkl"
    )
    os.system(
        "cd kitti_eval_system && ln -s /opt/vastai/vaststreamx/data/datasets/kitti_val/label_2 label"
    )

    os.system("cd kitti_eval_system && mkdir -p results/data")

    os.system(cmd)

    os.system("cd kitti_eval_system && ./evaluate_object  ./label ./results/")

    clear_data()
