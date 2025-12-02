import open3d_vis_utils as V
import numpy as np
import argparse
import open3d

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument("--task", type=str, default="pointcloud", choices=["pointcloud", "box3d"])
parser.add_argument('--points_file', type=str, default=None, help='specify the path of pointcloud file')
parser.add_argument('--result_file', type=str, default=None, help='specify the path of vacc result file')
args = parser.parse_args()

def decode_vacc_result_npz(result_file):
    data = np.load(result_file)
    box = data["boxes"]
    scores = data["score"]
    labels = data["label"]
    record_dict = {
        'pred_boxes': box,
        'pred_scores': scores,
        'pred_labels': labels
    }
    return [record_dict]

def vis_box3d(points_file, result_file):
    points = np.fromfile(points_file, dtype=np.float32).reshape(-1, 4)
    pred_dicts = decode_vacc_result_npz(result_file)
    V.draw_scenes(
            points=points, ref_boxes=pred_dicts[0]['pred_boxes'],
            ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'].astype(np.int32)
        )

def vis_pointcloud(points_file):
    points = np.fromfile(points_file, dtype=np.float32).reshape(-1, 4)

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    if args.task == "box3d":
        vis_box3d(points_file = args.points_file, result_file = args.result_file)
    else:
        vis_pointcloud(points_file = args.points_file)



