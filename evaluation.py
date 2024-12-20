import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

# def compute_chamfer_distance(pred_points, gt_points):
#     print("Computing Chamfer Distance...")
#     pred_kdtree = KDTree(pred_points)
#     gt_kdtree = KDTree(gt_points)

#     dist_pred_to_gt, _ = pred_kdtree.query(pred_points, k=1)
#     dist_gt_to_pred, _ = gt_kdtree.query(gt_points, k=1)

#     # print(dist_gt_to_pred)
#     # print(dist_gt_to_pred)

#     cd = dist_pred_to_gt.mean() + dist_gt_to_pred.mean()
    
#     return cd

def compute_voxel_iou(pred_points, gt_points, voxel_size=0.05, min_bound=None, max_bound=None):
    print("Computing Voxel IoU...")
    if min_bound is None:
        min_bound = np.minimum(pred_points.min(axis=0), gt_points.min(axis=0))
    if max_bound is None:
        max_bound = np.maximum(pred_points.max(axis=0), gt_points.max(axis=0))

    dim = ((max_bound - min_bound) / voxel_size).astype(int) + 1

    def points_to_voxel_coords(points, min_bound, voxel_size):
        coords = ((points - min_bound) / voxel_size).astype(int)
        coords = coords[(coords[:,0]>=0)&(coords[:,0]<dim[0])&
                        (coords[:,1]>=0)&(coords[:,1]<dim[1])&
                        (coords[:,2]>=0)&(coords[:,2]<dim[2])]
        return coords

    pred_voxel_coords = points_to_voxel_coords(pred_points, min_bound, voxel_size)
    gt_voxel_coords = points_to_voxel_coords(gt_points, min_bound, voxel_size)

    pred_set = set(map(tuple, pred_voxel_coords))
    gt_set = set(map(tuple, gt_voxel_coords))

    inter_set = pred_set.intersection(gt_set)
    union_set = pred_set.union(gt_set)
    iou = len(inter_set) / len(union_set) if len(union_set) > 0 else 0.0
    return iou

if __name__ == '__main__':
    print("Starting the script...")
    pred_ply_path = "/Users/22695/OneDrive/CV_Proj/spann3r/output/demo/t1/t1_conf0.001.ply"
    gt_ply_path = "/Users/22695/OneDrive/CV_Proj/spann3r/pointcloud/test_1.ply"

    pred_pcd = o3d.io.read_point_cloud(pred_ply_path)
    gt_pcd = o3d.io.read_point_cloud(gt_ply_path)

    print("Pred point cloud loaded with", len(pred_pcd.points), "points.")
    print("GT point cloud loaded with", len(gt_pcd.points), "points.")

    threshold = 100

    pred_points = np.asarray(pred_pcd.points)
    gt_points = np.asarray(gt_pcd.points)

    # cd = compute_chamfer_distance(pred_points, gt_points)
    # print(f"Chamfer Distance: {cd}")

    voxel_iou = compute_voxel_iou(pred_points, gt_points, voxel_size=0.05)
    print(f"Voxel IoU: {voxel_iou}")
