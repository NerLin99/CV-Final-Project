import open3d as o3d

if __name__ == '__main__':
    # pcd = o3d.io.read_point_cloud("/Users/22695/OneDrive/CV_Proj/spann3r/output/demo/s00567/s00567_conf0.001.ply")
    # pcd = o3d.io.read_point_cloud("/Users/22695/OneDrive/CV_Proj/spann3r/output/demo/s00646/s00646_conf0.001.ply")
    # pcd = o3d.io.read_point_cloud("/Users/22695/OneDrive/CV_Proj/spann3r/output/demo/scan1/scan1_conf0.001.ply")
    pcd = o3d.io.read_point_cloud("/Users/22695/OneDrive/CV_Proj/spann3r/pointcloud/test_2.ply")

    o3d.visualization.draw_geometries([pcd])