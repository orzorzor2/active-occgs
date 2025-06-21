import torch
import numpy as np
import open3d as o3d
from typing import Tuple
import os

class ExplorationMap:
    def __init__(self, bounding_box, voxel_size, device='cpu', transform=None, use_xyz_filter = None, xy_sampling_step = None, gs_z_levels = None):
        """
        Initialize the ExplorationMap with a bounding box, voxel size, device, and optional transform.

        Parameters:
            bounding_box (list)               : [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            voxel_size (float)                : Size of each voxel in the grid
            device (str)                      : The device to store the tensors ('cpu' or 'cuda')
            transform (torch.Tensor, optional): 4x4 transformation matrix to reposition the origin (sim2slam)
        """
        self.bounding_box = bounding_box
        self.voxel_size = voxel_size
        self.device = device
        self.sim2slam = transform.to(self.device) if transform is not None else torch.eye(4, device=device)
        self.slam2sim = torch.inverse(self.sim2slam)
        self.occupancy_grid, self.origin = self.create_occupancy_grid() # self.origin in Sim Space
        self.gs_z_levels = gs_z_levels 
        self.update_prev_free_voxels(
            use_xyz_filter=use_xyz_filter, xy_sampling_step=xy_sampling_step, gs_z_levels=gs_z_levels
            )

    def create_occupancy_grid(self):
        """
        Create a 3D occupancy grid based on the bounding box and voxel size.

        Returns:
            occupancy_grid (torch.Tensor): A 3D tensor representing the occupancy grid
            origin (torch.Tensor)        : The origin point of the occupancy grid in world coordinates
        """
        # Compute the dimensions of the occupancy grid
        min_bound = torch.tensor([self.bounding_box[0][0], self.bounding_box[1][0], self.bounding_box[2][0]], device=self.device)
        max_bound = torch.tensor([self.bounding_box[0][1], self.bounding_box[1][1], self.bounding_box[2][1]], device=self.device)
        grid_dimensions = ((max_bound - min_bound) / self.voxel_size).ceil().long()

        # Create an empty occupancy grid initialized to 0 (unoccupied)
        occupancy_grid = torch.zeros(*grid_dimensions, dtype=torch.float32, device=self.device, requires_grad=False)
        
        # Return the occupancy grid and the origin (minimum bound)
        return occupancy_grid, min_bound

    def mark_voxel_occupied(self, indices):
        """
        Mark a voxel as occupied in the occupancy grid.

        Parameters:
            indices (tuple): The indices of the voxel to mark as occupied
        """
        self.occupancy_grid[indices] = 1.0

    def get_world_coordinates_from_grid(self, 
                                        value: float = None, 
                                        in_slam_world: bool = False
                                        ) -> torch.Tensor:
        """
        Get the world coordinates of voxels in the grid.

        Parameters:
            value: If specified, return coordinates of voxels with this value

        Returns:
            world_coords: Coordinates in the world frame, shape (N, 3)
        """
        if value is None:
            indices = torch.nonzero(self.occupancy_grid, as_tuple=False)
        else:
            indices = torch.nonzero(self.occupancy_grid == value, as_tuple=False)
        
        # Convert voxel indices to world coordinates
        world_coords_sim = self.origin + indices * self.voxel_size
        N = world_coords_sim.shape[0]
        homogeneous_coords = torch.cat((world_coords_sim, torch.ones(N, 1, device=self.device)), dim=1)
        world_coords = torch.mm(homogeneous_coords, self.sim2slam.T)[:, :3] if in_slam_world else homogeneous_coords[:, :3] 
        return world_coords

    def transform_xyz_to_vxl(self, xyz: torch.Tensor):
        """
    
        Args:
            xyz: [N,3], XYZ in Simulator coordinate system
    
        Returns:
            vxl: [N,3], voxel indices in voxel space
    
        Attributes:
            
        """
        return (xyz - self.origin) / self.voxel_size
  
    def compute_min_distance_from_occ(self, grid: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        """
        Compute the closest distance to the occupied voxel to each point in a tensor of query points in a 3D occupancy grid,

        Args:
            grid (torch.Tensor): A 3D occupancy grid where occupied voxels are marked by 1, and free space by 0 or -1.
            query_points (torch.Tensor): A tensor of shape (M, 3), where each row represents a query point (x, y, z).

        Returns:
            min_distance (torch.Tensor): min distance to the occupied grid
        """
        ### Step 1: Get coordinates of all occupied voxels in the grid ###
        occupied_voxel_coords = torch.nonzero(grid == 1, as_tuple=False)  ### Shape (num_occupied, 3)

        ### initialization case ###
        if occupied_voxel_coords.size(0) == 0:
            return query_points, query_points.new_empty((0, 3))

        ### Step 2: Expand query points and occupied voxel coordinates for broadcasting ###
        query_points_exp = query_points.unsqueeze(1).float()  ### Shape (M, 1, 3)
        occupied_voxel_coords_exp = occupied_voxel_coords.unsqueeze(0)  ### Shape (1, num_occupied, 3)

        ### Step 3: Calculate squared Euclidean distances between each query point and each occupied voxel ###
        distances = torch.norm(query_points_exp - occupied_voxel_coords_exp, dim=2)  ### Shape (M, num_occupied)

        ### Step 4: Find the minimum distance and the corresponding nearest occupied voxel for each query point ###
        min_distances, _ = distances.min(dim=1)  ### Shape (M,)
        return min_distances

    @torch.no_grad()
    def find_free_indices(self, grid: torch.Tensor, query_points: torch.Tensor, dist_thre: float = 0.5, batch_size: int = 10000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Finds free region indices by computing the closest occupied voxel to each point in a tensor of query points in a 3D occupancy grid,
        along with the distance to each closest occupied voxel.

        Args:
            grid (torch.Tensor): A 3D occupancy grid where occupied voxels are marked by 1, and free space by 0 or -1.
            query_points (torch.Tensor): A tensor of shape (M, 3), where each row represents a query point (x, y, z).
            dist_thre (float): distance from the surface threshold

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of shape (M, 3) with the coordinates of the truncated free regions.
                - A tensor of shape (M, 3) with the coordinates of the neglected free regions.
        """
        ### Step 1: 获取所有occupied voxels###
        occupied_voxel_coords = torch.nonzero(grid == 1, as_tuple=False)  ### Shape (num_occupied, 3)      例如11177

        ### initialization case ###
        if occupied_voxel_coords.size(0) == 0:       # 当num_occupied == 0 时，返回query_points, query_points.new_empty((0, 3))
            return query_points, query_points.new_empty((0, 3))    #当环境中没有occupied voxel时, 无法计算距离, 直接将所有query_points视为自由空间

        ### Step 2: Expand query points and occupied voxel coordinates for broadcasting ###
        query_points_exp = query_points.unsqueeze(1).float()  ### 这里query_points的形状是 (M, 1, 3)           例如10241?
        occupied_voxel_coords_exp = occupied_voxel_coords.unsqueeze(0)  ### 扩展了第一个维度, 形状变成了 (1, num_occupied, 3)

        ### Step 3: 计算每个点到每个occupied voxels的距离 ###
        min_distances = []
        # batch_size = 10000
        num_repeat = query_points.shape[0] // batch_size
        num_repeat = num_repeat + 1 if num_repeat * batch_size < query_points.shape[0] else num_repeat

        #  如果需要批处理
        if num_repeat > 0:
            for i in range(num_repeat):     # 计算当前批次的点到所有occupied voxels的距离
                dist = torch.norm(query_points_exp[batch_size*i:batch_size*(i+1)] - occupied_voxel_coords_exp, dim=2)    # dist形状为 [10000,11177]
                min_distances.append(dist.min(dim=1)[0])   # 找出每个点到最近occupied voxel的距离
            min_distances = torch.cat(min_distances, dim=0)
        #  如果不需要批处理
        else:
            distances = torch.norm(query_points_exp - occupied_voxel_coords_exp, dim=2)  ### Shape (M, num_occupied)

            ### Step 4: 找出每个点到最近占据体素的距离,这里的距离是用voxel_size 来衡量的 ###
            min_distances, _ = distances.min(dim=1)  ### Shape (M,)

        ### Step 5: 找出截断自由区域和被忽略自由区域的索引  ###
        valid_free_indices_mask = (min_distances * self.voxel_size) > dist_thre    #  用欧氏距离和dist_thre比较来判断free voxels, 进行 细筛
        truncated_free_indices = query_points[valid_free_indices_mask]     
        neglected_free_indices = query_points[~valid_free_indices_mask]
        return truncated_free_indices, neglected_free_indices

    @torch.no_grad()
    def update_from_depth_map(self, 
                              depth_map : torch.Tensor,
                              intrinsics: torch.Tensor,
                              extrinsics: torch.Tensor,
                              surface_dist_thre: float,
                              find_free_indices_bs: int = 10000
                              ) -> None:
        """
        Update the occupancy grid from a depth map, marking free and occupied space.

        Parameters:
            depth_map : The depth map of shape (H, W)
            intrinsics: Camera intrinsic matrix of shape (3, 3)
            extrinsics: Camera extrinsic matrix of shape (4, 4). world-to-camera
            surface_dist_thre: threshold that free space has to be away from occupied voxel
        """
        # Get all unexplored voxel coordinates
        D, H, W = self.occupancy_grid.shape
        x, y, z = torch.meshgrid(torch.arange(D, device=self.device), torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        grid_indices = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        unexplored_mask = self.occupancy_grid.flatten() == 0
        grid_indices = grid_indices[unexplored_mask]      # 找出值==0 的voxel的索引   grid_indices形状[N, 3]   第一步:先选出 unexplored mask
        
        # Convert grid indices to world coordinates
        world_coords_sim = self.origin + grid_indices * self.voxel_size          # 形状都是 N * 3
        
        # Transform world coordinates to camera coordinates
        N = world_coords_sim.shape[0]
        homogeneous_coords_sim = torch.cat((world_coords_sim, torch.ones(N, 1, device=self.device)), dim=1)
        homogeneous_coords_slam = torch.mm(homogeneous_coords_sim, self.sim2slam.T)
        camera_coords = torch.mm(homogeneous_coords_slam, extrinsics.T)[:, :3] # from world_slam to camera_slam    camera_coords形状[N, 3]

        # Project camera coordinates to pixel coordinates
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        u = (camera_coords[:, 0] * fx / camera_coords[:, 2]) + cx    #   u 形状为 [N]
        v = (camera_coords[:, 1] * fy / camera_coords[:, 2]) + cy    #   v 形状为 [N]
        depth = camera_coords[:, 2]               #   用针孔相机模型投影,找到3D点在相机坐标系下的深度

        # Filter valid projections within image bounds
        H, W = depth_map.shape
        valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth > 0)    # 形状为 [N]   例如506098
        u = u[valid_mask].long()                                              # 形状为 [N]   例如79088
        v = v[valid_mask].long()                                              # 形状为 [N]   例如79088
        depth = depth[valid_mask]                                             # 形状变成了 [N],  例如79088
        grid_indices = grid_indices[valid_mask]                               # 形状为 [M, 3]  第二步: 在unexplored mask中, 选出valid_mask

        # Update occupancy grid based on depth map
        depth_map_values = depth_map[v, u]                                    # 形状为 [N]   例如79088
        occupied_mask = (depth_map_values - depth).abs() < self.voxel_size  #  判断体素是否occupied了,用的是深度差 和 voxel_size 的比较


        #  因为SplaTAM在相机靠近表面时不会更新地图, 所以需要保持自由空间与表面保持5个体素的距离（通过surface_dist_thre参数控制), 防止相机太靠近表面
        free_mask = (depth_map_values - depth) > surface_dist_thre  
        #  这里通过深度差和surface_dist_thre 的比较，来判断是否是free的voxel, 进行了 粗筛   下方的grid_indices[free_mask]为第三步
                                                                                                       
        free_indices, neglected_free_indices = self.find_free_indices(self.occupancy_grid, grid_indices[free_mask], dist_thre=surface_dist_thre, batch_size=find_free_indices_bs)



        # 判断完后, 开始对free_voxel 和 neglected_free_voxel 进行标记   
        # new_free_mask = self.occupancy_grid[free_indices[:, 0], free_indices[:, 1], free_indices[:, 2]] != -1
        # self._new_free_voxels = free_indices[new_free_mask]
        self.occupancy_grid[free_indices[:, 0], free_indices[:, 1], free_indices[:, 2]] = -1.0
        self.occupancy_grid[neglected_free_indices[:, 0], neglected_free_indices[:, 1], neglected_free_indices[:, 2]] = -2.0

        # 最后把occupied voxel 也标记了 标记为1
        occupied_indices = grid_indices[occupied_mask]
        self.occupancy_grid[occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2]] = 1.0
    
    def update_prev_free_voxels(self, use_xyz_filter: bool = True, xy_sampling_step: float = 1.0, gs_z_levels = None):
        """ update last stored free grid
        
        Attributes:
            prev_free_voxles: [N, 3]. indices of free voxels
            
        """
        self.prev_free_voxels = self.get_free_voxels(use_xyz_filter, xy_sampling_step, gs_z_levels)

    def get_new_free_voxels(self, use_xyz_filter: bool = True, xy_sampling_step: float = 1.0, gs_z_levels = None) -> torch.Tensor:
        """ get free voxels compared to last stored free grid

        Args:
            use_xyz_filter: use XYZ location filter
    
        Returns:
            new_free_voxels: [N, 3]. indices of free voxels
        """
        prev_free_voxels = self.prev_free_voxels
        new_free_voxels = self.get_free_voxels(use_xyz_filter, xy_sampling_step, gs_z_levels)    #  形状是 [N, 3]

        # Flatten each row into a single unique value (row-wise hashing)
        prev_flat = prev_free_voxels.view(-1, 1, 3)
        new_flat = new_free_voxels.view(1, -1, 3)
        
        # Find elements in new_free_voxels that don't match any in prev_free_voxels
        mask = (prev_flat == new_flat).all(dim=-1).any(dim=0)  #  得到 [N, M]的矩阵
        unique_new_voxels = new_free_voxels[~mask]

        return unique_new_voxels
        # return self._new_free_voxels
    
    def get_free_voxels(self, use_xyz_filter: bool = True, xy_sampling_step: float = 1.0, gs_z_levels = None) -> torch.Tensor:
        """ get free voxels in the global map
        
        Args:
            use_xyz_filter: use XYZ location filter
            xy_sampling_step: XY sampling step unit(meter)

        Returns:
            free_voxels: [N, 3]. indices of free voxels
        """
        free_mask = self.occupancy_grid == -1.0
        free_voxels = torch.stack(torch.where(free_mask), dim=1)

        if use_xyz_filter:
            gs_z_levels = torch.tensor(gs_z_levels, dtype=free_voxels.dtype, device=free_voxels.device)
            num_skip_vxl = xy_sampling_step / self.voxel_size # voxel_size = 0.05
            xyz_mask = (free_voxels[:, 0] % num_skip_vxl == 0) * (free_voxels[:, 1] % num_skip_vxl == 0) * (free_voxels[:, 2:3] == gs_z_levels).any(dim=1)
            filetered_free_voxels = free_voxels[xyz_mask]
            return filetered_free_voxels
        else:
            return free_voxels
    
    @staticmethod
    def transform_world_to_camera(world_coords: torch.Tensor, camera_extrinsics: torch.Tensor):
        """
        Transform occupancy grid coordinates from world to camera coordinates.

        Parameters:
            world_coords     : Coordinates in the world frame, shape (N, 3)
            camera_extrinsics: Extrinsic matrix [R|t] of shape (4, 4)

        Returns:
            camera_coords (torch.Tensor): Transformed coordinates in the camera frame, shape (N, 3)
        """
        # Convert world_coords to homogeneous coordinates (N, 4)
        N = world_coords.shape[0]
        homogeneous_coords = torch.cat((world_coords, torch.ones(N, 1, device=world_coords.device)), dim=1)
        
        # Apply the extrinsic transformation to obtain camera coordinates
        camera_coords = torch.mm(homogeneous_coords, camera_extrinsics[:3, :].T)
        
        return camera_coords[:, :3]

    def visualize(self, time_idx: int = 0, in_slam_world: bool = False):
        """
        Visualize the exploration map using Open3D. (in the SLAM coordinate system)

        Parameters:
            time_idx     : current iteration
            in_slam_world: convert point cloud to be in SplaTAM coorindate system, otherwise in simulation/GT system
        """
        import open3d as o3d

        # Get world coordinates of occupied, free, and unexplored voxels
        occupied_coords = self.get_world_coordinates_from_grid(1.0, in_slam_world).cpu().numpy()
        free_coords = self.get_world_coordinates_from_grid(-1.0, in_slam_world).cpu().numpy()
        unexplored_coords = self.get_world_coordinates_from_grid(0.0, in_slam_world).cpu().numpy()

        # Create point clouds for each type of voxel
        occupied_pcd = o3d.geometry.PointCloud()
        occupied_pcd.points = o3d.utility.Vector3dVector(occupied_coords)
        occupied_pcd.paint_uniform_color([1, 0, 0])  # Red for occupied

        free_pcd = o3d.geometry.PointCloud()
        free_pcd.points = o3d.utility.Vector3dVector(free_coords)
        free_pcd.paint_uniform_color([0, 1, 0])  # Green for free space

        unexplored_pcd = o3d.geometry.PointCloud()
        unexplored_pcd.points = o3d.utility.Vector3dVector(unexplored_coords)
        unexplored_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for unexplored

        # Mark the 8 corners of the bounding box with different colors
        corners = np.array([
            [self.bounding_box[0][0], self.bounding_box[1][0], self.bounding_box[2][0]],
            [self.bounding_box[0][0], self.bounding_box[1][0], self.bounding_box[2][1]],
            [self.bounding_box[0][0], self.bounding_box[1][1], self.bounding_box[2][0]],
            [self.bounding_box[0][0], self.bounding_box[1][1], self.bounding_box[2][1]],
            [self.bounding_box[0][1], self.bounding_box[1][0], self.bounding_box[2][0]],
            [self.bounding_box[0][1], self.bounding_box[1][0], self.bounding_box[2][1]],
            [self.bounding_box[0][1], self.bounding_box[1][1], self.bounding_box[2][0]],
            [self.bounding_box[0][1], self.bounding_box[1][1], self.bounding_box[2][1]]
        ])
        corner_colors = [
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
            [1, 1, 0],  # Yellow
            [0.5, 0.5, 0],  # Olive
            [0.5, 0, 0.5],  # Purple
            [0, 0.5, 0.5],  # Teal
            [0.5, 0.5, 1],  # Light Blue
            [1, 0.5, 0.5]  # Light Red
        ]
        corner_pcds = []
        for i, corner in enumerate(corners):
            corner_pcd = o3d.geometry.PointCloud()
            if in_slam_world:
                corner = (self.sim2slam.cpu().numpy() @  np.concatenate([corner, np.array([1.0])]))[:3]
            corner_pcd.points = o3d.utility.Vector3dVector([corner])
            corner_pcd = corner_pcd.voxel_down_sample(voxel_size=0.1)
            corner_pcd.points = o3d.utility.Vector3dVector([corner])
            corner_pcd.paint_uniform_color(corner_colors[i])
            corner_pcds.append(corner_pcd)

        # Store the colored point cloud as a PLY file
        all_pcds = [occupied_pcd] + corner_pcds
        combined_pcd = o3d.geometry.PointCloud()
        for pcd in all_pcds:
            combined_pcd += pcd
        o3d.io.write_point_cloud(f'tmp/3d_plot/combined_{time_idx:04}.ply', combined_pcd)
        # o3d.io.write_point_cloud(f'tmp/3d_plot/occupied_{time_idx:04}.ply', occupied_pcd)
        # o3d.io.write_point_cloud(f'tmp/3d_plot/free_{time_idx:04}.ply', free_pcd)
        # o3d.io.write_point_cloud(f'tmp/3d_plot/unexplored_{time_idx:04}.ply', unexplored_pcd)

    def transform_points_slam2sim(self, slam_coords: torch.Tensor) -> torch.Tensor:
        """将SLAM坐标系中的点转换到模拟器坐标系
        
        Args:
            slam_coords: [N,3], 在SLAM坐标系中的点
            
        Returns:
            sim_coords: [N,3], 在模拟器坐标系中的点
        """
        # 需要添加同质坐标进行变换
        N = slam_coords.shape[0]
        homogeneous_coords = torch.cat((slam_coords, torch.ones(N, 1, device=self.device)), dim=1)
        
        # 应用从SLAM到模拟器的变换矩阵
        sim_coords = torch.mm(homogeneous_coords, self.slam2sim.T)[:, :3]
        
        return sim_coords

    def compute_gaussian_based_occupancy(self, 
                                          gaussian_means: torch.Tensor,
                                          gaussian_scales: torch.Tensor,
                                          gaussian_opacities: torch.Tensor) -> torch.Tensor:
        """计算基于高斯分布的体素占据概率(向量化版本，无边界检查)"""
        # 初始化与occupancy_grid相同形状的空张量
        voxel_occupancy = torch.zeros_like(self.occupancy_grid, device=self.device)
        
        # 将SLAM坐标系中的高斯中心转换到模拟器坐标系
        gaussian_means_sim = self.transform_points_slam2sim(gaussian_means)
        
        # 计算每个高斯中心对应的体素索引
        voxel_indices = self.transform_xyz_to_vxl(gaussian_means_sim).ceil().long()
        
        # 直接使用index_put_向量化更新体素占据概率
        voxel_occupancy.index_put_(
            (voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]),
            gaussian_opacities,
            accumulate=True
        )
        
        # 归一化占据概率，使其范围在[0,1]
        voxel_occupancy = torch.clamp(voxel_occupancy, 0.0, 1.0)
        
        return voxel_occupancy
        
    def get_voxel_centers(self) -> torch.Tensor:
        """计算所有体素的中心点坐标
        
        Returns:
            centers: 形状为[D*H*W, 3]的体素中心点坐标张量
        """
        D, H, W = self.occupancy_grid.shape
        x = torch.arange(0, D, device=self.device)
        y = torch.arange(0, H, device=self.device)
        z = torch.arange(0, W, device=self.device)
        
        # 使用meshgrid生成坐标网格
        x, y, z = torch.meshgrid(x, y, z, indexing='ij')
        
        # 将indices转换为世界坐标
        centers = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).float()
        centers = self.origin + centers * self.voxel_size
        
        return centers

    def visualize_gaussian_occupancy(self, 
                                    gaussian_means: torch.Tensor,
                                    gaussian_scales: torch.Tensor,
                                    gaussian_opacities: torch.Tensor,
                                    time_idx: int = 0,
                                    threshold: float = 0.01,
                                    in_slam_world: bool = False):
        """
        可视化基于高斯分布的体素占据概率
        
        Parameters:
            gaussian_means: 高斯中心点坐标，形状为[N, 3]，在SLAM坐标系中
            gaussian_scales: 高斯尺度，形状为[N, 1]或[N, 3]
            gaussian_opacities: 高斯不透明度，形状为[N]
            time_idx: 当前迭代索引，用于文件命名
            threshold: 体素占据概率阈值，只显示超过该阈值的体素
            in_slam_world: 是否将点云转换到SLAM坐标系中显示
        """
        import open3d as o3d
        
        # 计算基于高斯的体素占据概率
        voxel_occupancy = self.compute_gaussian_based_occupancy(
            gaussian_means, gaussian_scales, gaussian_opacities)
        
        # 只保留超过阈值的体素
        thresholded_mask = voxel_occupancy > threshold
        
        # 获取超过阈值的体素索引
        occupied_indices = torch.nonzero(thresholded_mask, as_tuple=False)
        
        # 转换为世界坐标（模拟器坐标系）
        occupied_coords_sim = self.origin + occupied_indices * self.voxel_size
        
        # 如果需要转换到SLAM世界坐标系
        if in_slam_world:
            N = occupied_coords_sim.shape[0]
            homogeneous_coords = torch.cat((occupied_coords_sim, torch.ones(N, 1, device=self.device)), dim=1)
            occupied_coords = torch.mm(homogeneous_coords, self.sim2slam.T)[:, :3]
        else:
            occupied_coords = occupied_coords_sim
        
        # 创建点云
        occupied_pcd = o3d.geometry.PointCloud()
        occupied_pcd.points = o3d.utility.Vector3dVector(occupied_coords.cpu().numpy())
        
        # 根据体素占据概率设置颜色（红色越深表示概率越高）
        if occupied_indices.shape[0] > 0:
            probs = voxel_occupancy[occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2]]
            colors = torch.zeros((occupied_indices.shape[0], 3), device=self.device)
            colors[:, 0] = probs  # 红色通道设置为概率值
            colors[:, 1] = 0.3 * (1.0 - probs)  # 绿色通道
            colors[:, 2] = 0.3 * (1.0 - probs)  # 蓝色通道
            occupied_pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        
        # 保存点云文件
        os.makedirs('tmp/gaussian_voxels', exist_ok=True)
        o3d.io.write_point_cloud(f'tmp/gaussian_voxels/gaussian_voxels_{time_idx:04}.ply', occupied_pcd)
        
        # # 同时保存体素概率值，以便后续分析
        # np.save(f'tmp/gaussian_voxels/voxel_probs_{time_idx:04}.npy', voxel_occupancy.cpu().numpy())
        
        return voxel_occupancy
    
    # def visualize_raw_gaussians(self,
    #                           gaussian_means: torch.Tensor,
    #                           gaussian_scales: torch.Tensor,
    #                           gaussian_opacities: torch.Tensor,
    #                           time_idx: int = 0,
    #                           opacity_threshold: float = 0.01,
    #                           in_slam_world: bool = False):
    #     """
    #     可视化未经过体素映射的原始高斯点
        
    #     Parameters:
    #         gaussian_means: 高斯中心点坐标，形状为[N, 3]，在SLAM坐标系中
    #         gaussian_scales: 高斯尺度，形状为[N, 3]
    #         gaussian_opacities: 高斯不透明度，形状为[N]或[N, 1]
    #         time_idx: 当前迭代索引，用于文件命名
    #         opacity_threshold: 不透明度阈值，只显示超过该阈值的高斯点
    #         in_slam_world: 是否直接使用SLAM坐标系中的点（默认为True）
    #     """
    #     import open3d as o3d
        
    #     # 确保gaussian_opacities是一维张量
    #     if gaussian_opacities.ndim > 1:
    #         gaussian_opacities = gaussian_opacities.squeeze(-1)
        
    #     # 筛选出不透明度大于阈值的高斯点
    #     valid_mask = gaussian_opacities > opacity_threshold
    #     filtered_means = gaussian_means[valid_mask]
    #     filtered_scales = gaussian_scales[valid_mask]
    #     filtered_opacities = gaussian_opacities[valid_mask]
        
    #     print(f"总高斯点数量: {gaussian_means.shape[0]}, 筛选后数量: {filtered_means.shape[0]}")
        
    #     # 如果不需要在SLAM坐标系中显示，则转换到模拟器坐标系
    #     if not in_slam_world:
    #         filtered_means = self.transform_points_slam2sim(filtered_means)
        
    #     # 创建点云
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(filtered_means.cpu().numpy())
        
    #     # 根据不透明度和尺度设置颜色和点大小
    #     colors = torch.zeros((filtered_means.shape[0], 3), device=self.device)
        
    #     # 归一化不透明度，用于颜色映射
    #     normalized_opacities = filtered_opacities / filtered_opacities.max()
        
    #     # 使用热力图颜色方案：不透明度高的点为红色，低的点为蓝色
    #     colors[:, 0] = normalized_opacities  # 红色通道
    #     colors[:, 2] = 1.0 - normalized_opacities  # 蓝色通道
        
    #     pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        
    #     # 创建目录并保存原始高斯点的可视化结果
    #     os.makedirs('tmp/raw_gaussians', exist_ok=True)
    #     o3d.io.write_point_cloud(f'tmp/raw_gaussians/raw_gaussians_{time_idx:04}.ply', pcd)
        
    #     # 保存高斯参数供后续分析
    #     gaussian_data = {
    #         'means': filtered_means.cpu().numpy(),
    #         'scales': filtered_scales.cpu().numpy(),
    #         'opacities': filtered_opacities.cpu().numpy()
    #     }
    #     np.save(f'tmp/raw_gaussians/gaussian_params_{time_idx:04}.npy', gaussian_data)
        
    #     # 创建带尺度信息的可视化（使用小球代表高斯）
    #     # 注意：这可能会在高斯点数量很多时变得非常慢
    #     if filtered_means.shape[0] <= 10000:  # 限制点的数量以避免性能问题
    #         spheres = []
    #         means_np = filtered_means.cpu().numpy()
    #         scales_np = filtered_scales.cpu().numpy()
            
    #         # 根据尺度计算球体半径，可以取尺度的平均值或最大值
    #         if len(scales_np.shape) > 1 and scales_np.shape[1] == 3:
    #             radii = np.mean(scales_np, axis=1)
    #         else:
    #             radii = scales_np.squeeze()
            
    #         # 创建一个多边形集合，用于可视化
    #         for i in range(min(5000, filtered_means.shape[0])):  # 进一步限制为最多5000个球体
    #             sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(radii[i]))
    #             sphere.translate(means_np[i])
    #             sphere.paint_uniform_color(colors[i].cpu().numpy())
    #             spheres.append(sphere)
            
    #         # 合并所有球体并保存
    #         if spheres:
    #             combined_mesh = o3d.geometry.TriangleMesh()
    #             for sphere in spheres:
    #                 combined_mesh += sphere
    #             o3d.io.write_triangle_mesh(f'tmp/raw_gaussians/gaussian_spheres_{time_idx:04}.obj', combined_mesh)
        
    #     return filtered_means, filtered_scales, filtered_opacities

