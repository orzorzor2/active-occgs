"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from tensorboardX import SummaryWriter
import torch

from src.naruto.cfg_loader import argument_parsing, load_cfg
from src.planner import init_planner
from src.slam import init_SLAM_model
from src.simulator import init_simulator
from src.utils.timer import Timer
from src.utils.general_utils import fix_random_seed, InfoPrinter, update_module_step
from src.visualization import init_visualizer


if __name__ == "__main__":
    info_printer = InfoPrinter("ActiveLang")
    timer = Timer()

    ##################################################
    ### argument parsing and load configuration
    ##################################################
    info_printer("Parsing arguments...", 0, "Initialization")
    args = argument_parsing()
    info_printer("Loading configuration...", 0, "Initialization")
    main_cfg = load_cfg(args)
    main_cfg.dump(os.path.join(main_cfg.dirs.result_dir, 'main_cfg.json'))
    info_printer.update_total_step(main_cfg.general.num_iter)
    info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)

    ##################################################
    ### Fix random seed
    ##################################################
    info_printer("Fix random seed...", 0, "Initialization")
    fix_random_seed(main_cfg.general.seed)

    ##################################################
    ### initialize logger
    ##################################################
    log_savedir = os.path.join(main_cfg.dirs.result_dir, "logger")
    os.makedirs(log_savedir, exist_ok=True)
    logger = SummaryWriter(f'{log_savedir}')
    
    ##################################################
    ### initialize simulator
    ##################################################
    sim = init_simulator(main_cfg, info_printer)        

    ##################################################
    ### initialize SLAM module
    ##################################################
    slam = init_SLAM_model(main_cfg, info_printer, logger)
    map_iter_og = slam.config['mapping']['num_iters']

    ##################################################
    ### initialize planning module
    ##################################################
    planner = init_planner(main_cfg, info_printer)
    planner.update_sim(sim)
    # planner.init_local_planner()

    ##################################################
    ### initialize visualizer
    ##################################################
    visualizer = init_visualizer(main_cfg, info_printer)

    ##################################################
    ### Run ActiveLang
    ##################################################
    ## load initial pose and convert from RUB to RDF (splatam)) ##
    c2w_slam = planner.load_init_pose() # RUB
    c2w_slam[:3, 1] *= -1
    c2w_slam[:3, 2] *= -1 # RDF
    c2w_slam_init = c2w_slam.clone() # RDF

    ## initialize exploration map in slam ##
    T_sim2slam = torch.inverse(c2w_slam_init) # RDF # transformation that takes sim-world points to slam-world-origin (i.e. first camera)
    slam.init_exploration_map(T_sim2slam)

    planner.init_data(T_sim2slam)
    if main_cfg.planner.method == "active_gs":
        planner.init_local_planner()

    ### add timer for planning related timing ###
    planner.timer = timer

    ### create directory for gaussian voxels visualization ###
    os.makedirs('tmp/gaussian_voxels', exist_ok=True)

    for i in range(main_cfg.general.num_iter):                                 #为什么这里是2000但是最终到几百就停了
    # for i in range(0, main_cfg.general.num_iter, 10):
        ##################################################
        ### update module infomation (e.g. step)
        ##################################################
        update_module_step(i, [sim, slam, planner, visualizer])              #  这里具体是干什么

        ##################################################
        ### load pose and transform pose
        ##################################################
        if main_cfg.planner.method == "predefined_traj":
            c2w_slam = planner.update_pose(c2w_slam, i).to(c2w_slam.device) # RUB
            c2w_sim = c2w_slam.cpu().numpy().copy() # RUB
            ## convert back to RDF (splatam) ##
            c2w_slam[:3, 1] *= -1 
            c2w_slam[:3, 2] *= -1
        elif main_cfg.planner.method in ["active_lang", "active_gs"]:
            ## convert back to RUB (habitat) ##
            c2w_sim = c2w_slam.cpu().numpy().copy() # RDF
            c2w_sim[:3, 1] *= -1 
            c2w_sim[:3, 2] *= -1 # RUB
        else:
            raise NotImplementedError

        ## convert to relative pose (w.r.t first pose) ##
        c2w_slam_rel = torch.inverse(c2w_slam_init) @ c2w_slam # RDF
        
        ##################################################
        ### Simulation
        ##################################################
        timer.start("Simulation", "General")
        sim_out = sim.simulate(c2w_sim)       # depth由Habitat模拟器生成
        timer.end("Simulation")
        color = sim_out['color']         #  这里color 是 1*H*W*3 的tensor, 每个值代表相机坐标系下对应颜色
        depth = sim_out['depth']         #  这里depth 是 1*H*W 的tensor, 每个值代表相机坐标系下对应深度
        if main_cfg.visualizer.vis_rgbd:
            visualizer.visualize_rgbd(color, depth, main_cfg.visualizer.vis_rgbd_max_depth)
        
        ##################################################
        ### save data for comprehensive visualization
        ##################################################
        # if main_cfg.visualizer.enable_all_vis:
        #     visualizer.main(slam, planner, color, depth, c2w_slam)

        ##################################################
        ### Mapping optimization        这里没有调用到active_lang_planner.py的内容
        ##################################################
        ### get timer state ###
        planner_state = f"{planner.planning_state}_{planner.exploration_stage}" if planner.planning_state == "exploration" else planner.planning_state
        slam_state = f"SLAM_{planner_state}"
        timer.start(slam_state, "General")

        ### slam options ###
        force_map_update = planner.state == "planning" or planner.planning_state == "post_refinement"        
        dont_add_kf = planner.state == "planning"       
        only_use_global_keyframe = main_cfg.slam.use_global_keyframe and planner.planning_state == "post_refinement"
        slam.seperate_densification_res = not(planner.planning_state == "post_refinement")

        slam.online_recon_step(i, color, depth, c2w_slam_rel, force_map_update, dont_add_kf, only_use_global_keyframe)   # 这里的depth参数就是函数里的depth_map
        timer.end(slam_state)
        
        ##################################################
        ### Active Planning            主要是是这里调用了active_lang_planner.py的内容
        ##################################################
        if main_cfg.slam.enable_active_planning:
            if i == 0:
                ### update REFINE POOL ###
                planner.add_refine_pool_cand(slam.keyframe_list)

            ### get timer state ###
            planner_state = f"{planner.planning_state}_{planner.exploration_stage}" if planner.planning_state == "exploration" else planner.planning_state
            timer.start(planner_state, "General")

            c2w_slam_rel = planner.main(              #    这里调用了active_gs_planner.py的main函数,  input当前相对位姿, return 新的相对位姿
                c2w_slam_rel, 
                slam)
            c2w_slam = c2w_slam_init @ c2w_slam_rel     #  绝对位姿 = 初始位姿 * 相对位姿

            timer.end(planner_state)

            if planner.planning_state in ["refinement", "post_refinement"]:
                if force_map_update:
                    slam.config['mapping']['num_iters'] = main_cfg.slam.refine_map_iter
                else:
                    slam.config['mapping']['num_iters'] = map_iter_og
                    
            elif planner.planning_state == "done":
                break

            ### store data for visualization ###
            # if planner.state == "planning" and "exploration" in planner.planning_state:
            #     ### save params and render result ###
            #     # slam.print_and_save_result(f"step_{i:04}", ignore_first_frame=True)

            #     ### save information ###
            #     igs = []
            #     for key, val in planner.explore_pool.items():
            #         igs.append(val['ig'].detach().cpu().numpy())
            #     igs = np.asarray(igs)
            #     eval_dir_suffix = f"step_{i:04}"
            #     eval_dir = slam.eval_dir + "_" + eval_dir_suffix 
            #     os.makedirs(eval_dir, exist_ok=True)
            #     np.save(os.path.join(eval_dir, "information.npy"), igs)


    ##################################################
    ### Save Final Mesh and Checkpoint
    ##################################################
    slam.print_and_save_result("final", ignore_first_frame=True)

    ##################################################
    ### Runtime Analysis
    ##################################################
    timer.time_analysis(method='mean')
    print("per-iter SLAM_exploration_0: ", np.mean(timer.timers['SLAM_exploration_0']['duration'][4:][::5]))
    print("per-iter SLAM_exploration_1: ", np.mean(timer.timers['SLAM_exploration_1']['duration'][3:][::5]))
