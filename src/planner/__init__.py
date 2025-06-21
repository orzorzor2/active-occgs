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


import mmengine

from src.utils.general_utils import InfoPrinter


def init_planner(main_cfg: mmengine.Config, info_printer: InfoPrinter):
    """initialize planner

    Args:
        cfg (mmengine.Config)     : Configuration
        info_printer (InfoPrinter): information printer

    Returns:
        planner (Planner): Planner module

    """
    ##################################################
    ### NarutoPlanner
    ##################################################
    if main_cfg.planner.method == "naruto":
        info_printer("Initialize NARUTO Planner...", 0, "NarutoPlanner")
        from src.planner.naruto_planner import NarutoPlanner
        planner = NarutoPlanner(
            main_cfg,
            info_printer
            ) 
    elif main_cfg.planner.method == "predefined_traj":
        info_printer("Initialize PredefinedTraj Planner...", 0, "PredefinedPlanner")
        from src.planner.predefined_traj_planner import PreTrajPlanner
        planner = PreTrajPlanner(
            main_cfg,
            info_printer
            )
    elif main_cfg.planner.method == "active_lang":
        info_printer("Initialize ActiveLang Planner...", 0, "ActiveLangPlanner")
        from src.planner.active_lang_planner import ActiveLangPlanner
        planner = ActiveLangPlanner(
            main_cfg,
            info_printer
            ) 
    elif main_cfg.planner.method == "active_gs":
        info_printer("Initialize ActiveGS Planner...", 0, "ActiveGSPlanner")
        from src.planner.active_gs_planner import ActiveGSPlanner
        planner = ActiveGSPlanner(
            main_cfg,
            info_printer
            ) 

    else:
        assert False, f"Planner choices: [naruto, predefined_traj, active_lang, active_gs]. Current option: [{main_cfg.planner.method}]"
    return planner