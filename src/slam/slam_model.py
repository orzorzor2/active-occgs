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
from tensorboardX import SummaryWriter

from src.utils.general_utils import InfoPrinter


class SlamModel():
    def __init__(self, 
                 main_cfg    : mmengine.Config,
                 info_printer: InfoPrinter,
                 tb_logger   : SummaryWriter = None
                 ) -> None:
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer
            tb_logger (SummaryWriter): Tensorboard Writer
    
        Attributes:
            main_cfg (mmengine.Config): configurations
            slam_cfg (mmengine.Config): slam model configurations
            tb_logger (SummaryWriter): Tensorboard Writer
            
        """
        self.main_cfg = main_cfg
        self.slam_cfg = main_cfg.slam
        self.info_printer = info_printer
        if tb_logger is not None:
            self.tb_logger = tb_logger

    def update_step(self, step):
        """ update step information
    
        Args:
            step (int): step size
    
        """
        self.step = step

    def save_mesh(self):
        """
    
        Args:
            
    
        Returns:
            
    
        Attributes:
            
        """
        raise NotImplementedError