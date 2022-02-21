import torch
import argparse
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from nerf.utils import *


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=30):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([0, 0, 0, 1]) # scalar last
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        res = np.eye(3, dtype=np.float32)
        focal = self.H / (2 * np.tan(self.fovy / 2))
        res[0, 0] = res[1, 1] = focal
        res[0, 2] = self.W // 2
        res[1, 2] = self.H // 2
        return res
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

        # wrong: rotate along global x/y axis
        #self.rot = R.from_euler('xy', [-dy * 0.1, -dx * 0.1], degrees=True) * self.rot
    
    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.001 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])

        # wrong: pan in global coordinate system
        #self.center += 0.001 * np.array([-dx, -dy, dz])
    


class NeRFGUI:
    def __init__(self, opt, trainer, debug=True):
        dpg.create_context()
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius)
        self.trainer = trainer
        self.debug = debug
        self.register_dpg()
        self.update_texture()
        self.need_update = False
        

    def __del__(self):
        dpg.destroy_context()

    
    def update_texture(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?
        # TODO: dynamic rendering resolution to keep it fluent.
        
        if self.debug:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

        outputs = self.trainer.infer(self.cam.pose, self.cam.intrinsics, self.W, self.H)

        if self.debug:
            ender.record()
            torch.cuda.synchronize()
            dpg.set_value("_log_time", str(starter.elapsed_time(ender)))

        dpg.set_value("_texture", outputs['image'])

        
    def register_dpg(self):

        ### register texture 

        raw_data = np.zeros((self.W, self.H, 3)) # value should be in [0, 1]

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, raw_data, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        if self.debug:
            with dpg.window(label="Debug", tag="_debug_window", width=500, height=150):
                with dpg.group(horizontal=True):
                    dpg.add_text("Infer time: ")
                    dpg.add_text("nan", tag="_log_time")
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.cam.pose), tag="_log_pose")


        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

        
        dpg.create_viewport(title='Renderer', width=self.W, height=self.H, resizable=False)
        
        # TODO: seems dearpygui doesn't support resizing texture...
        # def callback_resize(sender, app_data):
        #     self.W = app_data[0]
        #     self.H = app_data[1]
        #     # how to reload texture ???

        # dpg.set_viewport_resize_callback(callback_resize)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):

        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.need_update:
                self.update_texture()
                self.need_update = False
            dpg.render_dearpygui_frame()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--W', type=int, default=800)
    parser.add_argument('--H', type=int, default=800)
    parser.add_argument('--radius', type=float, default=2, help="assume the camera is located around sphere(0, radius))")
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box(-bound, bound)")

    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")

    opt = parser.parse_args()

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork    

    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid", encoding_dir="sphere_harmonics", 
        num_layers=2, hidden_dim=64, geo_feat_dim=15, num_layers_color=3, hidden_dim_color=64, 
        cuda_ray=opt.cuda_ray,
    )        

    trainer = Trainer('ngp', vars(opt), model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='latest')

    gui = NeRFGUI(opt, trainer)
    gui.render()