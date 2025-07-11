import cv2
import numpy as np

from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import Config
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import OBAlignMode
from pyorbbecsdk import OBError
from pyorbbecsdk import VideoStreamProfile
from pyorbbecsdk import OBPoint2f, transformation2dto3d
from pyorbbecsdk import OBPoint3f, transformation3dto3d
from utils import frame_to_bgr_image, frame_to_rgb_image

class OrbbecColorViewer:
    def __init__(self, config, pipeline, calibrate=False) -> None:
        # init Orbbec camera color view
        self.config = config
        self.pipeline = pipeline
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            try:
                if calibrate:
                    color_profile: VideoStreamProfile = profile_list.get_default_video_stream_profile()
                else:
                    color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 30)
            except OBError as e:
                print(e)
                color_profile = profile_list.get_default_video_stream_profile()
            self.config.enable_stream(color_profile)
        except Exception as e:
            print(e)
            return

    def get_color_image(self):
        # return a color image, or None if failed
        frames: FrameSet = self.pipeline.wait_for_frames(100)
        if frames is None:
            print("get_color_image: no frames")
            return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            print("get_color_image: no color_frame")
            return None
        # covert to RGB format
        color_image = frame_to_bgr_image(color_frame)
        if color_image is None:
            print("get_color_image: failed to convert frame to rgb")
            return None
        return color_image


class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


ESC_KEY = 27
PRINT_INTERVAL = 1  # seconds
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

class OrbbecDepthViewer:
    def __init__(self, config, pipeline) -> None:
        self.config = config
        self.pipeline = pipeline
        self.temporal_filter = TemporalFilter(alpha=0.5)
        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            assert profile_list is not None
            try:
                depth_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.Y16, 30)
            except OBError as e:
                print("Error: ", e)
                depth_profile = profile_list.get_default_video_stream_profile()
            assert depth_profile is not None
            print("depth profile: ", type(depth_profile))
            self.config.enable_stream(depth_profile)
        except Exception as e:
            print(e)
            return

    def get_depth_data(self):
        try:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                return None
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                return None
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))

            depth_data = depth_data.astype(np.float32) * scale
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)
            # Apply temporal filtering
            depth_data = self.temporal_filter.process(depth_data)

            return depth_data
        except KeyboardInterrupt:
            return None

    def get_depth_image(self):
        try:
            depth_data = self.get_depth_data()

            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

            return depth_image
        except KeyboardInterrupt:
            return None



class OrbbecCamera:
    def __init__(self, color=True, depth=False, calibrate=False) -> None:
        # init Orbbec camera
        self.config = Config()
        self.pipeline = Pipeline()

        # if color viewer required
        if color:
            self.color_viewer = OrbbecColorViewer(self.config, self.pipeline, calibrate=calibrate)
        
        # if depth viewer required
        if depth:
            self.depth_viewer = OrbbecDepthViewer(self.config, self.pipeline)

        self.pipeline.start(self.config)

    def get_color_image(self):
        try:
            color_image = self.color_viewer.get_color_image()
        except:
            color_image = None
        finally:
            return color_image
        
    def get_depth_data(self):
        try:
            depth_data = self.depth_viewer.get_depth_data()
        except:
            depth_data = None
        finally:
            return depth_data

    def get_depth_image(self):
        try:
            depth_image = self.depth_viewer.get_depth_image()
        except:
            depth_image = None
        finally:
            return depth_image

    def run_with_callback(self, fn):
        while True:
            try:
                image_bgr = self.get_color_image() # openCV BGR format
                if image_bgr is None:
                    continue
                cv2.namedWindow("Color Image Viewer")
                cv2.imshow("Color Image Viewer", image_bgr)

                fn(image_bgr)
                if cv2.waitKey(1) in [ord('q'), ESC_KEY]:
                    break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)
                continue
        cv2.destroyAllWindows()


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_FPS = 10
IMAGE_WIDTH_CALIBRATION = 1920
IMAGE_HEIGHT_CALIBRATION = 1080

class OrbbecRGBDCamera:
    '''Depth2Color in HW mode & sync depth and color stream'''
    def __init__(self, eye_to_hand=False) -> None:
        # init Orbbec camera
        self.config = Config()
        self.pipeline = Pipeline()
        self.eye_to_hand = eye_to_hand
        # 3d transform parameters
        self.color_intrinsics = None
        self.color_distortion = None
        self.depth_intrinsics = None
        self.depth_distortion = None
        self.d2c_extrinsic = None
        # init config
        if self.eye_to_hand:
            self._config_with_eye_to_hand_mode()
        else:
            self._config_with_hw_d2c_mode()

    def _config_with_hw_d2c_mode(self):
        try:
            # Get the list of color stream profiles
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            assert profile_list is not None
            
            # Iterate through the color stream profiles
            for i in range(len(profile_list)):
                color_profile = profile_list[i]
                
                # Check if the color format is RGB
                if color_profile.get_format() != OBFormat.RGB:
                    continue
                if color_profile.get_width() != IMAGE_WIDTH or color_profile.get_height() != IMAGE_HEIGHT or color_profile.get_fps() > IMAGE_FPS:
                    continue
                
                # Get the list of hardware aligned depth-to-color profiles
                hw_d2c_profile_list = self.pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
                if len(hw_d2c_profile_list) == 0:
                    continue
                # Get the hardware aligned depth-to-color profile
                for j in range(len(hw_d2c_profile_list)):
                    hw_d2c_profile = hw_d2c_profile_list[j]
                    if hw_d2c_profile.get_width() == IMAGE_WIDTH and hw_d2c_profile.get_height() == IMAGE_HEIGHT and hw_d2c_profile.get_fps() <= IMAGE_FPS:
                        print("==> color_profile: ", color_profile.get_format(), color_profile)
                        print("==> hw_d2c_profile: ", hw_d2c_profile.get_format(), hw_d2c_profile)
                        # Enable the depth and color streams
                        self.config.enable_stream(color_profile)
                        self.config.enable_stream(hw_d2c_profile)
                        # Set the alignment mode to hardware alignment
                        self.config.set_align_mode(OBAlignMode.HW_MODE)
                        # 3d转换参数
                        self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsic()
                        self.color_distortion = color_profile.as_video_stream_profile().get_distortion()
                        self.depth_intrinsics = hw_d2c_profile.as_video_stream_profile().get_intrinsic()
                        self.depth_distortion = hw_d2c_profile.as_video_stream_profile().get_distortion()
                        self.d2c_extrinsic = hw_d2c_profile.get_extrinsic_to(color_profile)
                        return
                    else:
                        continue
        except Exception as e:
            print("_config_with_hw_d2c_mode error: ", e)
            raise e
        raise Exception("no matching hw_d2c_profile")

    def _config_with_eye_to_hand_mode(self):
        try:
            # Get the list of color stream profiles
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            assert profile_list is not None

            # Iterate through the color stream profiles
            for i in range(len(profile_list)):
                color_profile = profile_list[i]

                # Check if the color format is RGB
                if color_profile.get_format() != OBFormat.RGB:
                    continue
                if color_profile.get_width() != IMAGE_WIDTH_CALIBRATION or color_profile.get_height() != IMAGE_HEIGHT_CALIBRATION or color_profile.get_fps() > IMAGE_FPS:
                    continue

                print("==> color_profile: ", color_profile.get_format(), color_profile)
                # Enable the depth and color streams
                self.config.enable_stream(color_profile)
                # 相机参数
                self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsic()
                self.color_distortion = color_profile.as_video_stream_profile().get_distortion()
                return
        except Exception as e:
            print("_config_with_eye_to_hand_mode error: ", e)
            raise e
        raise Exception("no matching color_profile")

    def start(self):
        # sync the depth and color frames
        if not self.eye_to_hand:
            self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)

    def stop(self):
        self.pipeline.stop()

    def get_rgbd_data(self):
        # Wait for frames
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            raise Exception("No frames received")
        
        # Get the color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise Exception("No data received")
        if depth_frame.get_format() != OBFormat.Y16:
            raise Exception("depth format is not Y16")

        color_image = frame_to_rgb_image(color_frame) # orbbec data frame change to image
        # Get the depth data
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            (depth_frame.get_height(), depth_frame.get_width()))
        # Convert depth data to float32 and apply depth scale
        depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
        # Apply custom depth range, clip depth data
        # depth_data = np.clip(depth_data, MIN_DEPTH, MAX_DEPTH)
        depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)

        return (color_image, depth_data)

    def get_color_image(self):
        # Wait for frames
        frames = self.pipeline.wait_for_frames(100)
        if frames is None:
            raise Exception("No frames received")

        # Get the color image
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise Exception("No data received")
        color_image = frame_to_rgb_image(color_frame) # orbbec data frame change to image

        return color_image

    def transform_points(self, x, y, depth):
        '''
        refrence: 
            https://github.com/orbbec/OrbbecSDK_v2/blob/main/examples/3.advanced.coordinate_transform/coordinate_transform.cpp#L163
            https://orbbec.github.io/docs/OrbbecSDKv2/classob_1_1_coordinate_transform_helper.html
            https://github.com/orbbec/pyorbbecsdk/blob/v2-main/examples/coordinate_transform.py
        '''
        x, y = float(x), float(y) # width/column/x, height/row/y
        original_point = (x, y, depth)
        res = transformation2dto3d(OBPoint2f(x, y), depth, self.depth_intrinsics, self.d2c_extrinsic)
        # res = transformation3dto3d(OBPoint3f(x, y, depth), self.d2c_extrinsic)

        print(f"--------------------------------------------")
        print(f"Original point: {original_point}")
        print(f"Transformed point: {res}")
        print(f"--------------------------------------------")
        return res

    def run(self):
        self.start()
        while True:
            try:
                color_image, depth_data = self.get_rgbd_data()
                if color_image is None:
                    continue

                # Normalize depth data for display
                depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
                depth_image = cv2.applyColorMap(depth_image.astype(np.uint8), cv2.COLORMAP_JET)

                # Convert the image to openCV BGR format
                image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                # Blend the depth and color images
                blended_image = cv2.addWeighted(image_bgr, 0.5, depth_image, 0.5, 0)

                #resize the window
                cv2.namedWindow("HW D2C Align Viewer", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("HW D2C Align Viewer", 640, 480)
                
                # Display the result
                cv2.imshow("HW D2C Align Viewer", blended_image)
                if cv2.waitKey(1) in [ord('q'), ESC_KEY]:
                    break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)
                continue
        cv2.destroyAllWindows()

        # Stop the pipeline
        self.stop()

    def run_with_callback(self, fn):
        self.start()
        while True:
            try:
                color_image = self.get_color_image()
                if color_image is None:
                    continue
                # Convert the image to openCV BGR format
                image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                #resize the window
                cv2.namedWindow("Color Image Viewer")
                # Display the result
                cv2.imshow("Color Image Viewer", image_bgr)

                fn(image_bgr)
                if cv2.waitKey(1) in [ord('q'), ESC_KEY]:
                    break
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)
                continue
        cv2.destroyAllWindows()
        # Stop the pipeline
        self.stop()

if __name__ == "__main__":
    task = OrbbecRGBDCamera()
    task.run()
