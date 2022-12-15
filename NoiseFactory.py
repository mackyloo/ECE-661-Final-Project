import warnings

import Noise_Types as NT
class NoiseFactory(object):

    @staticmethod
    def get_shotNoise(intensity: int=0, p:float=0.5):
        noise = NT.ShotNoise(intensity=intensity, p=p)
        return noise.add_noise_to_img

    @staticmethod
    def get_saltNoise(intensity: int=0, p:float=0.5):
        noise = NT.SaltNoise(intensity=intensity, p=p)
        return noise.add_noise_to_img

    @staticmethod
    def get_scaleNoise(intensity: int=0, p:float=0.5):
        noise = NT.ScaleNoise(intensity=intensity, p=p)
        return noise.add_noise_to_img

    @staticmethod
    def get_motionBlurNoise(intensity: int=0, p:float=0.5):
        noise = NT.MotionBlurNoise(intensity=intensity, p=p)
        return noise.add_noise_to_img

    @staticmethod
    def get_coarsePepperNoise(intensity: int=0, p:float=0.5):
        noise = NT.CoarsePepperNoise(intensity=intensity, p=p)
        return noise.add_noise_to_img

    @staticmethod
    def get_noNoise(intensity:int=0, p:float=0.5):
        noise = NT.NoNoise(intensity=intensity, p=p)
        return noise.add_noise_to_img

    def get_noise_by_name(self, noise_type_name:str = None, intensity:int=0, p:float=0.5):
        if noise_type_name=="shot":
            return self.get_shotNoise(intensity=intensity, p=p)
        elif noise_type_name=="salt":
            return self.get_saltNoise(intensity=intensity, p=p)
        elif noise_type_name=="scale":
            return self.get_scaleNoise(intensity=intensity, p=p)
        elif noise_type_name=="motion-blur":
            return self.get_motionBlurNoise(intensity=intensity, p=p)
        elif noise_type_name=="coarse-pepper":
            return self.get_coarsePepperNoise(intensity=intensity, p=p)
        elif (noise_type_name != "standard") | (noise_type_name is not None):
            warnings.warn("NoiseFactory not supplied with an expected name, so noNoise supplied.")

        return self.get_noNoise()

