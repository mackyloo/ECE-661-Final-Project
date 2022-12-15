import imgaug.augmenters as iaa
import torch
import random

class NoiseType:

    def __init__(self, intensity: int = 0, p: float = 0.5):

        if intensity not in [0, 1, 2, 3, 4, 5]:
            raise ValueError("Noise intensity level is not one of the following: 0 (no-noise), 1, 2, 3, 4, 5 (maximum noise)")

        self.intensity = intensity
        self.p = p


    def apply_noise(self):
        '''
        This function allows noise to be randomly applied with probability = self.p
        :return: True if it should be applied, false otherwise
        '''
        rand_n = random.randint(0, 1000)/1000
        if rand_n <= self.p:
            return True

        return False

# unit8
class ShotNoise(NoiseType):
    def __init__(self, intensity: int = 0, p: float = 0.5):
        super().__init__(intensity=intensity, p=p)

        if intensity == 1:
            self.noise_multiple = 32 # 16/1024
        if intensity == 2:
            self.noise_multiple = 64 # 32/1024
        if intensity == 3:
            self.noise_multiple = 128  # 64/1024
        if intensity == 4:
            self.noise_multiple = 256   # 128/1024
        if intensity == 5:
            self.noise_multiple = 512  # 256/1024

    def add_noise_to_img(self, img):
        if self.apply_noise():
            img = img.permute(1,2,0)
            ones = torch.ones((32, 32, 3))
            for i in range(0, self.noise_multiple):
                x = random.randint(0,31)
                y = random.randint(0,31)
                ones[x,y,0]=0.0
                ones[x,y,1]=0.0
                ones[x,y,2]=0.0

            img = torch.multiply(img, ones)

            return img.permute(2,0,1)

        return img


class ScaleNoise(NoiseType):
    def __init__(self, intensity: int = 0, p: float = 0.5):
        super().__init__(intensity=intensity, p=p)
        if intensity == 1:
            self.intensity = 1.001
        elif intensity == 2:
            self.intensity = 1.5
        elif intensity == 3:
            self.intensity = 3
        elif intensity == 4:
            self.intensity = 6
        elif intensity == 5:
            self.intensity = 12

    def add_noise_to_img(self, img):
        if self.apply_noise():
            img = img*self.intensity

        return img

class SaltNoise(NoiseType):

    def __init__(self, intensity: int = 0, p: float = 0.5):
        super().__init__(intensity=intensity, p=p)

        # p = % of pixels replaced with a white-ish color
        if self.intensity == 1:
            self.aug = iaa.Salt(p=0.06)
        elif self.intensity == 2:
            self.aug = iaa.Salt(p=0.12)
        elif self.intensity == 3:
            self.aug = iaa.Salt(p=0.24)
        elif self.intensity == 4:
            self.aug = iaa.Salt(p=0.36)
        elif self.intensity == 5:
            self.aug = iaa.Salt(p=0.48)

    def add_noise_to_img(self, img):
        if self.apply_noise():
            if self.intensity >= 1:
                img = img.permute(1,2,0)
                img = img.numpy()
                img = self.aug.augment_image(image=img)
                img = torch.from_numpy(img)
                img = img.permute(2,0,1)
        return img

class NoNoise(NoiseType):
    def __init__(self, intensity:int=0, p: float = 0.5):
        super().__init__(intensity=intensity, p=p)

    def add_noise_to_img(self, img):
        return img

class MotionBlurNoise(SaltNoise):

    def __init__(self, intensity: int = 0, p: float = 0.5):
        super().__init__(intensity=intensity, p=p)

        # k the kernel size
        # angle the angle at which the blur is applied
        if self.intensity == 1:
            self.aug = iaa.blur.MotionBlur(k=3, angle = 90)
        elif self.intensity == 2:
            self.aug = iaa.blur.MotionBlur(k=5, angle = 90)
        elif self.intensity == 3:
            self.aug = iaa.blur.MotionBlur(k=7, angle = 90)
        elif self.intensity == 4:
            self.aug = iaa.blur.MotionBlur(k=9, angle = 90)
        elif self.intensity == 5:
            self.aug = iaa.blur.MotionBlur(k=11, angle = 90)

class CoarsePepperNoise(SaltNoise):
    def __init__(self, intensity: int = 0, p: float = 0.5):
        super().__init__(intensity=intensity, p=p)

        # p = Probability of changing a pixel to pepper noise.
        if self.intensity == 1:
            self.aug = iaa.arithmetic.CoarsePepper(p=0.06)
        elif self.intensity == 2:
            self.aug = iaa.arithmetic.CoarsePepper(p=0.12)
        elif self.intensity == 3:
            self.aug = iaa.arithmetic.CoarsePepper(p=0.24)
        elif self.intensity == 4:
            self.aug = iaa.arithmetic.CoarsePepper(p=0.36)
        elif self.intensity == 5:
            self.aug = iaa.arithmetic.CoarsePepper(p=0.48)


class JigsawNoise(SaltNoise):
    def __init__(self, intensity: int = 0, p: float = 0.5):
        super().__init__(intensity=intensity, p=p)

        # nb_iteration= The number of cutout blocks
        # nb_rows = How many rows the jigsaw pattern should have.
        if self.intensity >= 1:
            self.aug = iaa.geometric.Jigsaw(nb_rows=self.intensity*3, nb_cols=self.intensity*3)

