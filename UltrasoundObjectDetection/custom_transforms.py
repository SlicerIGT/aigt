# expands single channel grayscale image to 3 channel grayscale image. Expects torch tensor as input (call it after ToTensor in transform pipeline)
class Grayscale1Channelto3Channels():
    def __init__(self, num_channels=3):
        self.num_channels = num_channels
    
    def __call__(self, img):
        if img.shape[0] == 1:
            img = img.expand(self.num_channels,*img.shape[1:])
        return img
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"