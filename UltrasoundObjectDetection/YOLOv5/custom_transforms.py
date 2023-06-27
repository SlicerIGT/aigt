class ExpandChannels():
    def __init__(self, num_channels=3):
        self.num_channels = num_channels
    
    def __call__(self, img):
        img = img.expand(self.num_channels,*img.shape[1:])
        return img
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"