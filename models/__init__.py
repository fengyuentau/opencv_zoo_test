from .face_detection_yunet.yunet import YuNet
from .text_detection_east.east import EAST
# from .text_recognition.crnn import CRNN

class Registery:
    def __init__(self, name):
        self._name = name
        self._dict = dict()

    def get(self, key):
        return self._dict[key]

    def register(self, item):
        self._dict[item.__name__] = item

MODELS = Registery('Models')
MODELS.register(YuNet)
MODELS.register(EAST)