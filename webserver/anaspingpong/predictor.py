from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
THRESHOLD = .9
MODEL_WEIGHTS_FILE = '../model/20121512_detectron/model_final.pth'

class Predictor:
    def __init__(self, model_weights=MODEL_WEIGHTS_FILE,
                 threshold=THRESHOLD):
        print('Initializing predictor')
        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cpu'
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.WEIGHTS = model_weights
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.predictor = DefaultPredictor(cfg)
