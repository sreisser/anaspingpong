from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
THRESHOLD = .99
MODEL_WEIGHTS_FILE = '../model/20220203_detectron/model_final.pth'
YAML = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'


class Predictor:
    def __init__(self, model_weights=MODEL_WEIGHTS_FILE, threshold=THRESHOLD):
        print("Initializing predictor")
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.merge_from_file(model_zoo.get_config_file(YAML))
        cfg.MODEL.WEIGHTS = model_weights
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.predictor = DefaultPredictor(cfg)
