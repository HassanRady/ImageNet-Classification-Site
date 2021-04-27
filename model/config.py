import pathlib
import model

PACKAGE_ROOT = pathlib.Path(model.__file__).resolve().parent
MODEL_TRAINED_DIR = PACKAGE_ROOT.parent / 'model_trained'