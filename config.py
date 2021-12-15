# Path setting
DATASET_PATH = "dataset"
CORPUS_PATH = "corpus.bin"
WEIGHTS_PATH = 'weights.hdf5'
INPUTS_PATH = "inputs"
OUTPUTS_PATH = "outputs"

# 'loader.py'
EXTENSION = ['.musicxml', '.xml', '.mxl', '.midi', '.mid']

# 'train_model.py'
VAL_RATIO = 0.1
RNN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 512
EPOCHS = 10

# 'harmonizor.py'
TEMPERATURE = 0
RHYTHM_DENSITY = 0
LEAD_SHEET = True