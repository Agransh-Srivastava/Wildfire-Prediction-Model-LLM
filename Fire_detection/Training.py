from ultralytics import YOLO
# Load a model
model = YOLO('yolov8n.pt') # load an official model
PROJECT = 'wildfire' # project name
NAME = 'experiment_name' # run name

model.train(
  data = 'data.yaml',
  task = 'detect',
  epochs = 25,
  verbose = True,
  batch = 64,
  imgsz = 640,
  patience = 20,
  save = True,
  device = 0,
  workers = 8,
  project = PROJECT,
  name = NAME,
  cos_lr = True,
  lr0 = 0.0001,
  lrf = 0.00001,
  warmup_epochs = 3,
  warmup_bias_lr = 0.000001,
  optimizer = 'Adam',
  seed = 42,
)
