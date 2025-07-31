# ACIE

exp 1 Comparison with other methods ： yolov11/lop/incremental_cifar/incremental_cifar_experiment.py

exp 2 Yolov11 task incremental learning and exp4 Comparison of the four strategies of ACIE and exp5 Parameter Study：yolov11/exp_cifar100.py

exp 3 Combined with other methods 
replay method: yolov11/resnet18_replay_exp/resnet18_cifar100.py
other methods: pycil

how to use ACIE:

```
# initialize NeuronLifecycleManager
model = ResNet50(num_classes=100)
manager = NeuronLifecycleManager(
    model,
    plasticity_factor=1,   
    protection_factor=1,    
    reset_threshold=0.3,  # parameter reset_threshold
    protect_threshold=0.7, # parameter protect_threshold 
    plasticity_decay=1,
    update_interval=1
)
for task in tasklist:
  # use ACIE before a new task
  if task not first task:
    manager.step()
  # train
  for epoch in range(100):
      model.train()
      
  # eval
  model.eval()



```
