Execution Instructions: 

JUST RUN ALL CELLS IN THE 2 NOTEBOOKS AND IT SHOULD WORK

(provided the 2 data files are in the Data directory: 1) Data/train_data.pkl, 2) Data/test_data.pkl )

---------------------------------------------------------------------------------------------------------------------------------

Composition of Files

---------------------------------------------
Submission 1 -> Neural Network from Scratch:
---------------------------------------------

1. NN Scratch_Submission1.ipynb (jupyter notebook) -> just run the cells and it should work.

2. nn_utils.py (implements functions like standard saler and stratified split from scratch).
3. nn_layer.py (implements Linear Layer, Dropout, BatchNorm, backprop, forward pass, leaky_relu and other NN functions).

4. NeuralNet.py (uses nn_layers to build a fully connected Neural Network architecture consisting of multiple layers).
		(Can dynamically change architecture by adding more layers etc...)

5. nn_trainer.py (instantiates a NeuralNet and trains it for several epochs given parameters) -> used in main notebook.

---------------------------------------------
Submission 2 -> ResNet18 Ensemble:
---------------------------------------------

1. ResNet_MAIN_Submission_2.ipynb (jupyter notebook) -> just run the cells and it should work.

2. RetinaDataset.py (Subclasses Pytorch's Dataset and implements custom class for given data. Applies different transformations and augmentation techniques as well).

3. ResNetModel.py (Implements training loop for a ResNet Model. 2 Phases: 1) train last layer only, 2) train whole network).
