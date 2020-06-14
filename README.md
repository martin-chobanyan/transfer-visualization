# Visualizing change in a neural network (WIP)
In this project the change in a deep neural network as it is fine-tuned towards a new task is assessed using feature visualization. For a detailed description of this project, see this [blog post](add link here once the article is published).

There are two main parts to this repo:
- **feature_vis**: a python package which is a partial pytorch implementation of the tensorflow feature visualization library [lucid](https://github.com/tensorflow/lucid).
- **transfer**: a set of python scripts which perform the fine-tuning task, generate the feature visualizations using `feature_vis`, and compares the change in the resulting visualizations.

## feature_vis

To install this package locally, simply run the following from the root level of this project:
```
pip install .
```
Once installed, you can generate a feature visualization using the following code:
```python
from feature_vis.render import FeatureVisualizer
from torchvision.models import resnet50

# load a pre-trained model
model = resnet50(pretrained=True)
model.eval()

# set up the visualizer
visualize = FeatureVisualizer(model)

# create the feature visualization as a PIL image
img = visualize(act_idx=309)
```
The snippet above produces the feature visualization for the activation of the 309th index in resnet50's output vector. This corresponds to the "bee" class in ImageNet and the resulting feature visualization can be seen below:  
![Bee feature visualization](https://raw.githubusercontent.com/martin-chobanyan/transfer-visualization/master/resources/bee-visualization.png)

The easiest way to specify the target activation is to alter the model such that it outputs the target layer and then specify the index within the resulting activation vector or volume. Note, if the layer outputs a volume then the target activation is the average of the i-th feature map in the volume. Pytorch hooks can also be used isolate the target activation. See the docstrings in the code and the example notebooks for more details.  

Note, this is only a partial implementation of feature visualization as described in this [distill article](https://distill.pub/2017/feature-visualization/). Only the parts necessary for this project were converted to pytorch from lucid library.

## transfer
...
