# Visualizing change in a neural network (WIP)
In this project the change in a deep neural network as it is fine-tuned towards a new task is assessed using feature visualization. For a detailed description of this project, see this [blog post](add link here once the article is published).

There are two main parts to this repo:
- **feature_vis**: a python package which is a partial pytorch implementation of the tensorflow feature visualization library [lucid](https://github.com/tensorflow/lucid).
- **transfer**: a set of python scripts which perform the fine-tuning task, generate the feature visualizations using `feature_vis`, and compares the change in the resulting visualizations.

## feature_vis

To install this package locally, simply run `pip install .` from the root level of this project.
Once installed, you can generate a feature visualization using the following code:
```python
from feature_vis.render import FeatureVisualizer
from torchvision.models import resnet50

# load a pre-trained model
model = resnet50(pretrained=True)
model.eval()

# set up the visualizer
visualize = FeatureVisualizer(model)

# create the feature visualization
img = visualize(act_idx=309)
```
The snippet above produces the feature visualization of the 
