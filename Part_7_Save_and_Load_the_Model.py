import torch
import torchvision.models as models

# Saving and Loading Model Weights

# PyTorch models store teh learned parameters in an internal state dictionar, called state_dict.
# These can be presisted with the torch.save method

model = models.vgg16(weights = 'IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

# To load model weights, you need to create an instance of the same model first, and then load the parameters using load_state_dict() method
# weights_only = True limits the functions executed during unpickling to only those necessary for loading weights. Considered best practice when loading weights.

model = models.vgg16() # We do not specify ''weights'', i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only = True))
model.eval()

# Note : Be sure to call model.eval() efore inferencing to set the dropout nd batch normalization layers to evaluation mode.
#        Failing to do this will yield inconsistent inference results.

# Saving and Loading Models with Shapes

# We need to instantiate the model class first, because the class defines the structure of a network.
# We might want to save the structure of this class together with the model, in which case we can pass model (and not model.state_dict()) to the saving function.

torch.save(model, 'model.pth')

# Saving state_dict is considered the best practice. However, below we use weights_only = False because this involves
# loading the model, which is a legacy use case for torch.save.

model = torch.load('model.pth', weights_only = False)

# Note : This approach uses Python pickle module when serializing the model, thus it relies on the actual class definition
#        to be available when loading the model.