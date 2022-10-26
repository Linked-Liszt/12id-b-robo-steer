import robosteer12idb.models as models
import torch

def load_prod_model(model_fp: str) -> models.MLP:
    """
    Loads model state dict. Currently, 
    prod model is hard coded.

    TODO: Dynamic model parameter storage

    Params:
        model_fp: path to model checkpoint
    
    Returns:
        Initialized prod model

    """
    model = models.MLP(200, 256, 1, 2)
    model.load_state_dict(torch.load(model_fp))
    return model

def pred_usr_pos(reading, model_p) -> float:
    """
    Predicts user position from reading. 
    Reading must be interpolated before entry into
    this script. 

    Parms:
        reading: List or np array of size (200,) of the sensor values
                 Can use torch's batch style to process batches of (N, 200)

        model: model file path or model object itself
    
    Returns:
        floating point value of the predicted model output
    """
    if type(model_p) == str:
        model = load_prod_model(model_p)
    elif type(model_p) == model.MLP:
        model = model_p

    model.eval()
    with torch.no_grad():
        out = model(torch.Tensor(reading))
    
    if len(out) == 1:
        out = out.item()
    else:
        out = out.squeeze(1).numpy()
    
    return out
