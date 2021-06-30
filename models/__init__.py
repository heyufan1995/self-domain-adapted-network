from models.basemodel import AdaptorNet
from models.segmodel import SegANet
from models.synmodel import SynANet
import pdb
import logging
logger = logging.getLogger('global')
def create_model(args):
    """Create a model and load its weights if given
    """
    if 'syn' in args.task:
        adaptorNet = SynANet
    elif 'seg' in args.task:
        adaptorNet = SegANet
    else:
        raise NotImplementedError    
    model = adaptorNet(args)
    if args.resume_T:
        model.load_nets(args.resume_T, name='tnet')
    if args.resume_AE:
        model.load_nets(args.resume_AE, name='aenet')           
    return model