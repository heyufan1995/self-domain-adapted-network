from models.basemodel import AdaptorNet
from models.segmodel import SegANet
from models.synmodel import SynANet
import pdb
import logging
logger = logging.getLogger('global')
def create_model(args):
    """Create a model and load its weights if given
    """
    if args.task == 'syn':
        adaptorNet = SynANet
    elif args.task == 'seg':
        adaptorNet = SegANet
    else:
        raise NotImplementedError    
    model = adaptorNet(args)
    if args.resume_T:
        model.load_nets(args.resume_T, name='tnet')
    if args.resume_AE:
        model.load_nets(args.resume_AE, name='aenet')    
    if args.resume_LG:
        model.load_nets(args.resume_LG, name='lgan')           
    return model