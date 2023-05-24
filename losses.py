import lpips 
import sys

def get_loss(args):
    if args.loss == 'lpips':
        return lpips.LPIPS(net='alex') 
    else:
        sys.exit('Not a valid loss')

