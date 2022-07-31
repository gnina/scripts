#!/usr/bin/env python

'''Take a huge number of options and create a model file.  Some solver hyperparameters are included as well.
All models perform pose prediction and affinity prediction.
'''

import sys
import argparse

class Range():
    '''represent a continuous numerical range'''
    def __init__(self,lo,hi):
        self.min = lo
        self.max = hi
        
    def __contains__(self, v): #in operator
        return v >= self.min and v <= self.max
        
    def __str__(self): #str
        return '(%f : %f)' % (self.min,self.max)

    def __repr__(self):
        return '(%f : %f)' % (self.min,self.max)

    def __iter__(self):
        return [self.min,self.max].__iter__()
    
parser = argparse.ArgumentParser(description='Create model from parameters.')
#solver hyper parameters
parser.add_argument('--base_lr_exp',type=float,help='Initial learning rate exponent, for log10 scaling',default=-2,choices=Range(-5,0),metavar='lr')
parser.add_argument('--momentum',type=float,help="Momentum parameters, default 0.9",default=0.9,choices=Range(0,1),metavar='m')
parser.add_argument('--weight_decay_exp',type=float,help="Weight decay exponent (for log10 scaling)",default=-3,choices=Range(-10,0),metavar='w')
parser.add_argument('--solver',type=str,help="Solver type",default='SGD',choices=('SGD','Adam'))

#next two are for inv solver - plan to move to step?
#parser.add_argument('--gamma_exp',type=float,help="Gamma exponent (log10 scaling)",default=-3,choices=Range(-10,0),metavar='g')
#parser.add_argument('--power',type=float,help="Power, default 1",default=1,choices=Range(0,2),metavar='p')


# training parameters
parser.add_argument('--balanced',type=int,help="Balance training data",default=1,choices=(0,1))
parser.add_argument('--stratify_receptor',type=int,help="Stratify receptor",default=1,choices=(0,1))
parser.add_argument('--stratify_affinity',type=int,help="Stratify affinity, min=2,max=10",default=0,choices=(0,1))
parser.add_argument('--stratify_affinity_step',type=float,help="Stratify affinity step",default=1,choices=(1,2,4))
parser.add_argument('--resolution',type=float,help="Grid resolution",default=0.5,choices=(0.5,1.0)) #need smal
parser.add_argument('--jitter',type=float,help="Amount of jitter to apply",default=0.0,choices=Range(0,1),metavar='j') 
parser.add_argument('--ligmap',type=str,help="Ligand atom typing map to use",default='') 
parser.add_argument('--recmap',type=str,help="Receptor atom typing map to use",default='') 

# loss parameters
parser.add_argument('--loss_gap',type=float,help="Affinity loss gap",default=0,choices=Range(0,5),metavar='g')
parser.add_argument('--loss_penalty',type=float,help="Affinity loss penalty",default=0,choices=Range(0,5),metavar='p')
parser.add_argument('--loss_pseudohuber',type=int,help="Use pseudohuber loss",default=1,choices=(0,1))
parser.add_argument('--loss_delta',type=float,help="Affinity loss delta",default=4,choices=Range(0,8),metavar='d')
parser.add_argument('--ranklossmult',type=float,help="Affinity rank loss multiplier",default=0,choices=Range(0,1),metavar='r')
parser.add_argument('--ranklossneg',type=int,help="Affinity rank loss include neg",default=0,choices=(0,1))


#the model is N convolutional layers (each independently configured, N <= 5) followed by 1 or 2 fully connected

def add_conv_args(n,defaultwidth):
    '''add arguments for layer n, in the form conv<n>_<arg>
    A conv unit consists of a pool layer, a convolutional layer, 
    a normalization layer, and a non-linear layer
    The pool layer is defined by kernel size
    The conv layer is defined by kernel size, stride, initialization, and output width
    Note that a width of zero indicates the lack of the entire unit
    '''
    
    parser.add_argument('--pool%d_size'%n,type=int,help="Pooling size for layer %d"%n,default=2 if defaultwidth > 0 else 0,choices=(0,2,4,8))
    parser.add_argument('--pool%d_type'%n,type=str,help="Pooling type for layer %d"%n,default='MAX',choices=('MAX','AVE'))
    parser.add_argument('--conv%d_func'%n,type=str,help="Activation function in layer %d"%n,default='ReLU',choices=('ReLU','leaky','ELU','Sigmoid','TanH'))
    parser.add_argument('--conv%d_norm'%n,type=str,help="Normalization for layer %d"%n,default='none',choices=('BatchNorm','LRN','none'))
    parser.add_argument('--conv%d_size'%n,type=int,help="Convolutional kernel size for layer %d"%n,default=3,choices=(1,3,5,7))
    parser.add_argument('--conv%d_stride'%n,type=int,help="Convolutional stride for layer %d"%n,default=1,choices=(1,2,3,4))
    parser.add_argument('--conv%d_width'%n,type=int,help="Convolutional output width for layer %d"%n,default=defaultwidth,choices=(0,1,2,4,8,16,32,64,128,256,512,1024))
    parser.add_argument('--conv%d_init'%n,type=str,help="Weight initialization for layer %d"%n,default='xavier',choices=('gaussian','positive_unitball','uniform','xavier','msra','radial','radial.5'))


add_conv_args(1,32)
add_conv_args(2,64)
add_conv_args(3,128)
add_conv_args(4,0)
add_conv_args(5,0)

#fully connected layer
parser.add_argument('--fc_affinity_hidden',type=int,help='Hidden nodes in affinity fully connected layer; 0 for single layer',default=0,choices=(0,16,32,64,128,256,512,1024,2048,4096))
parser.add_argument('--fc_affinity_func',type=str,help="Activation function in for first affinity hidden layer",default='ReLU',choices=('ReLU','leaky','ELU','Sigmoid','TanH'))
parser.add_argument('--fc_affinity_hidden2',type=int,help='Second set of hidden nodes in affinity fully connected layer; 0 for single layer',default=0,choices=(0,32,64,128,256,512,1024,2048,4096))
parser.add_argument('--fc_affinity_func2',type=str,help="Activation function in for second affinity hidden layer",default='ReLU',choices=('ReLU','leaky','ELU','Sigmoid','TanH'))
parser.add_argument('--fc_affinity_init',type=str,help="Weight initialization for affinity fc",default='xavier',choices=('gaussian','positive_unitball','uniform','xavier','msra'))

parser.add_argument('--fc_pose_hidden',type=int,help='Hidden nodes in pose fully connected layer; 0 for single layer',default=0,choices=(0,32,64,128,256,512,1024,2048,4096))
parser.add_argument('--fc_pose_func',type=str,help="Activation function in for first pose hidden layer",default='ReLU',choices=('ReLU','leaky','ELU','Sigmoid','TanH'))
parser.add_argument('--fc_pose_hidden2',type=int,help='Second set of hidden nodes in pose fully connected layer; 0 for single layer',default=0,choices=(0,32,64,128,256,512,1024,2048,4096))
parser.add_argument('--fc_pose_func2',type=str,help="Activation function in for second pose hidden layer",default='ReLU',choices=('ReLU','leaky','ELU','Sigmoid','TanH'))
parser.add_argument('--fc_pose_init',type=str,help="Weight initialization for pose fc",default='xavier',choices=('gaussian','positive_unitball','uniform','xavier','msra'))


nonparms = parser.add_argument_group('non-parameter options')

def getoptions():
    '''return options that have choices'''
    ret = dict()
    for a in parser._actions:
        if type(a) == argparse._StoreAction:
            if a.type == bool:
                ret[a.dest] = (0,1)
            elif a.choices:
                ret[a.dest] = a.choices
    return ret

def getdefaults():
    '''return defaults for arguments with choices'''
    ret = dict()
    for a in parser._actions:
        if type(a) == argparse._StoreAction and a.choices:
            ret[a.dest] = a.default
    return ret
    
    
def boolstr(val):
    '''return proto boolean string'''
    return 'true' if val else 'false'
    
    
poollayer = '''
layer {{
  name: "unit{i}_pool"
  type: "Pooling"
  bottom: "{lastlayer}"
  top: "unit{i}_pool"
  pooling_param {{
    pool: {pooltype}
    kernel_size: {size}
    stride: {size}
  }}
}}'''

convolutionlayer = '''
layer {{
  name: "{0}"
  type: "Convolution"
  bottom: "{1}"
  top: "{0}"
  convolution_param {{
    num_output: {2}
    pad: {3}
    kernel_size: {4}
    stride: {5}
    weight_filler {{
      type: "xavier"
      symmetric_fraction: 1.0      
    }}
  }}
}}'''

normlayer = '''
layer {{
  name: "unit{1}_norm"
  type: "{2}"
  bottom: "{0}"
  top: "{0}n"
}}
layer {{
 name: "unit{1}_scale"
 type: "Scale"
 bottom: "{0}n"
 top: "{0}n"
 scale_param {{
  bias_term: true
 }}
}}
'''
    
def funclayer(inputname, layername, func):
    '''return activation unit layer'''
    extra = ''
    if func == 'leaky':
        func = 'ReLU'
        extra = 'relu_param{ negative_slope: 0.01}'
    return '''
layer {{
  name: "{1}"
  type: "{2}"
  bottom: "{0}"
  top: "{0}"
  {3}
}}
'''.format(inputname, layername, func,extra)

innerproductlayer = '''
layer {{
  name: "{1}"
  type: "InnerProduct"
  bottom: "{0}"
  top: "{1}"
  inner_product_param {{
    num_output: {3}
    weight_filler {{
      type: "{2}"
    }}
  }}
}}
'''


def create_model(args):
    '''Generate the model defined by args; first line (comment) contains solver/train.py arguments'''
    m = '# --base_lr %f --momentum %f --weight_decay %f -- solver %s\n' % (10**args.base_lr_exp,args.momentum,10**args.weight_decay_exp,args.solver)
    #test input
    m += '''
layer {
  name: "data"
  type: "MolGridData"
  top: "data"
  top: "label"
  top: "affinity"
  include {
    phase: TEST
  }
  molgrid_data_param {
        source: "TESTFILE"
        batch_size: 50
        dimension: 23.5
        resolution: %f
        shuffle: false
        ligmap: "%s"
        recmap: "%s"
        balanced: false
        has_affinity: true
        root_folder: "DATA_ROOT"
    }
  }
  ''' % (args.resolution,args.ligmap,args.recmap)
  
  #train input,
    m += '''
layer {
  name: "data"
  type: "MolGridData"
  top: "data"
  top: "label"
  top: "affinity"
  include {
    phase: TRAIN
  }
  molgrid_data_param {
        source: "TRAINFILE"
        batch_size:  50
        dimension: 23.5
        resolution: %f
        shuffle: true
        balanced: %s
        jitter: %f
        ligmap: "%s"
        recmap: "%s"        
        stratify_receptor: %s
        stratify_affinity_min: %d
        stratify_affinity_max: %d
        stratify_affinity_step: %f
        has_affinity: true
        random_rotation: true
        random_translate: 6
        root_folder: "DATA_ROOT"
    }
}
''' % (args.resolution, boolstr(args.balanced), args.jitter, args.ligmap, args.recmap, boolstr(args.stratify_receptor), 2 if args.stratify_affinity else 0, 10 if args.stratify_affinity else 0, args.stratify_affinity_step)

    #now pool/conv layers
        
    lastlayer = 'data'
    zeroseen = False
    vargs = vars(args)
    for i in range(1,6):
        poolsize = vargs['pool%d_size'%i]
        pooltype = vargs['pool%d_type'%i]
        if poolsize > 0:
            m += poollayer.format(i=i,lastlayer=lastlayer,size=poolsize,pooltype=pooltype)
            lastlayer = 'unit{0}_pool'.format(i)
            
        convwidth = vargs['conv%d_width'%i]
        
        if convwidth == 0:
            zeroseen = True
        elif convwidth > 0:
            if zeroseen:
                print("Invalid convolutional widths - non-zero layer after zero layer")
                sys.exit(-1)
            convlayer = 'unit%d_conv'%i
            ksize = vargs['conv%d_size'%i]
            stride = vargs['conv%d_stride'%i]
            func = vargs['conv%d_func'%i]
            norm = vargs['conv%d_norm'%i]
            init = vargs['conv%d_init'%i]
            pad = int(ksize/2)
            m += convolutionlayer.format(convlayer,lastlayer, convwidth, pad, ksize,stride)
            if norm != 'none':
                m += normlayer.format(convlayer, i, norm)
                convlayer += 'n'
            
            m += funclayer(convlayer, 'unit%d_func'%i, func)
            lastlayer = convlayer
        
        
        
    m += '''
layer {{
    name: "split"
    type: "Split"
    bottom: "{0}"
    top: "split"
}}
'''.format(lastlayer)
    #fully connected
    # pose
    
    if args.fc_pose_hidden == 0 and args.fc_pose_hidden2 != 0:
        print("Invalid pose hidden units")
        sys.exit(1)
        
    fcinfo = [] #tuples of name,numoutput
    if args.fc_pose_hidden > 0:
        fcinfo.append(('pose_fc', args.fc_pose_hidden, args.fc_pose_func))
        if args.fc_pose_hidden2 > 0:
            fcinfo.append(('pose_fc2', args.fc_pose_hidden2, args.fc_pose_func2))
    fcinfo.append(('pose_output',2, None))
            
    lastlayer = 'split'
    for (name,num,func) in fcinfo:        
        m += innerproductlayer.format(lastlayer, name, args.fc_pose_init, num)
        if func:
            m += funclayer(name,name+'_func',func)
        lastlayer = name
        
    # affinity
    
    if args.fc_affinity_hidden == 0 and args.fc_affinity_hidden2 != 0:
        print("Invalid affinity hidden units")
        sys.exit(1)
        
    fcinfo = [] #tuples of name,numoutput
    if args.fc_affinity_hidden > 0:
        fcinfo.append(('affinity_fc', args.fc_affinity_hidden, args.fc_affinity_func))
        if args.fc_affinity_hidden2 > 0:
            fcinfo.append(('affinity_fc2', args.fc_affinity_hidden2, args.fc_affinity_func2))
    fcinfo.append(('affinity_output',1,None))
            
    lastlayer = 'split'
    for (name,num,func) in fcinfo:        
        m += innerproductlayer.format(lastlayer, name, args.fc_affinity_init, num)
        lastlayer = name        
        if func:
            m += funclayer(name,name+'_func',func)
    
    #loss
    #pose
    m += '''layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "pose_output"
  bottom: "label"
  top: "loss"
}

layer {
  name: "output"
  type: "Softmax"
  bottom: "pose_output"
  top: "output"
}
layer {
  name: "labelout"
  type: "Split"
  bottom: "label"
  top: "labelout"
  include {
    phase: TEST
  }
}
'''
    #affinity loss
    m += '''layer {{
  name: "rmsd"
  type: "AffinityLoss"
  bottom: "affinity_output"
  bottom: "affinity"
  top: "rmsd"
  affinity_loss_param {{
    scale: 0.1
    gap: {0}
    pseudohuber: {1}
    delta: {2}
    penalty: {3}
    ranklossmult: {4}
    ranklossneg: {5}    
  }}
}}

layer {{
  name: "predaff"
  type: "Flatten"
  bottom: "affinity_output"
  top: "predaff"
}}

layer {{
  name: "affout"
  type: "Split"
  bottom: "affinity"
  top: "affout"
  include {{
    phase: TEST
  }}
}}
'''.format(args.loss_gap,boolstr(args.loss_pseudohuber),args.loss_delta, args.loss_penalty, args.ranklossmult, args.ranklossneg)
    
    return m
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    sys.stdout.write(create_model(args))
