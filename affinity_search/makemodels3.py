#!/usr/bin/env python

'''Generate models for affinity predictions'''

# variables: 
# non-linearity: ReLU, leaky, ELU, Sigmoid, TanH  (PReLU not current ndim compat)
# normalization: none, LRN, Batch
# learning rate: 0.01, 0.001, 0.1

modelstart = '''layer {
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
    resolution: 0.5
    shuffle: false
    balanced: false
    has_affinity: true
    root_folder: "../../"
  }
}
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
    resolution: 0.5
    shuffle: true
    balanced: true
    stratify_receptor: true
    stratify_affinity_min: 0
    stratify_affinity_max: 0
    stratify_affinity_step: 0
    has_affinity: true
    random_rotation: true
    random_translate: 2
    root_folder: "../../"
  }
}
'''

endmodel = '''layer {
    name: "split"
    type: "Split"
    bottom: "LASTCONV"
    top: "split"
}

layer {
  name: "output_fc"
  type: "InnerProduct"
  bottom: "split"
  top: "output_fc"
  inner_product_param {
    num_output: 2
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "output_fc"
  bottom: "label"
  top: "loss"
}

layer {
  name: "output"
  type: "Softmax"
  bottom: "output_fc"
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

layer {
  name: "output_fc_aff"
  type: "InnerProduct"
  bottom: "split"
  top: "output_fc_aff"
  inner_product_param {
    num_output: 1
    weight_filler {
      type: "xavier"
    }
  }
}

layer {
  name: "rmsd"
  type: "AffinityLoss"
  bottom: "output_fc_aff"
  bottom: "affinity"
  top: "rmsd"
  affinity_loss_param {
    scale: 0.1
    gap: 1
    penalty: 0
    pseudohuber: false
    delta: 0
  }
}

layer {
  name: "predaff"
  type: "Flatten"
  bottom: "output_fc_aff"
  top: "predaff"
}

layer {
  name: "affout"
  type: "Split"
  bottom: "affinity"
  top: "affout"
  include {
    phase: TEST
  }
}

'''

convunit = '''
layer {
  name: "unitNUMBER_pool"
  type: "Pooling"
  bottom: "INLAYER"
  top: "unitNUMBER_pool"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "unitNUMBER_conv1"
  type: "Convolution"
  bottom: "unitNUMBER_pool"
  top: "unitNUMBER_conv1"
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
  }
}'''


norms =  {
 'none': '',   
 'batch': '''layer {
  name: "unitNUMBER_norm"
  type: "BatchNorm"
  bottom: "unitNUMBER_conv1"
  top: "unitNUMBER_conv1"
}

layer {
 name: "unitNUMBER_scale"
 type: "Scale"
 bottom: "unitNUMBER_conv1"
 top: "unitNUMBER_conv1"
 scale_param {
  bias_term: true
 }
}
''', 
 'lrn': '''layer {
  name: "unitNUMBER_norm"
  type: "LRN"
  bottom: "unitNUMBER_conv1"
  top: "unitNUMBER_conv1"
}

layer {
 name: "unitNUMBER_scale"
 type: "Scale"
 bottom: "unitNUMBER_conv1"
 top: "unitNUMBER_conv1"
 scale_param {
  bias_term: true
 }
}
'''
}

relus = {
 'relu': '''layer {
  name: "unitNUMBER_func"
  type: "ReLU"
  bottom: "unitNUMBER_conv1"
  top: "unitNUMBER_conv1"
}''',
 'leaky': '''layer {
  name: "unitNUMBER_func"
  type: "ReLU"
  bottom: "unitNUMBER_conv1"
  top: "unitNUMBER_conv1"
  relu_param{
      negative_slope: 0.01
   }
}''',
 'elu':'''layer {
  name: "unitNUMBER_func"
  type: "ELU"
  bottom: "unitNUMBER_conv1"
  top: "unitNUMBER_conv1"
}''',
 'sigmoid':'''layer {
  name: "unitNUMBER_func"
  type: "Sigmoid"
  bottom: "unitNUMBER_conv1"
  top: "unitNUMBER_conv1"
}''',
 'tanh':'''layer {
  name: "unitNUMBER_func"
  type: "TanH"
  bottom: "unitNUMBER_conv1"
  top: "unitNUMBER_conv1"
}'''
}

# normalization: none, LRN (across and within), Batch
# learning rat
def create_unit(num, norm, func):
        
    ret = convunit.replace('NUMBER', str(num))
    if num == 1:
        ret = ret.replace('INLAYER','data')
    else:
        ret = ret.replace('INLAYER', 'unit%d_conv1'%(num-1))
    ret += norms[norm].replace('NUMBER', str(num))
    ret += relus[func].replace('NUMBER', str(num))
    return ret


def makemodel(norm, func):
    m = modelstart
    for i in [1,2,3]:
        m += create_unit(i, norm, func)
    m += endmodel.replace('LASTCONV','unit3_conv1')
    
    return m
    

models = []
for norm in sorted(norms.keys()):
    for func in sorted(relus.keys()):    
        model = makemodel(norm, func)
        m = 'affinity_%s_%s.model'%(norm,func)
        models.append(m)
        out = open(m,'w')
        out.write(model)
            
for m in models:
    for lr in [0.001, 0.01, 0.1]:
        print "train.py -m %s -p ../types/all_0.5_0_ --base_lr %f --keep_best -t 1000 -i 100000 --reduced -o all_%s_lr%.3f"%(m,lr,m.replace('.model',''),lr)
