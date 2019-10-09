#!/usr/bin/env python

'''Generate models for affinity predictions'''

# initialization: gaussian, positive_unitball, uniform, xavier, msra, radial
# kernel size: 3, 5

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
    batch_size: 10
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
    num_output: OUTPUT
    pad: PAD
    kernel_size: KSIZE
    stride: 1
    weight_filler {
      type: "FILLER"
      symmetric_fraction: FRACTION
    }
  }
}'''



finishunit = '''
layer {
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
layer {
  name: "unitNUMBER_func"
  type: "ELU"
  bottom: "unitNUMBER_conv1"
  top: "unitNUMBER_conv1"
}
''';

# normalization: none, LRN (across and within), Batch
# learning rat
# depth 3, width 32 (doubled)
def create_unit(num, ksize, filler, fraction):
    width = 32
    double = True
    ret = convunit.replace('NUMBER', str(num))
    if num == 1:
        ret = ret.replace('INLAYER','data')
    else:
        ret = ret.replace('INLAYER', 'unit%d_conv1'%(num-1))
                
    if num == 4:
        ksize = 3  #only 3x3 at this point
        
    pad = int(ksize/2)
    ret = ret.replace('PAD',str(pad))
    ret = ret.replace('KSIZE', str(ksize))
    outsize = width
    if double:
        outsize *= 2**(num-1)
    ret = ret.replace('OUTPUT', str(outsize)) 
    ret = ret.replace('FILLER', filler)
    ret = ret.replace('FRACTION', str(fraction))
        
    ret += finishunit.replace('NUMBER', str(num))
    return ret


def makemodel(ksize, filler, fraction):
    m = modelstart
    depth = 3
    for i in range(1,depth+1):
        m += create_unit(i, ksize, filler, fraction)
    m += endmodel.replace('LASTCONV','unit%d_conv1'%depth)
    
    return m
    

models = []
           
for filler in ['radial','gaussian', 'positive_unitball', 'uniform', 'xavier', 'msra','radial.5']:
    fraction = 1.0
    if filler == 'radial.5':
        filler = 'radial'
        fraction = 0.5
    for ksize in [5,3]:
        model = makemodel(ksize, filler, fraction)
        m = 'affinity_%s_%.1f_%d.model'%(filler,fraction,ksize)
        models.append(m)
        out = open(m,'w')
        out.write(model)
        
            
for m in models:
    print("train.py -m %s -p ../types/all_0.5_0_  --keep_best -t 1000 -i 100000 --reduced -o all_%s"%(m,m.replace('.model','')))
