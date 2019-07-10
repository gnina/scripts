#!/usr/bin/env python

'''Generate models for affinity predictions'''

# variables (for first layer only): 
# grid resolution: 0.25, 0.5, 1.0  # no 0.125 for now - blobs require small batch size
# convolution: none, 2x2 (stride 2), 4x4 (stride 4), 8x8 (stride 8)
#  convolution can have activation function after it or not, be grouped or not
# max pooling: none, 2x2, 4x4, 8x8
# above are combined to generate 1A grid for next layer

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
    batch_size: 1
    dimension: DIMENSION
    resolution: RESOLUTION
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
    batch_size:  20
    dimension: DIMENSION
    resolution: RESOLUTION
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


FIRSTLAYER

layer {
  name: "unit1_conv1"
  type: "Convolution"
  bottom: "INITIALNAME"
  top: "unit1_conv1"
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "unit1_norm"
  type: "LRN"
  bottom: "unit1_conv1"
  top: "unit1_conv1"
}

layer {
 name: "unit1_scale"
 type: "Scale"
 bottom: "unit1_conv1"
 top: "unit1_conv1"
 scale_param {
  bias_term: true
 }
}
layer {
  name: "unit1_func"
  type: "ELU"
  bottom: "unit1_conv1"
  top: "unit1_conv1"
}


layer {
  name: "unit2_pool"
  type: "Pooling"
  bottom: "unit1_conv1"
  top: "unit2_pool"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}


layer {
  name: "unit2_conv1"
  type: "Convolution"
  bottom: "unit2_pool"
  top: "unit2_conv1"
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "unit2_norm"
  type: "LRN"
  bottom: "unit2_conv1"
  top: "unit2_conv1"
}

layer {
 name: "unit2_scale"
 type: "Scale"
 bottom: "unit2_conv1"
 top: "unit2_conv1"
 scale_param {
  bias_term: true
 }
}
layer {
  name: "unit2_func"
  type: "ELU"
  bottom: "unit2_conv1"
  top: "unit2_conv1"
}


layer {
  name: "unit3_pool"
  type: "Pooling"
  bottom: "unit2_conv1"
  top: "unit3_pool"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}


layer {
  name: "unit3_conv1"
  type: "Convolution"
  bottom: "unit3_pool"
  top: "unit3_conv1"
  convolution_param {
    num_output: 32
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "unit3_norm"
  type: "LRN"
  bottom: "unit3_conv1"
  top: "unit3_conv1"
}

layer {
 name: "unit3_scale"
 type: "Scale"
 bottom: "unit3_conv1"
 top: "unit3_conv1"
 scale_param {
  bias_term: true
 }
}
layer {
  name: "unit3_func"
  type: "ELU"
  bottom: "unit3_conv1"
  top: "unit3_conv1"
}
layer {
    name: "split"
    type: "Split"
    bottom: "unit3_conv1"
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

poollayer = '''
layer {
  name: "initial_pool"
  type: "Pooling"
  bottom: "POOLINPUT"
  top: "initial_pool"
  pooling_param {
    pool: MAX
    kernel_size: SIZESTRIDE
    stride: SIZESTRIDE
  }
}
'''

convlayers = ['''layer {
  name: "initial_conv"
  type: "Convolution"
  bottom: "CONVINPUT"
  top: "initial_conv"
  convolution_param {
    num_output: 32
    pad: 0
    kernel_size: SIZESTRIDE
    stride: SIZESTRIDE
    weight_filler {
      type: "xavier"
    }
  }
}''', '''layer {
  name: "initial_conv"
  type: "Convolution"
  bottom: "CONVINPUT"
  top: "initial_conv"
  convolution_param {
    num_output: 35
    group: 35
    pad: 0
    kernel_size: SIZESTRIDE
    stride: SIZESTRIDE
    weight_filler {
      type: "xavier"
    }
  }
}''']

convafter = '''
layer {
  name: "initial_norm"
  type: "LRN"
  bottom: "initial_conv"
  top: "initial_norm"
}

layer {
 name: "initial_scale"
 type: "Scale"
 bottom: "initial_norm"
 top: "initial_scale"
 scale_param {
  bias_term: true
 }
}
layer {
  name: "initial_func"
  type: "ELU"
  bottom: "initial_scale"
  top: "initial_func"
}
'''

def makemodel(resolution, conv, pool, grouped, func, swapped):
    m = modelstart.replace('RESOLUTION','%.3f'%resolution)
    dim = 24-resolution
    m = m.replace('DIMENSION','%.3f'%dim)
    
    clayer = ''
    player = ''
    convname = 'initial_conv'
    if conv > 1: # have a layer
        clayer = convlayers[grouped]
        clayer = clayer.replace('SIZESTRIDE',str(conv))
        if func:
            clayer += convafter
            convname = 'initial_func'
    if pool > 1:
        player = poollayer.replace('SIZESTRIDE',str(pool))
    
    initial = ''
    
    if conv == 1 and pool == 1:
        m = m.replace('INITIALNAME','data')
    elif swapped:
        player = player.replace('POOLINPUT','data')
        if pool > 1:        
            ipool = 'initial_pool'
        else:
            ipool = 'data'
        clayer = clayer.replace('CONVINPUT',ipool)

        initial = player+clayer
        if conv > 1:
            iconv = convname
        else:
            iconv = 'initial_pool'
        m = m.replace('INITIALNAME',iconv)
    else:
        clayer = clayer.replace('CONVINPUT','data')
        if conv > 1:
            iconv = convname
        else:
            iconv = 'data'
        player = player.replace('POOLINPUT',iconv)
        initial = clayer+player
        if pool > 1:        
            ipool = 'initial_pool'
        else:
            ipool = convname
        m = m.replace('INITIALNAME',ipool)
        
    m = m.replace('FIRSTLAYER',initial)
    return m
    

models = []
for resolution in [0.25, 0.5, 1.0]:
    for conv in [1,2,4,8]:
        for pool in [1,2,4,8]:
            if resolution*conv*pool == 1.0:
                #valid combination
                for grouped in [0,1]:
                    for func in [0,1]:
                        for swapped in [0,1]:
                            if (conv == 1 or pool == 1) and swapped: # nothing to swap
                                continue
                            if conv == 1 and (grouped or func):
                                continue #no conv layer

                            model = makemodel(resolution, conv, pool, grouped, func, swapped)
                            m = 'affinity_%.3f_conv%d_pool%d_grouped%d_func%d_swap%d.model'%(resolution,conv,pool,grouped,func,swapped)
                            models.append(m)
                            out = open(m,'w')
                            out.write(model)

            
for m in models:
    print("train.py -m %s -p ../types/all_0.5_0_  --keep_best -t 1000 -i 100000 --reduced -o all_%s"%(m,m.replace('.model','')))
