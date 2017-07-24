#!/usr/bin/env python

'''Generate models for affinity predictions'''

# variables: 
# BALANCED(true)
# POSEPREDICT
# RECEPTOR(false)
# AFFMIN(0)
# AFFMAX(0)
# AFFSTEP(0)
basemodel = '''layer {
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
    root_folder: "../.."
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
    balanced: BALANCED
    stratify_receptor: RECEPTOR
    stratify_affinity_min: AFFMIN
    stratify_affinity_max: AFFMAX
    stratify_affinity_step: AFFSTEP
    has_affinity: true
    random_rotation: true
    random_translate: 2
    root_folder: "../.."
  }
}

layer {
  name: "unit1_pool"
  type: "Pooling"
  bottom: "data"
  top: "unit1_pool"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "unit1_conv1"
  type: "Convolution"
  bottom: "unit1_pool"
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
  name: "unit1_relu1"
  type: "ReLU"
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
    num_output: 64
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "unit2_relu1"
  type: "ReLU"
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
    num_output: 128
    pad: 1
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
    }
  }
}
layer {
  name: "unit3_relu1"
  type: "ReLU"
  bottom: "unit3_conv1"
  top: "unit3_conv1"
}

layer {
name: "split"
type: "Split"
bottom: "unit3_conv1"
top: "split"
}

POSEPREDICT

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
    gap: 0
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
posepredict='''layer {
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
'''

def makemodel(**kwargs):
    m = basemodel
    for (k,v) in kwargs.iteritems():
        m = m.replace(k,str(v))
    return m
    


conv = 3
resolution = 0.5

models = []
for pose in ['', posepredict]:    
    for balanced in ['true','false']:
        for receptor in ['true','false']:
            for affstrat in [True,False]:
                if affstrat:
                    amin = 2  #first group will be < 3
                    amax = 10 #last bin will be > 9
                    astep = 1
                else:
                    amin = amax = astep = 0
                                 
                model = makemodel(POSEPREDICT=pose, RECEPTOR=receptor,AFFMIN=amin,AFFMAX=amax,AFFSTEP=astep,BALANCED=balanced)
                m = 'affinity_p%d_rec%d_astrat%d_b%d.model'%(len(pose)>0,receptor=='true',affstrat,balanced=='true')
                models.append(m)
                out = open(m,'w')
                out.write(model)
            

unbalanced = set(['bestonly','crystal','posonly'])
single = set(['bestonly','crystal'])
for i in ['all','besty','posonly','crystal','bestonly']:
    #some around valid for the model - assume we die quickly?
    for m in models:
        if i in unbalanced:
             if'_b1' in m: continue
             if '_p1' in m: continue
        else: #balanced
            if '_b0' in m: continue
        if i in single:
            if '_rec1' in m: continue #only one per receptor, not much point

        print "train.py -m %s -p ../types/%s_0.5_0_ --keep_best -t 1000 -i 100000 --reduced -o %s_%s"%(m,i,i,m.replace('.model',''))
