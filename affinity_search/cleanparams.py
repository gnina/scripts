import makemodel

modeldefaults = makemodel.getdefaults()

def cleanparams(p):
    '''standardize params that do not matter'''
    for i in xrange(1,6):
        if p['conv%d_width'%i] == 0:
            for suffix in ['func', 'init', 'norm', 'size', 'stride', 'width']:
                name = 'conv%d_%s'%(i,suffix)
                p[name] = modeldefaults[name]
        if p['pool%d_size'%i] == 0:
            name = 'pool%d_type'%i
            p[name] = modeldefaults[name]
            
    if p['fc_pose_hidden'] == 0:
        p['fc_pose_func'] = modeldefaults['fc_pose_func']
        p['fc_pose_hidden2'] = modeldefaults['fc_pose_hidden2']
        p['fc_pose_func2'] = modeldefaults['fc_pose_func2']
    elif p['fc_pose_hidden2'] == 0:
        p['fc_pose_hidden2'] = modeldefaults['fc_pose_hidden2']
        p['fc_pose_func2'] = modeldefaults['fc_pose_func2']
        
    if p['fc_affinity_hidden'] == 0:
        p['fc_affinity_func'] = modeldefaults['fc_affinity_func']
        p['fc_affinity_hidden2'] = modeldefaults['fc_affinity_hidden2']
        p['fc_affinity_func2'] = modeldefaults['fc_affinity_func2']
    elif p['fc_affinity_hidden2'] == 0:
        p['fc_affinity_hidden2'] = modeldefaults['fc_affinity_hidden2']
        p['fc_affinity_func2'] = modeldefaults['fc_affinity_func2']        
        
    for (name,val) in p.iteritems():
        if 'item' in dir(val):
            p[name] = np.asscalar(val)
        if type(p[name]) == int:
            p[name] = float(p[name])
    return p
