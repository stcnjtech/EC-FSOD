NWPU_ALL_CATEGORIES = {
    1: ['airplane', 'ship', 'storagetank', 'baseball', 'tenniscourt',
    'basketball', 'groundtrackfield', 'harbor', 'bridge', 'vehicle'],
    2: ['airplane', 'ship', 'storagetank', 'baseball', 'tenniscourt',
    'basketball', 'groundtrackfield', 'harbor', 'bridge', 'vehicle'],
}

NWPU_NOVEL_CATEGORIES = {
    1: ["airplane", "baseball", "tenniscourt"],
    2: ["basketball","groundtrackfield","vehicle"],
}

NWPU_BASE_CATEGORIES = {
    1: ['ship', 'storagetank',
        'basketball', 'groundtrackfield', 'harbor', 'bridge', 'vehicle'],
    2: ['airplane', 'ship', 'storagetank', 'baseball', 'tenniscourt',
        'harbor', 'bridge'],
}

DIOR_ALL_CATEGORIES = {
    1: ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield',
    'groundtrackfield','harbor','overpass','ship','stadium',
    'storagetank','tenniscourt','trainstation','vehicle','windmill'],
    2: ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield',
    'groundtrackfield','harbor','overpass','ship','stadium',
    'storagetank','tenniscourt','trainstation','vehicle','windmill'],
    3: ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield',
    'groundtrackfield','harbor','overpass','ship','stadium',
    'storagetank','tenniscourt','trainstation','vehicle','windmill'],
    4: ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield',
    'groundtrackfield','harbor','overpass','ship','stadium',
    'storagetank','tenniscourt','trainstation','vehicle','windmill'],
}

DIOR_NOVEL_CATEGORIES = {
    1: ["ship", "baseballfield", "bridge","basketballcourt","chimney"],
    2: ["airplane","airport","Expressway-toll-station","harbor","groundtrackfield"],
    3: ["dam","golffield","storagetank","tenniscourt","vehicle"],
    4: ["Expressway-Service-area","overpass","stadium","trainstation","windmill"]
}

DIOR_BASE_CATEGORIES = {
    1: ['airplane', 'airport','dam', 'Expressway-Service-area', 'Expressway-toll-station', 
        'golffield','groundtrackfield','harbor','overpass','stadium',
    'storagetank','tenniscourt','trainstation','vehicle','windmill'],
    2: ['baseballfield', 'basketballcourt', 'bridge','chimney', 'dam',
         'Expressway-Service-area',  'golffield','overpass','ship','stadium',
    'storagetank','tenniscourt','trainstation','vehicle','windmill'],
    3: ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'Expressway-Service-area', 'Expressway-toll-station',
    'groundtrackfield','harbor','overpass','ship','stadium','trainstation','windmill'],
    4: ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam','Expressway-toll-station', 'golffield',
    'groundtrackfield','harbor','ship',
    'storagetank','tenniscourt','vehicle'],
}

def _get_nwpu_fewshot_instances_meta():
    ret = {
        "thing_classes": NWPU_ALL_CATEGORIES,
        "novel_classes": NWPU_NOVEL_CATEGORIES,
        "base_classes": NWPU_BASE_CATEGORIES,
    }
    return ret

def _get_dior_fewshot_instances_meta():
    ret = {
        "thing_classes": DIOR_ALL_CATEGORIES,
        "novel_classes": DIOR_NOVEL_CATEGORIES,
        "base_classes": DIOR_BASE_CATEGORIES,
    }
    return ret

def _get_builtin_metadata(dataset_name):
    if dataset_name == "nwpu_fewshot":
        return _get_nwpu_fewshot_instances_meta()
    elif dataset_name == "dior_fewshot":
        return _get_dior_fewshot_instances_meta()
    else:
        raise KeyError("No built-in metadata for dataset {}".format(dataset_name))