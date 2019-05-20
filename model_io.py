#!/usr/bin/env python3 -w

import sys
import os
import numpy as np
import json

from sklearn.cross_decomposition import PLSRegression

class Encoder4JSON(json.JSONEncoder):
    '''
    A standard NumPy array(a numpy.ndarray object) is not JSON serialisable.
    This is the workaround to transform all arrays to lists.
    '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(Encoder4JSON, self).default(obj)
        
def saveModelAsJSON(pls_model, filename):
    #print(pls_model.__dict__)
    model_dict = pls_model.__dict__
    json_txt = json.dumps(model_dict, cls=Encoder4JSON, indent=4, sort_keys=False)#True, separators=(',', ': '))
    #parsed = json.loads(pls_model.__dict__, cls=Encoder4JSON)
    #json_txt = json.dumps(parsed, indent=4, sort_keys=True)
    #print(json_txt)
    with open(filename, 'w') as file:
        file.write(json_txt)

def loadModelFromJSON(filename):
    pls_model = PLSRegression()
    json_data = open(filename).read()
    load_dict = json.loads(json_data)
    for k, v in enumerate(load_dict.keys()):
        #print(v)
        pls_model.__dict__[v] = load_dict[v]
    return pls_model

def getNumericFilenameList(path, extension='txt'):
    if not os.path.isdir(path):
        raise Exception('The directory(%s) of txt files doesn\'t exist.'%(path))

    item_list= os.listdir(path)
    number_list = []
    for item in item_list:
        #item_name = bytes(item, 'utf-8')
        #item_name = item.encode('utf-8')
        #full_name = path + u'/' + item_name
        full_name = path + '/' + item
        if os.path.isfile(full_name) and item.lower().endswith(extension.lower()):
            non_ext, ext = os.path.splitext(item)
            if non_ext.isdigit():
                number_list.append(int(non_ext))
    number_list.sort()
    filename_list = []
    for num in number_list:
        filename_list.append(str(num) + '.' + extension)
    return filename_list

def main(argv):
    pls = PLSRegression(n_components=10)
    saveModelAsJSON(pls, 'test.json')
    pls1 = loadModelFromJSON('test.json')
    assert(pls == pls1)
    return 0

if __name__ == '__main__':
    #sys.exit(main(sys.argv))
    main(sys.argv)
