import numpy as n
import util as u

def makew(name, idx, shape, params=None):
    foo=u.unpickle('/data/t-maoqu/convnets/netsaves/imnet21841aggregated/7.1500')
    #foo=u.unpickle('/home/NORTHAMERICA/t-maoqu/share/net/47.250')
    if name=='transfer':
        return 0.1*n.ones((shape[0], shape[1]), n.single)
    for i in foo['model_state']['layers']:
        if i['name']==name:
            return i['weights'][idx]

def makeb(name, shape, params=None):
    foo=u.unpickle('/data/t-maoqu/convnets/netsaves/imnet21841aggregated/7.1500')
    #foo=u.unpickle('/home/NORTHAMERICA/t-maoqu/share/net/47.250')
    for i in foo['model_state']['layers']:
        if i['name']==name:
            return i['biases']
