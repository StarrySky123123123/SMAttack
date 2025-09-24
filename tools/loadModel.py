import os
import sys

sys.path.append("..")

from models.stgcn.st_gcn import STGCN_Model
from models.msg3d.msg3d import Model as MCG3D_Model
from models.agcn.agcn import Model as AGCN_Model
from models.hdgcn.HDGCN import Model as HDGCN_Model
from models.ctrgcn.ctrgcn import Model as CTRGCN_Model

def weights_init(model):
    with torch.no_grad():
        for child in list(model.children()):
            print("init ", child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('weights initialization finished!')


def loadSTGCN():
    load_path = 'your_path'
    print(f'加载STGCN模型:{load_path}')
    graph_args = {
        "layout": "ntu-rgb+d",
        "strategy": "spatial"
    }
    stgcn = STGCN_Model(3, 60, graph_args, True)
    stgcn.eval()
    pretrained_weights = torch.load(load_path)
    stgcn.load_state_dict(pretrained_weights)
    stgcn.cuda()

    return stgcn


def loadSTGCN120():
    load_path = 'your_path'
    print(f'加载STGCN120模型:{load_path}')
    graph_args = {
        "layout": "ntu-rgb+d",
        "strategy": "spatial"
    }
    stgcn = STGCN_Model(3, 120, graph_args, True)
    stgcn.eval()
    pretrained_weights = torch.load(load_path)
    stgcn.load_state_dict(pretrained_weights)
    stgcn.cuda()

    return stgcn


def loadSTGCN_Kinetics():
    load_path = 'your_path'
    print(f'加载STGCN-Kinetics模型:{load_path}')
    graph_args = {
        "layout": "openpose",
        "strategy": "spatial"
    }
    stgcn = STGCN_Model(3, 400, graph_args, True)
    stgcn.eval()
    pretrained_weights = torch.load(load_path)
    stgcn.load_state_dict(pretrained_weights)
    stgcn.cuda()

    return stgcn


def loadMSG3D():
    load_path = 'your_path'
    print(f'加载MSG3D模型:{load_path}')
    mcg3d = MCG3D_Model(
        num_class=60,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )
    mcg3d.eval()
    pretrained_weights = torch.load(load_path)
    mcg3d.load_state_dict(pretrained_weights)
    mcg3d.cuda()

    return mcg3d


def loadMSG3D120():
    load_path = 'your_path'
    print(f'加载MSG3D120模型:{load_path}')
    mcg3d = MCG3D_Model(
        num_class=120,
        num_point=25,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.ntu_rgb_d.AdjMatrixGraph'
    )
    mcg3d.eval()
    pretrained_weights = torch.load(load_path)
    mcg3d.load_state_dict(pretrained_weights)
    mcg3d.cuda()

    return mcg3d


def loadMSG3D_Kinetics():
    load_path = 'your_path'
    print(f'加载MSG3D_Kinetics模型:{load_path}')
    mcg3d = MCG3D_Model(
        num_class=400,
        num_point=18,
        num_person=2,
        num_gcn_scales=8,
        num_g3d_scales=8,
        graph='graph.kinetics.AdjMatrixGraph'
    )
    mcg3d.eval()
    pretrained_weights = torch.load(load_path)
    mcg3d.load_state_dict(pretrained_weights)
    mcg3d.cuda()

    return mcg3d

def loadAGCN():
    load_path = 'your_path'
    print(f'加载agcn模型:{load_path}')
    agcn = AGCN_Model(
        num_class=60,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    agcn.eval()
    pretrained_weights = torch.load(load_path)
    agcn.load_state_dict(pretrained_weights)
    agcn.cuda()

    return agcn

def loadAGCN120():
    load_path = 'your_path'
    print(f'加载agcn120模型:{load_path}')
    agcn = AGCN_Model(
        num_class=120,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'})
    agcn.eval()
    pretrained_weights = torch.load(load_path)
    agcn.load_state_dict(pretrained_weights)
    agcn.cuda()

    return agcn


def loadAGCN_Kinetics():
    load_path = 'your_path'
    print(f'加载agcn_kinetics模型:{load_path}')
    agcn = AGCN_Model(
        num_class=400,
        num_point=18,
        num_person=2,
        graph='graph.kinetics.Graph', graph_args={'labeling_mode': 'spatial'})
    agcn.eval()
    pretrained_weights = torch.load(load_path)
    agcn.load_state_dict(pretrained_weights)
    agcn.cuda()

    return agcn


def loadHDGCN():
    load_path = 'your_path'
    print(f'加载hdgcn模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial",
        "CoM": 21
    }
    hdgcn = HDGCN_Model(
        graph='graph.ntu_rgb_d_hierarchy.Graph',
        graph_args=graph_args
    )
    hdgcn.eval()
    pretrained_weights = torch.load(load_path)
    hdgcn.load_state_dict(pretrained_weights)
    hdgcn.cuda()

    return hdgcn


def loadHDGCN120():
    load_path = 'your_path'
    print(f'加载hdgcn120模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial",
        "CoM": 21
    }
    hdgcn = HDGCN_Model(
        num_class=120,
        graph='graph.ntu_rgb_d_hierarchy.Graph',
        graph_args=graph_args
    )
    hdgcn.eval()
    pretrained_weights = torch.load(load_path)
    hdgcn.load_state_dict(pretrained_weights)
    hdgcn.cuda()

    return hdgcn


def loadHDGCN_Kinetics():
    load_path = 'your_path'
    print(f'加载hdgcn_kinetics模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial",
        "CoM": 1
    }
    hdgcn = HDGCN_Model(
        num_class=400,
        num_point=18,
        graph='graph.ntu_rgb_d_hierarchy.Graph_Kinetics',
        graph_args=graph_args
    )
    hdgcn.eval()
    pretrained_weights = torch.load(load_path)
    hdgcn.load_state_dict(pretrained_weights)
    hdgcn.cuda()

    return hdgcn


def loadCTRGCN():
    load_path = 'your_path'
    print(f'加载ctrgcn模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial"
    }
    ctrgcn = CTRGCN_Model(
        num_class=60,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph',
        graph_args=graph_args
    )
    ctrgcn.eval()
    pretrained_weights = torch.load(load_path)
    ctrgcn.load_state_dict(pretrained_weights)
    ctrgcn.cuda()

    return ctrgcn


def loadCTRGCN120():
    load_path = 'your_path'
    print(f'加载ctrgcn模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial"
    }
    ctrgcn = CTRGCN_Model(
        num_class=120,
        num_point=25,
        num_person=2,
        graph='graph.ntu_rgb_d.Graph',
        graph_args=graph_args
    )
    ctrgcn.eval()
    pretrained_weights = torch.load(load_path)
    ctrgcn.load_state_dict(pretrained_weights)
    ctrgcn.cuda()

    return ctrgcn


def loadCTRGCN_Kinetics():
    load_path = 'your_path'
    print(f'加载ctrgcn_kinetics模型:{load_path}')
    graph_args = {
        "labeling_mode": "spatial"
    }
    ctrgcn = CTRGCN_Model(
        num_class=400,
        num_point=18,
        num_person=2,
        graph='graph.kinetics.Graph',
        graph_args=graph_args
    )
    ctrgcn.eval()
    pretrained_weights = torch.load(load_path)
    ctrgcn.load_state_dict(pretrained_weights)
    ctrgcn.cuda()

    return ctrgcn


def getModel(AttackedModel):
    if AttackedModel == 'msg3d':
        model = loadMSG3D()
    elif AttackedModel == 'msg3d120':
        model = loadMSG3D120()
    elif AttackedModel == 'msg3d_kinetics':
        model = loadMSG3D_Kinetics()
    elif AttackedModel == 'agcn':
        model = loadAGCN()
    elif AttackedModel == 'agcn120':
        model = loadAGCN120()
    elif AttackedModel == 'agcn_kinetics':
        model = loadAGCN_Kinetics()
    elif AttackedModel == 'hdgcn':
        model = loadHDGCN()
    elif AttackedModel == 'hdgcn120':
        model = loadHDGCN120()
    elif AttackedModel == 'hdgcn_kinetics':
        model = loadHDGCN_Kinetics()
    elif AttackedModel == 'ctrgcn':
        model = loadCTRGCN()
    elif AttackedModel == 'ctrgcn120':
        model = loadCTRGCN120()
    elif AttackedModel == 'ctrgcn_kinetics':
        model = loadCTRGCN_Kinetics()
    elif AttackedModel == 'stgcn120':
        model = loadSTGCN120()
    elif AttackedModel == 'stgcn_kinetics':
        model = loadSTGCN_Kinetics()
    else:
        model = loadSTGCN()
    model.eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"模型总参数数量：{total_params:.2f} M")
    return model
