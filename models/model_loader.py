import sys
import torch

## default weights paths
DGNN_WEIGHTS_PATH = "weights/dgnn_weights.pt"
STGCN_WEIGHTS_PATH = "weights/stgcn_500_5.pt"

def load_stgcn(weights_path):
    print("Loading stgcn...")
    sys.path.append("models/stgcn/")
    import st_gcn

    model = st_gcn.Model(3, 4, [])
    model.load_state_dict(torch.load(STGCN_WEIGHTS_PATH))
    model.cuda()
    model.double()
    return model


def load_dgnn(weights_path):
    print("Loading dgnn...")
    sys.path.append("models/dgnn/")
    import dgnn

    model = dgnn.Model()
    model.cuda()
    model.load_state_dict(torch.load(DGNN_WEIGHTS_PATH))
    return model

# def load_taew(weights_path):
# TODO
#     print("Loading taew...")
#     sys.path.append("models/taew/")
#     import hap
#     model = hap.HAPPY()
#     model.cuda()
#     loaded_vars = torch.load(weights_path)
#     model.load_state_dict(loaded_vars['model_dict'])
#     model_GRU_h_enc = loaded_vars['h_enc']
#     model_GRU_h_dec1 = loaded_vars['h_dec1']
#     model_GRU_h_dec = loaded_vars['h_dec']

#     return model

