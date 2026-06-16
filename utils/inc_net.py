import copy
import logging
import numpy as np
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear

def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    if name == 'cnn1d':
        from convs.cnn1d import CNN1DConvNet
        return CNN1DConvNet()
    
    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        self.convnet.load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )

class IL2ANet(IncrementalNet):

    def update_fc(self, num_old, num_total, num_aux):
        fc = self.generate_fc(self.feature_dim, num_total+num_aux)
        if self.fc is not None:
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:num_old] = weight[:num_old]
            fc.bias.data[:num_old] = bias[:num_old]
        del self.fc
        self.fc = fc

class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class BiasLayer_BIC(nn.Module):
    def __init__(self):
        super(BiasLayer_BIC, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer_BIC())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(DERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logics: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.args))
        else:
            self.convnets.append(get_convnet(self.args))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)
        # self.aux_fc = self.generate_fc(self.out_dim, new_task_size)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.args = args
        self._device = args["device"][0]

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        # fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        fc = self.generate_fc(self.feature_dim, nb_classes).to(self._device)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:

                weight = torch.cat([weight, nextperiod_initialization])
            else:
                # weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.convnets.append(get_convnet(self.args))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc
    

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x , bias=True):
        ret_x = x.clone()
        ret_x = (self.alpha+1) * x # + self.beta
        if bias:
            ret_x = ret_x + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class BEEFISONet(nn.Module):
    def __init__(self, args, pretrained):
        super(BEEFISONet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.old_fc = None
        self.new_fc = None
        self.task_sizes = []
        self.forward_prototypes = None
        self.backward_prototypes = None
        self.args = args
        self.biases = nn.ModuleList()
        self._device = args["device"][0]

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        
        if self.old_fc is None:
            fc = self.new_fc
            out = fc(features)
        else:
            '''
            merge the weights
            '''
            new_task_size = self.task_sizes[-1]
            # fc_weight = torch.cat([self.old_fc.weight,torch.zeros((new_task_size,self.feature_dim-self.out_dim)).cuda()],dim=0)
            fc_weight = torch.cat([self.old_fc.weight,torch.zeros((new_task_size,self.feature_dim-self.out_dim)).to(self._device)],dim=0)
            new_fc_weight = self.new_fc.weight
            new_fc_bias = self.new_fc.bias
            for i in range(len(self.task_sizes)-2,-1,-1):
                new_fc_weight = torch.cat([*[self.biases[i](self.backward_prototypes.weight[i].unsqueeze(0),bias=False) for _ in range(self.task_sizes[i])],new_fc_weight],dim=0)
                new_fc_bias = torch.cat([*[self.biases[i](self.backward_prototypes.bias[i].unsqueeze(0),bias=True) for _ in range(self.task_sizes[i])], new_fc_bias])
            fc_weight = torch.cat([fc_weight,new_fc_weight],dim=1)
            # fc_bias = torch.cat([self.old_fc.bias,torch.zeros(new_task_size).cuda()])
            fc_bias = torch.cat([self.old_fc.bias, torch.zeros(new_task_size).to(self.args["device"])])
            fc_bias=+new_fc_bias
            logits = features@fc_weight.permute(1,0)+fc_bias
            out = {"logits":logits}        

            new_fc_weight = self.new_fc.weight
            new_fc_bias = self.new_fc.bias
            for i in range(len(self.task_sizes)-2,-1,-1):
                new_fc_weight = torch.cat([self.backward_prototypes.weight[i].unsqueeze(0),new_fc_weight],dim=0)
                new_fc_bias = torch.cat([self.backward_prototypes.bias[i].unsqueeze(0), new_fc_bias])
            out["train_logits"] = features[:,-self.out_dim:]@new_fc_weight.permute(1,0)+new_fc_bias 
        out.update({"eval_logits": out["logits"],"energy_logits":self.forward_prototypes(features[:,-self.out_dim:])["logits"]})
        return out

    def update_fc_before(self, nb_classes):
        new_task_size = nb_classes - sum(self.task_sizes)
        self.biases = nn.ModuleList([BiasLayer() for i in range(len(self.task_sizes))])
        self.convnets.append(get_convnet(self.args))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        if self.new_fc is not None:
            self.fe_fc = self.generate_fc(self.out_dim, nb_classes)
            self.backward_prototypes = self.generate_fc(self.out_dim,len(self.task_sizes))
            self.convnets[-1].load_state_dict(self.convnets[0].state_dict())
        self.forward_prototypes = self.generate_fc(self.out_dim, nb_classes)
        self.new_fc = self.generate_fc(self.out_dim,new_task_size)
        self.task_sizes.append(new_task_size)
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc
    
    def update_fc_after(self):
        if self.old_fc is not None:
            old_fc = self.generate_fc(self.feature_dim, sum(self.task_sizes))
            new_task_size = self.task_sizes[-1]
            # old_fc.weight.data = torch.cat([self.old_fc.weight.data,torch.zeros((new_task_size,self.feature_dim-self.out_dim)).cuda()],dim=0)
            old_fc.weight.data = torch.cat([self.old_fc.weight.data, torch.zeros((new_task_size, self.feature_dim - self.out_dim)).to(self._device)], dim=0)
            new_fc_weight = self.new_fc.weight.data
            new_fc_bias = self.new_fc.bias.data
            for i in range(len(self.task_sizes)-2,-1,-1):
                new_fc_weight = torch.cat([*[self.biases[i](self.backward_prototypes.weight.data[i].unsqueeze(0),bias=False) for _ in range(self.task_sizes[i])], new_fc_weight],dim=0)
                new_fc_bias = torch.cat([*[self.biases[i](self.backward_prototypes.bias.data[i].unsqueeze(0),bias=True) for _ in range(self.task_sizes[i])], new_fc_bias])
            old_fc.weight.data = torch.cat([old_fc.weight.data,new_fc_weight],dim=1)
            # old_fc.bias.data = torch.cat([self.old_fc.bias.data,torch.zeros(new_task_size).cuda()])
            old_fc.bias.data = torch.cat([self.old_fc.bias.data, torch.zeros(new_task_size).to(self._device)])
            old_fc.bias.data+=new_fc_bias
            self.old_fc = old_fc
        else:
            self.old_fc  = self.new_fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma


class AdaptiveNet(nn.Module):
    def __init__(self, args, pretrained):
        super(AdaptiveNet, self).__init__()
        self.convnet_type = args["convnet_type"]
        self.TaskAgnosticExtractor , _ = get_convnet(args, pretrained) #Generalized blocks
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList() #Specialized Blocks
        self.pretrained=pretrained
        self.out_dim=None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []
        self.args=args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.AdaptiveExtractors)
    
    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out=self.fc(features) #{logits: self.fc(features)}

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"] 

        out.update({"aux_logits":aux_logits,"features":features})
        out.update({"base_features":base_feature_map})
        return out
                
        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''
        
    def update_fc(self,nb_classes):
        _ , _new_extractor = get_convnet(self.args)
        if len(self.AdaptiveExtractors)==0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            logging.info(self.AdaptiveExtractors[-1])
            self.out_dim=self.AdaptiveExtractors[-1].feature_dim        
        fc = self.generate_fc(self.feature_dim, nb_classes)             
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)
 
    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['convnet']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k:v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k:v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class VectorGate(nn.Module):
    def __init__(self, feature_dim):
        super(VectorGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
    def forward(self, phi_x, a_x):
        combined = torch.cat([phi_x, a_x], dim=1)
        return self.gate(combined)


class AFSICIDSNet(nn.Module):
    def __init__(self, args, pretrained=False):
        super(AFSICIDSNet, self).__init__()
        self.args = args
        self.convnet = get_convnet(args, pretrained)
        self.feature_dim = self.convnet.out_dim
        self.fc = None
        self.stability_encoder = None
        self.plasticity_adapter = None
        self.gate = None
        self._device = args["device"][0]

    def extract_vector(self, x):
        if self.stability_encoder is None:
            if hasattr(self.convnet, "extract_vector"):
                return self.convnet.extract_vector(x)
            else:
                return self.convnet(x)["features"]
        else:
            self.stability_encoder.eval()
            with torch.no_grad():
                if hasattr(self.stability_encoder, "extract_vector"):
                    phi_x = self.stability_encoder.extract_vector(x)
                else:
                    phi_x = self.stability_encoder(x)["features"]
            
            if hasattr(self.plasticity_adapter, "extract_vector"):
                a_x = self.plasticity_adapter.extract_vector(x)
            else:
                a_x = self.plasticity_adapter(x)["features"]
            
            g = self.gate(phi_x, a_x)
            z = g * phi_x + (1.0 - g) * a_x
            return z

    def forward(self, x):
        z = self.extract_vector(x)
        out = self.fc(z)
        out.update({"features": z})
        return out

    def freeze_stability_encoder(self):
        if self.stability_encoder is not None:
            for p in self.stability_encoder.parameters():
                p.requires_grad = False
            self.stability_encoder.eval()

    def unfreeze_adapter(self):
        if self.plasticity_adapter is not None:
            for p in self.plasticity_adapter.parameters():
                p.requires_grad = True
            self.plasticity_adapter.train()
        if self.gate is not None:
            for p in self.gate.parameters():
                p.requires_grad = True
            self.gate.train()

    def unfreeze_incremental_params(self):
        """Alias for unfreeze_adapter, matching instruction API."""
        self.unfreeze_adapter()
        if self.fc is not None:
            for p in self.fc.parameters():
                p.requires_grad = True

    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes, sigma=True)
        if self.fc is not None:
            nb_output = self.fc.out_features
            fc.weight.data[:nb_output] = self.fc.weight.data[:nb_output]
            if self.fc.sigma is not None:
                fc.sigma.data = self.fc.sigma.data
        del self.fc
        self.fc = fc

    def transition_to_incremental_stage(self):
        class FrozenFeatureExtractor(nn.Module):
            def __init__(self, extractor):
                super().__init__()
                self.extractor = copy.deepcopy(extractor)
                for p in self.extractor.parameters():
                    p.requires_grad = False
                self.extractor.eval()
            def forward(self, x):
                return self.extractor(x)
            def extract_vector(self, x):
                if hasattr(self.extractor, "extract_vector"):
                    return self.extractor.extract_vector(x)
                else:
                    return self.extractor(x)["features"]

        if self.stability_encoder is None:
            self.stability_encoder = FrozenFeatureExtractor(self.convnet)
            # After the first incremental stage, the old base convnet is kept only
            # as the frozen stability branch. It should not appear trainable.
            for p in self.convnet.parameters():
                p.requires_grad = False
            self.convnet.eval()
        else:
            class FusedFeatureExtractor(nn.Module):
                def __init__(self, stability, plasticity, gate):
                    super().__init__()
                    self.stability = copy.deepcopy(stability)
                    self.plasticity = copy.deepcopy(plasticity)
                    self.gate = copy.deepcopy(gate)
                    for p in self.parameters():
                        p.requires_grad = False
                    self.eval()
                def extract_vector(self, x):
                    phi_x = self.stability.extract_vector(x)
                    a_x = self.plasticity.extract_vector(x)
                    g = self.gate(phi_x, a_x)
                    return g * phi_x + (1.0 - g) * a_x
                def forward(self, x):
                    return {"features": self.extract_vector(x)}

            self.stability_encoder = FusedFeatureExtractor(self.stability_encoder, self.plasticity_adapter, self.gate)
        
        class BottleneckFeatureAdapter(nn.Module):
            def __init__(self, feature_source, feature_dim, bottleneck_dim):
                super().__init__()
                self.frozen_source = copy.deepcopy(feature_source)
                for p in self.frozen_source.parameters():
                    p.requires_grad = False
                self.frozen_source.eval()
                self.adapter = nn.Sequential(
                    nn.Linear(feature_dim, bottleneck_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(bottleneck_dim, feature_dim),
                )

            def extract_vector(self, x):
                self.frozen_source.eval()
                with torch.no_grad():
                    base = self.frozen_source.extract_vector(x)
                return base + self.adapter(base)

            def forward(self, x):
                return {"features": self.extract_vector(x)}

        bottleneck_dim = int(self.args.get("adapter_bottleneck", max(8, self.feature_dim // 4)))
        self.plasticity_adapter = BottleneckFeatureAdapter(self.stability_encoder, self.feature_dim, bottleneck_dim)
        self.gate = VectorGate(self.feature_dim)
        self.to(self._device)

    def init_new_class_weights_from_prototypes(self, prototypes, class_ids):
        for cid in class_ids:
            if cid < self.fc.out_features:
                if isinstance(prototypes, dict):
                    if cid not in prototypes:
                        continue
                    proto = prototypes[cid]
                else:
                    if cid >= len(prototypes):
                        continue
                    proto = prototypes[cid]
                if isinstance(proto, np.ndarray):
                    proto = torch.from_numpy(proto).float()
                proto = proto.to(self.fc.weight.device)
                proto_norm = proto / (torch.norm(proto, p=2) + 1e-8)
                self.fc.weight.data[cid] = proto_norm

    def get_trainable_incremental_params(self):
        params = []
        if self.plasticity_adapter is not None:
            params.extend(self.plasticity_adapter.parameters())
        if self.gate is not None:
            params.extend(self.gate.parameters())
        if self.fc is not None:
            params.extend(self.fc.parameters())
        return params

    def get_incremental_state_dict(self):
        """Return state dict of only incremental (adapter/gate/fc) parameters."""
        state = {}
        full_sd = self.state_dict()
        for k, v in full_sd.items():
            if "plasticity_adapter.frozen_source" in k:
                continue
            if any(sub in k for sub in ["plasticity_adapter.adapter", "gate", "fc"]):
                state[k] = v
        return state

    def load_incremental_state_dict(self, state_dict):
        """Load only incremental parameters from a state dict."""
        own_sd = self.state_dict()
        for k, v in state_dict.items():
            if k in own_sd:
                own_sd[k] = v
        self.load_state_dict(own_sd)
