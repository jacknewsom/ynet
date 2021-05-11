import torch
import torch.nn as nn
import sparseconvnet as scn
from ynet.models_losses.scn_network_base import SCNNetworkBase
from ynet.models_losses.ppn import PPN, PPNLoss
from ynet.models_losses.lovasz import MaskLovaszInterLoss
from collections import defaultdict
from ynet.models_losses.feature_mappings import PoolFeatureMapping, ConvolutionalFeatureMapping
from ynet.models_losses.uresnet import UResNetEncoder, UResNetDecoder
from module.utils.torch_modules import SinusoidalPositionalEncoding

class YResNetEncoder(SCNNetworkBase):
    def __init__(self, cfg, name='yresnet_encoder'):
        super(YResNetEncoder, self).__init__(cfg, name='network_base')
        self.model_config = cfg[name]

        # YResNet Configurations
        # Conv block repetition factor
        self.reps = self.model_config.get('reps', 2)  
        self.kernel_size = self.model_config.get('kernel_size', 2)
        self.num_strides = self.model_config.get('num_strides', 5)
        self.num_filters = self.model_config.get('filters', 16)
        self.nPlanes = [i*self.num_filters for i in range(1, self.num_strides+1)]
        # [filter size, filter stride]
        self.downsample = [self.kernel_size, 2]  

        dropout_prob = self.model_config.get('dropout_prob', 0.5)

        # Define Sparse YResNet Encoder
        self.encoding_block = scn.Sequential()
        self.encoding_conv = scn.Sequential()
        for i in range(self.num_strides):
            m = scn.Sequential()
            for _ in range(self.reps):
                self._resnet_block(m, self.nPlanes[i], self.nPlanes[i])
            self.encoding_block.add(m)
            m = scn.Sequential()
            if i < self.num_strides-1:
                m.add(
                    scn.BatchNormLeakyReLU(self.nPlanes[i], leakiness=self.leakiness)).add(
                    scn.Convolution(self.dimension, self.nPlanes[i], self.nPlanes[i+1], \
                        self.downsample[0], self.downsample[1], self.allow_bias)).add(
                    scn.Dropout(p=dropout_prob))
            self.encoding_conv.add(m)
            

    def forward(self, x):
        '''
        Vanilla YResNet Encoder
        INPUTS:
            - x (scn.SparseConvNetTensor): output from inputlayer (self.input)
        RETURNS:
            - features_encoder (list of SparseConvNetTensor): list of feature
            tensors in encoding path at each spatial resolution.
        '''
        # Embeddings at each layer
        features_enc = [x]
        # Loop over Encoding Blocks to make downsampled segmentation/clustering masks.
        for i, layer in enumerate(self.encoding_block):
            x = self.encoding_block[i](x)
            features_enc.append(x)
            x = self.encoding_conv[i](x)

        res = {
            "features_enc": features_enc,
            "deepest_layer": x
        }

        return res

class YResNetDecoder(SCNNetworkBase):
    def __init__(self, cfg, name='yresnet_decoder'):
        super(YResNetDecoder, self).__init__(cfg, name='network_base')
        self.model_config = cfg[name]

        self.reps = self.model_config.get('reps', 2)  # Conv block repetition factor
        self.kernel_size = self.model_config.get('kernel_size', 2)
        self.num_strides = self.model_config.get('num_strides', 5)
        self.num_filters = self.model_config.get('filters', 16)
        self.nPlanes = [i*self.num_filters for i in range(1, self.num_strides+1)]
        self.downsample = [self.kernel_size, 2]  # [filter size, filter stride]
        self.concat = scn.JoinTable()
        self.add = scn.AddTable()
        dropout_prob = self.model_config.get('dropout_prob', 0.5)

        self.encoder_num_filters = self.model_config.get('encoder_num_filters', None)
        if self.encoder_num_filters is None:
            self.encoder_num_filters = self.num_filters
        self.encoder_nPlanes = [i*self.encoder_num_filters for i in range(1, self.num_strides+1)]

        # Define Sparse YResNet Decoder.
        self.decoding_block = scn.Sequential()
        self.decoding_conv = scn.Sequential()
        for idx, i in enumerate(list(range(self.num_strides-2, -1, -1))):
            if idx == 0:
                m = scn.Sequential().add(
                    scn.BatchNormLeakyReLU(self.encoder_nPlanes[i+1], leakiness=self.leakiness)).add(
                    scn.Deconvolution(self.dimension, self.encoder_nPlanes[i+1], self.nPlanes[i],
                        self.downsample[0], self.downsample[1], self.allow_bias))
            else:
                m = scn.Sequential().add(
                    scn.BatchNormLeakyReLU(self.nPlanes[i+1], leakiness=self.leakiness)).add(
                    scn.Deconvolution(self.dimension, self.nPlanes[i+1], self.nPlanes[i],
                        self.downsample[0], self.downsample[1], self.allow_bias)).add(
                    scn.Dropout(p=dropout_prob))
            self.decoding_conv.add(m)
            m = scn.Sequential()
            for j in range(self.reps):
                self._resnet_block(m, self.nPlanes[i] + (self.encoder_nPlanes[i] \
                    if j == 0 else 0), self.nPlanes[i])
            self.decoding_block.add(m)

    def forward(self, features_enc, deepest_layer):
        '''
        Vanilla YResNet Decoder
        INPUTS:
            - features_enc (list of scn.SparseConvNetTensor): output of encoder.
        RETURNS:
            - features_dec (list of scn.SparseConvNetTensor): list of feature
            tensors in decoding path at each spatial resolution.
        '''
        features_dec = []
        x = deepest_layer
        for i, layer in enumerate(self.decoding_conv):
            encoder_feature = features_enc[-i-2]
            x = layer(x)
            x = self.concat([encoder_feature, x])
            x = self.decoding_block[i](x)
            features_dec.append(x)
        return features_dec

class YNet(SCNNetworkBase):
    supported_mapping_ops = ['conv', 'pool']

    def __init__(self, cfg, name='ynet_full'):
        super().__init__(cfg, name)

        self.model_config = cfg[name]
        self.num_filters = self.model_config.get('filters', 16)
        self.seed_dim = self.model_config.get('seed_dim', 1)
        self.sigma_dim = self.model_config.get('sigma_dim', 1)
        self.embedding_dim = self.model_config.get('embedding_dim', 3)
        self.inputKernel = self.model_config.get('input_kernel_size', 3)
        self.coordConv = self.model_config.get('coordConv', False)

        # YResNet Configurations
        # operation for mapping latent secondary features to primary features
        self.mapping_op = self.model_config.get('mapping_operation', 'pool')
        assert self.mapping_op in self.supported_mapping_ops

        # Network Freezing Options
        self.encoder_freeze = self.model_config.get('encoder_freeze', False)
        self.embedding_freeze = self.model_config.get('embedding_freeze', False)
        self.seediness_freeze = self.model_config.get('seediness_freeze', False)

        # Input Layer Configurations and commonly used scn operations.
        self.input = scn.Sequential().add(
            scn.InputLayer(self.dimension, self.spatial_size, mode=3)).add(
            scn.SubmanifoldConvolution(self.dimension, self.nInputFeatures, \
            self.num_filters, self.inputKernel, self.allow_bias)) # Kernel size 3, no bias
        self.add = scn.AddTable()

        # Preprocessing logic for secondary
        self.t_bn = scn.BatchNormLeakyReLU(1, leakiness=self.leakiness)
        self.netinnet = scn.Sequential()
        self._resnet_block(self.netinnet, 1, self.num_filters)

        # Timing information
        max_seq_len = self.model_config.get('max_seq_len', 5)
        self.pe = SinusoidalPositionalEncoding(max_seq_len, 1)

        # Backbone YResNet. Do NOT change namings!
        self.primary_encoder = YResNetEncoder(cfg, name='yresnet_encoder')
        self.secondary_encoder = YResNetEncoder(cfg, name='yresnet_encoder')

        if self.mapping_op == 'conv':
            self.mapping = ConvolutionalFeatureMapping(
                self.dimension, self.nPlanes[-1], self.nPlanes[-1], 2, 2, False
            )
        elif self.mapping_op == 'pool':
            self.mapping = PoolFeatureMapping(self.dimension, 2, 2,)

        self.seed_net = YResNetDecoder(cfg, name='seediness_decoder')
        self.cluster_net = YResNetDecoder(cfg, name='embedding_decoder')

        # Encoder-Decoder 1x1 Connections
        encoder_planes = [i for i in self.primary_encoder.nPlanes]
        cluster_planes = [i for i in self.cluster_net.nPlanes]
        seed_planes = [i for i in self.seed_net.nPlanes]

        self.skip_mode = self.model_config.get('skip_mode', 'default')

        self.cluster_skip = scn.Sequential()
        self.seed_skip = scn.Sequential()

        # Output Layers
        self.output_cluster = scn.Sequential()
        self._nin_block(self.output_cluster, self.cluster_net.num_filters, 4)
        self.output_cluster.add(scn.OutputLayer(self.dimension))

        self.output_seediness = scn.Sequential()
        self._nin_block(self.output_seediness, self.seed_net.num_filters, 1)
        self.output_seediness.add(scn.OutputLayer(self.dimension))

        if self.skip_mode == 'default':
            for p1, p2 in zip(encoder_planes, cluster_planes):
                self.cluster_skip.add(scn.Identity())
            for p1, p2 in zip(encoder_planes, seed_planes):
                self.seed_skip.add(scn.Identity())

        elif self.skip_mode == '1x1':
            for p1, p2 in zip(encoder_planes, cluster_planes):
                self._nin_block(self.cluster_skip, p1, p2)

            for p1, p2 in zip(encoder_planes, seed_planes):
                self._nin_block(self.seed_skip, p1, p2)

        else:
            raise ValueError('Invalid skip connection mode!')

        # Freeze Layers
        if self.encoder_freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if self.embedding_freeze:
            for p in self.cluster_net.parameters():
                p.requires_grad = False
            for p in self.output_cluster.parameters():
                p.requires_grad = False

        if self.seediness_freeze:
            for p in self.seed_net.parameters():
                p.requires_grad = False
            for p in self.output_seediness.parameters():
                p.requires_grad = False

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, primary, secondary):
        '''
        YResNet encoder
        INPUTS:
            - x (scn.SparseConvNetTensor): output from inputlayer (self.input) of primary data
            - t (scn.SparseConvNetTensor): output from inputlayer (self.input) of secondary data
        RETURNS:
            - features_encoder (list of SparseConvNetTensor): list of feature
            tensors in encoding path at each spatial resolution.
        '''

        # load primary, convert to SparseConvNetTensor, and encode
        point_cloud, = primary
        coords = point_cloud[:, 0:self.dimension+1].float()
        normalized_coords = (coords[:, :self.embedding_dim] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        features = point_cloud[:, self.dimension+1].float().view(-1, 1)
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)
        x = self.input((coords, features))
        primary_encoder_res = self.primary_encoder(x)
        x_e = primary_encoder_res['deepest_layer']

        # load secondary, convert to SparseConvNetTensor, and encode
        point_cloud_sec, = secondary
        coords_sec = point_cloud_sec[:, 0:self.dimension+1].float()
        normalized_coords_sec = (coords_sec[:, :self.embedding_dim] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        features_sec = point_cloud_sec[:, self.dimension+1].float().view(-1, 1)
        if self.coordConv:
            features_sec = torch.cat([normalized_coords_sec, features_sec], dim=1)
        # skip the initial convolution
        t = self.input[0]((coords_sec, features_sec))

        # preprocessing stuff
        t_p_bn = self.t_bn(t)
        t_nin = self.netinnet(t_p_bn)
        t_nin.features = t_nin.features + self.pe.pe[t_nin.get_spatial_locations()[:, -1]]

        # encode secondary
        secondary_encoder_res = self.secondary_encoder(t_nin)
        t_e = secondary_encoder_res['deepest_layer']

        # feature mapping
        t_mapped = self.mapping(t_e, x_e)
        x_t = self.add([x_e, t_mapped])

        features_enc = primary_encoder_res['features_enc']
        features_enc[-1] = x_t
        deepest_layer = x_t

        # Seediness decoder inputs
        seed_decoder_input = [None]
        for i, layer in enumerate(features_enc[1:]):
            seed_decoder_input.append(self.seed_skip[i](layer))
        deep_seed = self.seed_skip[-1](deepest_layer)
        
        # Clustering decoder inputs
        cluster_decoder_input = [None]
        for i, layer in enumerate(features_enc[1:]):
            cluster_decoder_input.append(self.cluster_skip[i](layer))
        deep_cluster = self.cluster_skip[-1](deepest_layer)

        # Decode along seediness path
        features_seediness = self.seed_net(seed_decoder_input, deep_seed)
        seediness = features_seediness[-1]
        seediness = self.output_seediness(seediness)

        # Decode along clustering path
        features_cluster = self.cluster_net(features_enc, deepest_layer)
        embeddings = features_cluster[-1]
        embeddings = self.output_cluster(embeddings)
        embeddings[:, :self.embedding_dim] = self.tanh(embeddings[:, :self.embedding_dim])
        embeddings[:, :self.embedding_dim] += normalized_coords

        res = {
            'embeddings': [embeddings[:, :self.embedding_dim]],
            'margins': [2 * self.sigmoid(embeddings[:, self.embedding_dim:])],
            'seediness': [self.sigmoid(seediness)],
            'coords': [coords]
        }
        return res

class YNetLoss(torch.nn.modules.loss._Loss):
    def __init__(self, cfg):
        super(YNetLoss, self).__init__()

        # Initialize loss components
        self.loss_config = cfg['full_chain_loss']
        self.spice_loss = MaskLovaszInterLoss(cfg, name='full_chain_loss')

        # Initialize the loss weights
        self.clustering_weight = self.loss_config.get('clustering_weight', 1.0)

    def forward(self, out, cluster_label):
        '''
        Forward propagation for YNetLoss
        INPUTS:
            - out (dict): result from forwarding YNet, with
            1) segmenation decoder 2) clustering decoder 3) seediness decoder,
            and PPN attachment to the segmentation branch.
            - cluster_label (list of Tensors): input data tensor of shape N x 10
              In row-index order:
              1. x coordinates
              2. y coordinates
              3. z coordinates
              4. batch indices
              5. energy depositions
              6. cluster labels
              7. segmentation labels (0-5, includes ghosts)
        '''

        # Apply the segmenation loss
        coords = cluster_label[0][:, :4]
        segment_label = cluster_label[0][:, -1]
        segment_label_tensor = torch.cat((coords, segment_label.reshape(-1,1)), dim=1)

        # Apply the CNN dense clustering loss
        fragment_label = cluster_label[0][:, 5]
        batch_idx = coords[:, -1].unique()
        res_cnn_clust = defaultdict(int)
        for bidx in batch_idx:
            # Get the loss input for this batch
            batch_mask = coords[:, -1] == bidx
            highE_mask = segment_label[batch_mask] != 4
            embedding_batch_highE = out['embeddings'][0][batch_mask][highE_mask]
            margins_batch_highE = out['margins'][0][batch_mask][highE_mask]
            seed_batch_highE = out['seediness'][0][batch_mask][highE_mask]
            slabels_highE = segment_label[batch_mask][highE_mask]
            clabels_batch_highE = fragment_label[batch_mask][highE_mask]

            # Get the clustering loss, append results
            loss_class, acc_class = self.spice_loss.combine_multiclass(
                embedding_batch_highE, margins_batch_highE,
                seed_batch_highE, slabels_highE, clabels_batch_highE)

            loss, accuracy = 0, 0
            for key, val in loss_class.items():
                res_cnn_clust[key+'_loss'] += (sum(val) / len(val))
                loss += (sum(val) / len(val))
            for key, val in acc_class.items():
                res_cnn_clust[key+'_accuracy'] += val
                accuracy += val

            res_cnn_clust['loss'] += loss/len(loss_class.values())/len(batch_idx)
            res_cnn_clust['accuracy'] += accuracy/len(acc_class.values())/len(batch_idx)

        cnn_clust_acc, cnn_clust_loss = res_cnn_clust['accuracy'], res_cnn_clust['loss']

        # Combine the results
        accuracy = res_cnn_clust['accuracy']
        loss = self.clustering_weight*res_cnn_clust['loss']

        res = {}
        res.update(res_cnn_clust)
        res['cnn_clust_accuracy'] = cnn_clust_acc
        res['cnn_clust_loss'] = cnn_clust_loss
        res['loss'] = loss
        res['accuracy'] = accuracy

        return res