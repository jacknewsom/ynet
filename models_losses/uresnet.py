import torch
import torch.nn as nn
import sparseconvnet as scn
from ynet.models_losses.scn_network_base import SCNNetworkBase

class UResNetEncoder(SCNNetworkBase):
    def __init__(self, cfg, name='uresnet_encoder'):
        super(UResNetEncoder, self).__init__(cfg, name='network_base')
        self.model_config = cfg[name]

        # UResNet Configurations
        # Conv block repetition factor
        self.reps = self.model_config.get('reps', 2)  
        self.kernel_size = self.model_config.get('kernel_size', 2)
        self.num_strides = self.model_config.get('num_strides', 5)
        self.num_filters = self.model_config.get('filters', 16)
        self.nPlanes = [i*self.num_filters for i in range(1, self.num_strides+1)]
        # [filter size, filter stride]
        self.downsample = [self.kernel_size, 2]  

        dropout_prob = self.model_config.get('dropout_prob', 0)

        # Define Sparse UResNet Encoder
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
        Vanilla UResNet Encoder
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

class UResNetDecoder(SCNNetworkBase):

    def __init__(self, cfg, name='uresnet_decoder'):
        super(UResNetDecoder, self).__init__(cfg, name='network_base')
        self.model_config = cfg[name]

        self.reps = self.model_config.get('reps', 2)  # Conv block repetition factor
        self.kernel_size = self.model_config.get('kernel_size', 2)
        self.num_strides = self.model_config.get('num_strides', 5)
        self.num_filters = self.model_config.get('filters', 16)
        self.nPlanes = [i*self.num_filters for i in range(1, self.num_strides+1)]
        self.downsample = [self.kernel_size, 2]  # [filter size, filter stride]
        self.concat = scn.JoinTable()
        self.add = scn.AddTable()
        dropout_prob = self.model_config.get('dropout_prob', 0)

        self.encoder_num_filters = self.model_config.get('encoder_num_filters', None)
        if self.encoder_num_filters is None:
            self.encoder_num_filters = self.num_filters
        self.encoder_nPlanes = [i*self.encoder_num_filters for i in range(1, self.num_strides+1)]

        # Define Sparse UResNet Decoder.
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
        Vanilla UResNet Decoder
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

class UNet(SCNNetworkBase):
    '''
    Neural network driving feature extraction for clustering
    '''

    def __init__(self, cfg, name='unet_full'):
        super().__init__(cfg, name)

        self.model_config = cfg[name]
        self.num_filters = self.model_config.get('filters', 16)
        self.ghost = self.model_config.get('ghost', False)
        self.seed_dim = self.model_config.get('seed_dim', 1)
        self.sigma_dim = self.model_config.get('sigma_dim', 1)
        self.embedding_dim = self.model_config.get('embedding_dim', 3)
        self.num_classes = self.model_config.get('num_classes', 5)
        self.num_gnn_features = self.model_config.get('num_gnn_features', 16)
        self.inputKernel = self.model_config.get('input_kernel_size', 3)
        self.coordConv = self.model_config.get('coordConv', False)

        # Network Freezing Options
        self.encoder_freeze = self.model_config.get('encoder_freeze', False)
        self.ppn_freeze = self.model_config.get('ppn_freeze', False)
        self.segmentation_freeze = self.model_config.get('segmentation_freeze', False)
        self.embedding_freeze = self.model_config.get('embedding_freeze', False)
        self.seediness_freeze = self.model_config.get('seediness_freeze', False)

        # Input Layer Configurations and commonly used scn operations.
        self.input = scn.Sequential().add(
            scn.InputLayer(self.dimension, self.spatial_size, mode=3)).add(
            scn.SubmanifoldConvolution(self.dimension, self.nInputFeatures, \
            self.num_filters, self.inputKernel, self.allow_bias)) # Kernel size 3, no bias
        self.concat = scn.JoinTable()
        self.add = scn.AddTable()

        # Backbone UResNet. Do NOT change namings!
        self.encoder = UResNetEncoder(cfg, name='uresnet_encoder')

        # self.seg_net = UResNetDecoder(cfg, name='segmentation_decoder')
        self.seed_net = UResNetDecoder(cfg, name='seediness_decoder')
        self.cluster_net = UResNetDecoder(cfg, name='embedding_decoder')

        # Encoder-Decoder 1x1 Connections
        encoder_planes = [i for i in self.encoder.nPlanes]
        # seg_planes = [i for i in self.seg_net.nPlanes]
        cluster_planes = [i for i in self.cluster_net.nPlanes]
        seed_planes = [i for i in self.seed_net.nPlanes]

        # print("Encoder Planes: ", encoder_planes)
        # print("Seg Planes: ", seg_planes)
        # print("Cluster Planes: ", cluster_planes)
        # print("Seediness Planes: ", seed_planes)

        self.skip_mode = self.model_config.get('skip_mode', 'default')

        # self.seg_skip = scn.Sequential()
        self.cluster_skip = scn.Sequential()
        self.seed_skip = scn.Sequential()

        # print(self.seg_skip)
        # print(self.cluster_skip)
        # print(self.seed_skip)

        # Output Layers
        self.output_cluster = scn.Sequential()
        self._nin_block(self.output_cluster, self.cluster_net.num_filters, 4)
        self.output_cluster.add(scn.OutputLayer(self.dimension))

        self.output_seediness = scn.Sequential()
        self._nin_block(self.output_seediness, self.seed_net.num_filters, 1)
        self.output_seediness.add(scn.OutputLayer(self.dimension))

        '''
        self.output_segmentation = scn.Sequential()
        self._nin_block(self.output_segmentation, self.seg_net.num_filters, self.num_classes)
        self.output_segmentation.add(scn.OutputLayer(self.dimension))
        '''

        '''
        self.output_gnn_features = scn.Sequential()
        sum_filters = self.seg_net.num_filters + self.seed_net.num_filters + self.cluster_net.num_filters
        self._resnet_block(self.output_gnn_features, sum_filters, self.num_gnn_features)
        self._nin_block(self.output_gnn_features, self.num_gnn_features, self.num_gnn_features)
        self.output_gnn_features.add(scn.OutputLayer(self.dimension))
        '''

        if self.ghost:
            self.linear_ghost = scn.Sequential()
            self._nin_block(self.linear_ghost, self.num_filters, 2)
            # self.linear_ghost.add(scn.OutputLayer(self.dimension))

        # PPN
        # self.ppn  = PPN(cfg)

        if self.skip_mode == 'default':

            '''
            for p1, p2 in zip(encoder_planes, seg_planes):
                self.seg_skip.add(scn.Identity())
            '''

            for p1, p2 in zip(encoder_planes, cluster_planes):
                self.cluster_skip.add(scn.Identity())
            for p1, p2 in zip(encoder_planes, seed_planes):
                self.seed_skip.add(scn.Identity())
            '''
            self.ppn_transform = scn.Sequential()
            ppn1_num_filters = seg_planes[self.ppn.ppn1_stride-self.ppn._num_strides]
            self._nin_block(self.ppn_transform, encoder_planes[-1], ppn1_num_filters)
            '''

        elif self.skip_mode == '1x1':

            '''
            for p1, p2 in zip(encoder_planes, seg_planes):
                self._nin_block(self.seg_skip, p1, p2)
            '''

            for p1, p2 in zip(encoder_planes, cluster_planes):
                self._nin_block(self.cluster_skip, p1, p2)

            for p1, p2 in zip(encoder_planes, seed_planes):
                self._nin_block(self.seed_skip, p1, p2)

            # self.ppn_transform = scn.Identity()

        else:
            raise ValueError('Invalid skip connection mode!')

        # Freeze Layers
        if self.encoder_freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print('Encoder Freezed')

        '''
        if self.ppn_freeze:
            for p in self.ppn.parameters():
                p.requires_grad = False
            print('PPN Freezed')
        '''

        '''
        if self.segmentation_freeze:
            for p in self.seg_net.parameters():
                p.requires_grad = False
            for p in self.output_segmentation.parameters():
                p.requires_grad = False
            print('Segmentation Branch Freezed')
        '''

        if self.embedding_freeze:
            for p in self.cluster_net.parameters():
                p.requires_grad = False
            for p in self.output_cluster.parameters():
                p.requires_grad = False
            print('Clustering Branch Freezed')

        if self.seediness_freeze:
            for p in self.seed_net.parameters():
                p.requires_grad = False
            for p in self.output_seediness.parameters():
                p.requires_grad = False
            print('Seediness Branch Freezed')

        # Pytorch Activations
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        #print(self)


    def forward(self, primary):
        '''
        primary is list of length minibatch size (assumes mbs = 1)
        primary[0] has 3 spatial coordinates + 1 batch coordinate + 1 feature

        label has shape (point_cloud.shape[0] + 5*num_labels, 1)
        label contains segmentation labels for each point + coords of gt points
        RETURNS:
            - feature_enc: encoder features at each spatial resolution.
            - feature_dec: decoder features at each spatial resolution.
        '''

        point_cloud, = primary
        coords = point_cloud[:, 0:self.dimension+1].float()
        normalized_coords = (coords[:, :self.embedding_dim] - float(self.spatial_size) / 2) \
                    / (float(self.spatial_size) / 2)
        features = point_cloud[:, self.dimension+1].float().view(-1, 1)
        if self.coordConv:
            features = torch.cat([normalized_coords, features], dim=1)
        x = self.input((coords, features))

        encoder_res = self.encoder(x)
        features_enc = encoder_res['features_enc']
        deepest_layer = encoder_res['deepest_layer']

        # print([t.features.shape for t in features_enc])

        '''
        seg_decoder_input = [None]
        for i, layer in enumerate(features_enc[1:]):
            seg_decoder_input.append(self.seg_skip[i](layer))
        deep_seg = self.seg_skip[-1](deepest_layer)
        '''

        seed_decoder_input = [None]
        for i, layer in enumerate(features_enc[1:]):
            seed_decoder_input.append(self.seed_skip[i](layer))
        deep_seed = self.seed_skip[-1](deepest_layer)
        #
        cluster_decoder_input = [None]
        for i, layer in enumerate(features_enc[1:]):
            cluster_decoder_input.append(self.cluster_skip[i](layer))
        deep_cluster = self.cluster_skip[-1](deepest_layer)

        # print([t.features.shape for t in seg_decoder_input[1:]])
        # print([t.features.shape for t in seed_decoder_input[1:]])
        # print([t.features.shape for t in cluster_decoder_input[1:]])

        features_cluster = self.cluster_net(features_enc, deepest_layer)
        features_seediness = self.seed_net(seed_decoder_input, deep_seed)
        # features_seg = self.seg_net(seg_decoder_input, deep_seg)

        # segmentation = features_seg[-1]
        embeddings = features_cluster[-1]
        seediness = features_seediness[-1]

        # features_gnn = self.concat([segmentation, seediness, embeddings])
        # features_gnn = self.output_gnn_features(features_gnn)

        embeddings = self.output_cluster(embeddings)
        embeddings[:, :self.embedding_dim] = self.tanh(embeddings[:, :self.embedding_dim])
        embeddings[:, :self.embedding_dim] += normalized_coords

        res = {}

        '''
        ppn_inputs = {
            'ppn_feature_enc': seg_decoder_input,
            'ppn_feature_dec': [self.ppn_transform(deep_seg)] + features_seg
        }
        '''
    
        '''
        if self.ghost:
            ghost_mask = self.linear_ghost(segmentation)
            res['ghost'] = [ghost_mask.features]
            # ppn_inputs['ghost'] = res['ghost'][0]
        '''


        # print(ppn_inputs['ppn_feature_dec'][-1].features.shape)

        seediness = self.output_seediness(seediness)
        # segmentation = self.output_segmentation(segmentation)

        res.update({
            'embeddings': [embeddings[:, :self.embedding_dim]],
            'margins': [2 * self.sigmoid(embeddings[:, self.embedding_dim:])],
            'seediness': [self.sigmoid(seediness)],
        #    'features_gnn': [features_gnn],
        #    'segmentation': [segmentation],
            'coords': [coords]
        })

        # ppn_res = self.ppn(ppn_inputs)
        # res.update(ppn_res)

        return res
UResNet = UNet


class SegmentationLoss(torch.nn.modules.loss._Loss):
    """
    Loss definition for UResNet.
    For a regular flavor UResNet, it is a cross-entropy loss.
    For deghosting, it depends on a configuration parameter `ghost`:
    - If `ghost=True`, we first compute the cross-entropy loss on the ghost
    point classification (weighted on the fly with sample statistics). Then we
    compute a mask = all non-ghost points (based on true information in label)
    and within this mask, compute a cross-entropy loss for the rest of classes.
    - If `ghost=False`, we compute a N+1-classes cross-entropy loss, where N is
    the number of classes, not counting the ghost point class.
    """
    INPUT_SCHEMA = [
        ["parse_sparse3d_scn", (int,), (3, 1)]
    ]

    def __init__(self, cfg, reduction='sum'):
        super(SegmentationLoss, self).__init__(reduction=reduction)
        self._cfg = cfg['uresnet_lonely']
        self._ghost = self._cfg.get('ghost', False)
        self._ghost_label = self._cfg.get('ghost_label', -1)
        self._num_classes = self._cfg.get('num_classes', 5)
        self._alpha = self._cfg.get('alpha', 1.0)
        self._beta = self._cfg.get('beta', 1.0)
        self._weight_loss = self._cfg.get('weight_loss', False)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')

    def distances(self, v1, v2):
        v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
        return torch.sqrt(torch.pow(v2_2 - v1_2, 2).sum(2))

    def forward(self, result, label, weights=None):
        """
        result[0], label and weight are lists of size #gpus = batch_size.
        segmentation has as many elements as UResNet returns.
        label[0] has shape (N, 1) where N is #pts across minibatch_size events.
        Assumptions
        ===========
        The ghost label is the last one among the classes numbering.
        If ghost = True, then num_classes should not count the ghost class.
        If ghost_label > -1, then we perform only ghost segmentation.
        """
        assert len(result['segmentation']) == len(label)
        batch_ids = [d[:, -2] for d in label]
        uresnet_loss, uresnet_acc = 0., 0.
        uresnet_acc_class = [0.] * self._num_classes
        count_class = [0.] * self._num_classes
        mask_loss, mask_acc = 0., 0.
        ghost2ghost, nonghost2nonghost = 0., 0.
        count = 0
        for i in range(len(label)):
            for b in batch_ids[i].unique():
                batch_index = batch_ids[i] == b

                event_segmentation = result['segmentation'][i][batch_index]  # (N, num_classes)
                event_label = label[i][batch_index][:, -1][:, None]  # (N, 1)
                event_label = torch.squeeze(event_label, dim=-1).long()
                if self._ghost_label > -1:
                    event_label = (event_label == self._ghost_label).long()

                elif self._ghost:
                    # check and warn about invalid labels
                    unique_label,unique_count = torch.unique(event_label,return_counts=True)
                    if (unique_label > self._num_classes).long().sum():
                        print('Invalid semantic label found (will be ignored)')
                        print('Semantic label values:',unique_label)
                        print('Label counts:',unique_count)
                    event_ghost = result['ghost'][i][batch_index]  # (N, 2)
                    # 0 = not a ghost point, 1 = ghost point
                    mask_label = (event_label == self._num_classes).long()
                    # loss_mask = self.cross_entropy(event_ghost, mask_label)
                    num_ghost_points = (mask_label == 1).sum().float()
                    num_nonghost_points = (mask_label == 0).sum().float()
                    fraction = num_ghost_points / (num_ghost_points + num_nonghost_points)
                    weight = torch.stack([fraction, 1. - fraction]).float()
                    loss_mask = torch.nn.functional.cross_entropy(event_ghost, mask_label, weight=weight)
                    mask_loss += loss_mask
                    # mask_loss += torch.mean(loss_mask)

                    # Accuracy of ghost mask: fraction of correcly predicted
                    # points, whether ghost or nonghost
                    with torch.no_grad():
                        predicted_mask = torch.argmax(event_ghost, dim=-1)

                        # Accuracy ghost2ghost = fraction of correcly predicted
                        # ghost points as ghost points
                        if float(num_ghost_points.item()) > 0:
                            ghost2ghost += (predicted_mask[event_label == 5] == 1).sum().item() / float(num_ghost_points.item())

                        # Accuracy noghost2noghost = fraction of correctly predicted
                        # non ghost points as non ghost points
                        if float(num_nonghost_points.item()) > 0:
                            nonghost2nonghost += (predicted_mask[event_label < 5] == 0).sum().item() / float(num_nonghost_points.item())

                        # Global ghost predictions accuracy
                        acc_mask = predicted_mask.eq_(mask_label).sum().item() / float(predicted_mask.nelement())
                        mask_acc += acc_mask

                    # Now mask to compute the rest of UResNet loss
                    mask = event_label < self._num_classes
                    event_segmentation = event_segmentation[mask]
                    event_label = event_label[mask]
                else:
                    # check and warn about invalid labels
                    unique_label,unique_count = torch.unique(event_label,return_counts=True)
                    if (unique_label >= self._num_classes).long().sum():
                        print('Invalid semantic label found (will be ignored)')
                        print('Semantic label values:',unique_label)
                        print('Label counts:',unique_count)
                    # Now mask to compute the rest of UResNet loss
                    mask = event_label < self._num_classes
                    event_segmentation = event_segmentation[mask]
                    event_label = event_label[mask]

                if event_label.shape[0] > 0:  # FIXME how to handle empty mask?
                    # Loss for semantic segmentation
                    if self._weight_loss:
                        class_count = [(event_label == c).sum().float() for c in range(self._num_classes)]
                        w = torch.Tensor([1.0 / c if c.item() > 0 else 0. for c in class_count]).double()
                        #w = torch.Tensor([2.0, 2.0, 5.0, 10.0, 2.0]).double()
                        #w = 1.0 - w / w.sum()
                        if torch.cuda.is_available():
                            w = w.cuda()
                        #print(class_count, w, class_count[0].item() > 0)
                        loss_seg = torch.nn.functional.cross_entropy(event_segmentation, event_label, weight=w.float())
                    else:
                        loss_seg = self.cross_entropy(event_segmentation, event_label)
                        if weights is not None:
                            loss_seg *= weights[i][batch_index][:, -1].float()
                    uresnet_loss += torch.mean(loss_seg)

                    # Accuracy for semantic segmentation
                    with torch.no_grad():
                        predicted_labels = torch.argmax(event_segmentation, dim=-1)
                        acc = predicted_labels.eq_(event_label).sum().item() / float(predicted_labels.nelement())
                        uresnet_acc += acc

                        # Class accuracy
                        for c in range(self._num_classes):
                            class_mask = event_label == c
                            class_count = class_mask.sum().item()
                            if class_count > 0:
                                uresnet_acc_class[c] += predicted_labels[class_mask].sum().item() / float(class_count)
                                count_class[c] += 1

                count += 1

        if self._ghost:
            results = {
                'accuracy': uresnet_acc/count,
                'loss': (self._alpha * uresnet_loss + self._beta * mask_loss)/count,
                'mask_acc': mask_acc / count,
                'mask_loss': self._beta * mask_loss / count,
                'uresnet_loss': self._alpha * uresnet_loss / count,
                'uresnet_acc': uresnet_acc / count,
                'ghost2ghost': ghost2ghost / count,
                'nonghost2nonghost': nonghost2nonghost / count
            }
        else:
            results = {
                'accuracy': uresnet_acc/count,
                'loss': uresnet_loss/count
            }
        for c in range(self._num_classes):
            if count_class[c] > 0:
                results['accuracy_class_%d' % c] = uresnet_acc_class[c]/count_class[c]
            else:
                results['accuracy_class_%d' % c] = -1.
        return results