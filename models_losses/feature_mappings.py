import torch
import sparseconvnet as scn

class PoolFeatureMapping(torch.nn.Module):
    '''Operation for non-parametric mapping of features of sparse tensor a to sparse tensor b's 
    active sites
    '''
    def __init__(self, dimension=3, pool_size=2, pool_stride=2):
        super().__init__()
        self.pooling = scn.MaxPooling(dimension, pool_size, pool_stride)
        self.unpooling = scn.UnPooling(dimension, pool_size, pool_stride)

    def forward(self, a, b):
        '''Maps features of `scn.SparseConvNetTensor` a to active sites of b using maxpooling
        and unpooling operations

        args:
            a: `scn.SparseConvNetTensor` with features to map
            b: `scn.SparseConvNetTensor` with active sites to map to
        '''
        assert torch.all(a.spatial_size == b.spatial_size), 'Inputs must have same spatial size'

        x = self.pooling(a)
        x.metadata = b.metadata
        x = self.unpooling(x)

        return x

class ConvolutionalFeatureMapping(torch.nn.Module):
    '''Operation for learnable mapping of features of sparse tensor to sparse tensor b's
    active sites
    '''

    def __init__(self, dimension, n_in, n_out, filter_size, filter_stride, bias):
        super().__init__()
        self.convolution = scn.Convolution(dimension, n_in, n_out, filter_size, filter_stride, bias)
        self.deconvolution = scn.Deconvolution(dimension, n_in, n_out, filter_size, filter_stride, bias)

    def forward(self, a, b):
        '''Maps features of a to active sites of b using convolutional operations

        args:
            a: `scn.SparseConvNetTensor` with features to map
            b: `scn.SparseConvNetTensor` with active sites to map to
        '''
        assert torch.all(a.spatial_size == b.spatial_size), 'Inputs must have same spatial size'
        assert a.features.shape[1] == self.convolution.nIn, 'Incorrect number of input channels'

        x = self.convolution(a)
        x.metadata = b.metadata
        x = self.deconvolution(x)

        return x