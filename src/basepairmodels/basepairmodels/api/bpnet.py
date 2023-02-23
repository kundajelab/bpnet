from basepairmodels.api.profilemodel import ProfileModel

class BPNet(ProfileModel):
    

    def __init__(self, input_seq_len=3088, output_len=1000, num_tasks=2, 
                 num_bias_profiles=2, filters=64, num_dilation_layers=9, 
                 conv1_kernel_size=21, dilation_kernel_size=3, 
                 profile_kernel_size=25, counts_loss_weight=10):
        
        self.num_bias_profiles = num_bias_profiles
        self.filters = filters
        self.num_dilation_layers = num_dilation_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dilation_kernel_size = dilation_kernel_size
        self.profile_kernel_size = profile_kernel_size
        self.counts_loss_weight = counts_loss_weight
    
        # call base class constructor
        super().__init__(input_seq_len=input_seq_len, output_len=output_len, 
                         num_tasks=num_tasks)
        

        

