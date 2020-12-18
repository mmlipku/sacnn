import torch
import torch.nn as nn

class sacnn(nn.Module):

    def __init__(self):
        super(sacnn, self).__init__()

        self.conv_f = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.conv_m1 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.conv_m2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv_m3 = nn.Conv3d(32, 16, kernel_size=3, padding=1)
        self.conv_m4 = nn.Conv3d(16, 16, kernel_size=3, padding=1)
        self.conv_m5 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv_m6 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv_m7 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv_l = nn.Conv3d(64, 1, kernel_size=3, padding=(0,1,1))

        self.attn1 = Self_Attn(32)
        self.attn2 = Self_Attn(16)
        self.attn3 = Self_Attn(32)

        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.relu(self.conv_f(x))
        out = self.relu(self.conv_m1(out))
        out = self.relu(self.conv_m2(out))
        out,p1 = self.attn1(out)

        out = self.relu(self.conv_m3(out))
        out = self.relu(self.conv_m4(out))
        out,p2 = self.attn2(out)

        out = self.relu(self.conv_m5(out))
        out = self.conv_m6(out)
        out,p3 = self.attn3(out)

        out = self.relu(self.conv_m7(out))
        out = self.conv_l(out)
        out = out.squeeze(2)

        #return out,p1,p2,p3
        return out       

class Self_Attn(nn.Module):

    def __init__(self,in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv3d(in_channels = in_dim , out_channels = in_dim , kernel_size= 3, padding=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) 
    def forward(self,x):

        m_batchsize,C,D,width,height = x.size()

        plane_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) 
        plane_key =  self.key_conv(x).view(m_batchsize,-1,width*height) 
        p_energy =  torch.bmm(plane_query,plane_key) 
        p_attention = self.softmax(p_energy) 

        depth_query = self.query_conv(x).view(m_batchsize,-1,D).permute(0,2,1)
        depth_key = self.key_conv(x).view(m_batchsize,-1,D)
        d_energy =  torch.bmm(depth_query,depth_key)
        d_attention = self.softmax(d_energy)

        plane_value = self.value_conv(x).view(m_batchsize,-1,width*height) 
        p_out = torch.bmm(plane_value,p_attention.permute(0,2,1))
        p_out = p_out.view(m_batchsize,C,D,width,height)

        depth_value = self.value_conv(x).view(m_batchsize,-1,D)
        d_out = torch.bmm(depth_value,d_attention.permute(0,2,1))
        d_out = d_out.view(m_batchsize,C,D,width,height)
        
        out = self.gamma*(p_out+d_out) + x
        return out,p_attention

 
