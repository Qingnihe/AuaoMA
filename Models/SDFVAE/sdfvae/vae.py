import torch
import torch.nn as nn
from data_config import *

class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnit, self).__init__()
        self.model = nn.Sequential(
                     nn.Conv2d(in_channels, out_channels, kernel, stride, padding), nonlinearity)
    def forward(self, x):
        return self.model(x)

class ConvUnitTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, out_padding=0, nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnitTranspose, self).__init__()
        if nonlinearity == 'none':
            self.model = nn.Sequential(
                     nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding),
                     )
        else:
            self.model = nn.Sequential(
                        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding), 
                        nonlinearity
                        )
    def forward(self, x):
        return self.model(x)

class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        self.model = nn.Sequential(
                     nn.Linear(in_features, out_features),nonlinearity)
    def forward(self, x):
        return self.model(x)


class VAE(nn.Module):
    def __init__(self, s_dim=8, d_dim=10, conv_dim=100, hidden_dim=40,
                 T=20, w=36, n=24, enc_dec='CNN', nonlinearity=None):
        super(VAE, self).__init__()
        
        self.s_dim = s_dim
        self.d_dim = d_dim
        self.conv_dim = conv_dim
        self.hidden_dim = hidden_dim
        self.T = T
        self.w = w
        self.n = n
        self.enc_dec = enc_dec
        self.device = global_device
        self.nonlinearity = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        self.dec_init_dim = self.s_dim+self.d_dim+self.hidden_dim 

        self.d_lstm_prior = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.d_mean_prior = nn.Linear(self.hidden_dim, self.d_dim)
        self.d_logvar_prior = nn.Linear(self.hidden_dim, self.d_dim)
        self.phi_d_prior = nn.Sequential(
                           nn.Linear(self.d_dim, self.hidden_dim),
                           self.nonlinearity)

        self.enc_d_prior = nn.Sequential(
                           nn.Linear(self.hidden_dim, self.hidden_dim),
                           self.nonlinearity,
                           nn.Linear(self.hidden_dim, self.hidden_dim))

        # Bidirectional LSTM with option bidirectional=True
        self.s_lstm = nn.LSTM(self.conv_dim, self.hidden_dim,
                                  1, batch_first=True, bidirectional=True)

        self.s_mean = LinearUnit(self.hidden_dim*2, self.s_dim)
        self.s_logvar = LinearUnit(self.hidden_dim*2, self.s_dim)

        self.phi_conv = nn.Sequential(
                     nn.Linear(self.conv_dim, self.hidden_dim),
                     self.nonlinearity,
                     nn.Linear(self.hidden_dim, self.hidden_dim),
                     self.nonlinearity)

        self.phi_d = nn.Sequential(
                     nn.Linear(self.d_dim, self.hidden_dim),
                     self.nonlinearity)

        self.enc_d = nn.Sequential(
                   nn.Linear(2*self.hidden_dim, self.hidden_dim),
                   self.nonlinearity,
                   nn.Linear(self.hidden_dim, self.hidden_dim),
                   self.nonlinearity)

        self.d_mean = nn.Linear(self.hidden_dim, self.d_dim)
        self.d_logvar = nn.Linear(self.hidden_dim, self.d_dim)

        self.d_rnn = nn.LSTMCell(2*self.hidden_dim, self.hidden_dim, bias=True)
         
        if self.enc_dec == 'CNN':
            '''
            if self.n == 16 or self.n == 24:
                k0_0, s0_0, p0_0 = 2, 2, 0
                k0_1, s0_1, p0_1 = 2, 2, 0
                k0_2, s0_2, p0_2 = 2, 2, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            elif self.n == 38:
                k0_0, s0_0, p0_0 = 2, 2, 1
                k0_1, s0_1, p0_1 = 2, 2, 0
                k0_2, s0_2, p0_2 = 2, 2, 0
                sd_0 = int(((s0_0 * p0_0) + self.n) / (k0_0 * k0_1 * k0_2))
            elif self.n == 48:
                k0_0, s0_0, p0_0 = 4, 4, 0
                k0_1, s0_1, p0_1 = 2, 2, 0
                k0_2, s0_2, p0_2 = 2, 2, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            elif self.n == 19:  # ASD
                k0_0, s0_0, p0_0 = 2, 1, 0
                k0_1, s0_1, p0_1 = 3, 3, 0
                k0_2, s0_2, p0_2 = 2, 2, 0
                sd_0 = int((5 + self.n) / (k0_0 * k0_1 * k0_2))
            elif self.n == 51:  # SWAT
                k0_0, s0_0, p0_0 = 5, 2, 0
                k0_1, s0_1, p0_1 = 3, 3, 0
                k0_2, s0_2, p0_2 = 2, 2, 0
                sd_0 = int((5 + self.n) / (k0_0 * k0_1 * k0_2))
            elif self.n == 55:  # MSL
                k0_0, s0_0, p0_0 = 5, 5, 0
                k0_1, s0_1, p0_1 = 2, 1, 0
                k0_2, s0_2, p0_2 = 2, 2, 0
                sd_0 = int((1 + self.n) / (k0_0 * k0_1 * k0_2))
            elif self.n == 25:  # SMAP
                k0_0, s0_0, p0_0 = 2, 1, 0
                k0_1, s0_1, p0_1 = 3, 3, 0
                k0_2, s0_2, p0_2 = 2, 2, 0
                sd_0 = int(((s0_0 * p0_0) + self.n) / (k0_0 * k0_1 * k0_2))
            elif self.n == 123:  # WADI
                k0_0, s0_0, p0_0 = 3, 3, 0
                k0_1, s0_1, p0_1 = 6, 5, 0
                k0_2, s0_2, p0_2 = 2, 2, 0
                sd_0 = int((5 + self.n) / (k0_0 * k0_1 * k0_2))
            '''
            if self.n == 24:
                k0_0, s0_0, p0_0, op0_0 = 2, 2, 0, 0
                k0_1, s0_1, p0_1, op0_1 = 2, 2, 0, 0
                k0_2, s0_2, p0_2, op0_2 = 2, 2, 0, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            elif self.n == 16:
                k0_0, s0_0, p0_0, op0_0 = 2, 2, 0, 0
                k0_1, s0_1, p0_1, op0_1 = 2, 2, 0, 0
                k0_2, s0_2, p0_2, op0_2 = 2, 2, 0, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            elif self.n == 19:  # ASD
                k0_0, s0_0, p0_0, op0_0 = 2, 2, 0, 1
                k0_1, s0_1, p0_1, op0_1 = 2, 2, 0, 1
                k0_2, s0_2, p0_2, op0_2 = 2, 2, 0, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            elif self.n == 38:  # SMD or yidong-38
                k0_0, s0_0, p0_0, op0_0 = 2, 2, 0, 0
                k0_1, s0_1, p0_1, op0_1 = 2, 2, 0, 1
                k0_2, s0_2, p0_2, op0_2 = 3, 3, 0, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
                # sd_0=int(((s0_0*p0_0)+self.n)/(k0_0*k0_1*k0_2))
            elif self.n == 48:
                k0_0, s0_0, p0_0, op0_0 = 4, 4, 0, 0
                k0_1, s0_1, p0_1, op0_1 = 2, 2, 0, 0
                k0_2, s0_2, p0_2, op0_2 = 2, 2, 0, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            elif self.n == 49:  # CTF
                k0_0, s0_0, p0_0, op0_0 = 4, 4, 0, 1
                k0_1, s0_1, p0_1, op0_1 = 2, 2, 0, 0
                k0_2, s0_2, p0_2, op0_2 = 2, 2, 0, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            elif self.n == 51:  # SWAT
                k0_0, s0_0, p0_0, op0_0 = 3, 3, 0, 0
                k0_1, s0_1, p0_1, op0_1 = 4, 4, 0, 1
                k0_2, s0_2, p0_2, op0_2 = 2, 2, 0, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            elif self.n == 55:  # MSL
                k0_0, s0_0, p0_0, op0_0 = 3, 3, 0, 1
                k0_1, s0_1, p0_1, op0_1 = 3, 3, 0, 0
                k0_2, s0_2, p0_2, op0_2 = 2, 2, 0, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            elif self.n == 25:  # SMAP
                k0_0, s0_0, p0_0, op0_0 = 2, 2, 0, 1
                k0_1, s0_1, p0_1, op0_1 = 2, 2, 0, 0
                k0_2, s0_2, p0_2, op0_2 = 2, 2, 0, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            elif self.n == 123:  # WADI
                k0_0, s0_0, p0_0, op0_0 = 2, 2, 0, 1
                k0_1, s0_1, p0_1, op0_1 = 4, 4, 0, 1
                k0_2, s0_2, p0_2, op0_2 = 3, 3, 0, 0
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            elif self.n == 22:  # yidong-22
                k0_0, s0_0, p0_0, op0_0 = 2, 2, 0, 0
                k0_1, s0_1, p0_1, op0_1 = 2, 2, 0, 1
                k0_2, s0_2, p0_2, op0_2 = 2, 2, 0, 1
                sd_0 = int(self.n / (k0_0 * k0_1 * k0_2))
            else:
                raise ValueError('Invalid Kpi numbers: {}, choose from the candidate set [16,22,24,38,48].'.format(self.n))
            '''
            if self.w == 36:
                k1_0, s1_0, p1_0 = 3, 3, 0
                k1_1, s1_1, p1_1 = 2, 2, 0
                k1_2, s1_2, p1_2 = 2, 2, 0
                sd_1 = int(self.w / (k1_0 * k1_1 * k1_2))
            elif self.w == 144:
                k1_0, s1_0, p1_0 = 4, 4, 0
                k1_1, s1_1, p1_1 = 4, 4, 0
                k1_2, s1_2, p1_2 = 3, 3, 0
                sd_1 = int(self.w / (k1_0 * k1_1 * k1_2))
            elif self.w == 288:
                k1_0, s1_0, p1_0 = 8, 8, 0
                k1_1, s1_1, p1_1 = 4, 4, 0
                k1_2, s1_2, p1_2 = 3, 3, 0
                sd_1 = int(self.w / (k1_0 * k1_1 * k1_2))
            '''
            if self.w == 36:
                k1_0, s1_0, p1_0, op1_0 = 3, 3, 0, 0
                k1_1, s1_1, p1_1, op1_1 = 2, 2, 0, 0
                k1_2, s1_2, p1_2, op1_2 = 2, 2, 0, 0
                sd_1 = int(self.w / (k1_0 * k1_1 * k1_2))
            elif self.w == 144:
                k1_0, s1_0, p1_0, op1_0 = 4, 4, 0, 0
                k1_1, s1_1, p1_1, op1_1 = 4, 4, 0, 0
                k1_2, s1_2, p1_2, op1_2 = 3, 3, 0, 0
                sd_1 = int(self.w / (k1_0 * k1_1 * k1_2))
            elif self.w == 288:
                k1_0, s1_0, p1_0, op1_0 = 8, 8, 0, 0
                k1_1, s1_1, p1_1, op1_1 = 4, 4, 0, 0
                k1_2, s1_2, p1_2, op1_2 = 3, 3, 0, 0
                sd_1 = int(self.w / (k1_0 * k1_1 * k1_2))
            elif self.w == 60:
                k1_0,s1_0,p1_0,op1_0=8,2,4,0
                k1_1,s1_1,p1_1,op1_1=8,2,4,1
                k1_2,s1_2,p1_2,op1_2=8,2,4,0
                sd_1=9
            else:
                raise ValueError('Invalid window size: {}, choose from the set [36,144,288,60]'.format(self.w))

            self.krl = [[k0_0, k1_0], [k0_1, k1_1], [k0_2, k1_2]]
            self.srd = [[s0_0, s1_0], [s0_1, s1_1], [s0_2, s1_2]]
            self.pd = [[p0_0, p1_0], [p0_1, p1_1], [p0_2, p1_2]]
            self.opd = [[op0_0, op1_0], [op0_1, op1_1], [op0_2, op1_2]]
            self.cd = [64, sd_0, sd_1]

            self.conv = nn.Sequential(
                ConvUnit(1, 8, kernel=(self.krl[0][0], self.krl[0][1]),
                         stride=(self.srd[0][0], self.srd[0][1]),
                         padding=(self.pd[0][0], self.pd[0][1])),
                ConvUnit(8, 32, kernel=(self.krl[1][0], self.krl[1][1]),
                         stride=(self.srd[1][0], self.srd[1][1]),
                         padding=(self.pd[1][0], self.pd[1][1])),
                ConvUnit(32, 64, kernel=(self.krl[2][0], self.krl[2][1]),
                         stride=(self.srd[2][0], self.srd[2][1]),
                         padding=(self.pd[2][0], self.pd[2][1]))
            )

            self.conv_fc = nn.Sequential(
                LinearUnit(self.cd[0] * self.cd[1] * self.cd[2], self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.conv_dim))

            self.deconv_fc_mu = nn.Sequential(
                LinearUnit(self.dec_init_dim, self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.cd[0] * self.cd[1] * self.cd[2]))
            '''
            self.deconv_mu = nn.Sequential(
                ConvUnitTranspose(64, 32, kernel=(self.krl[2][0], self.krl[2][1]),
                                  stride=(self.srd[2][0], self.srd[2][1]),
                                  padding=(self.pd[2][0], self.pd[2][1])),
                ConvUnitTranspose(32, 8, kernel=(self.krl[1][0], self.krl[1][1]),
                                  stride=(self.srd[1][0], self.srd[1][1]),
                                  padding=(self.pd[1][0], self.pd[1][1])),
                ConvUnitTranspose(8, 1, kernel=(self.krl[0][0], self.krl[0][1]),
                                  stride=(self.srd[0][0], self.srd[0][1]),
                                  padding=(self.pd[0][0], self.pd[0][1]),
                                  nonlinearity=nn.Tanh())
            )
            self.deconv_fc_logsigma = nn.Sequential(
                LinearUnit(self.dec_init_dim, self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.cd[0] * self.cd[1] * self.cd[2]))
            self.deconv_logsigma = nn.Sequential(
                ConvUnitTranspose(64, 32, kernel=(self.krl[2][0], self.krl[2][1]),
                                  stride=(self.srd[2][0], self.srd[2][1]),
                                  padding=(self.pd[2][0], self.pd[2][1])),
                ConvUnitTranspose(32, 8, kernel=(self.krl[1][0], self.krl[1][1]),
                                  stride=(self.srd[1][0], self.srd[1][1]),
                                  padding=(self.pd[1][0], self.pd[1][1])),
                ConvUnitTranspose(8, 1, kernel=(self.krl[0][0], self.krl[0][1]),
                                  stride=(self.srd[0][0], self.srd[0][1]),
                                  padding=(self.pd[0][0], self.pd[0][1]),
                                  nonlinearity=nn.Tanh())
            )
            '''
            self.deconv_mu = nn.Sequential(
                ConvUnitTranspose(64, 32, kernel=(self.krl[2][0], self.krl[2][1]),
                                  stride=(self.srd[2][0], self.srd[2][1]),
                                  out_padding=(self.opd[2][0], self.opd[2][1]),
                                  padding=(self.pd[2][0], self.pd[2][1])
                                  ),
                ConvUnitTranspose(32, 8, kernel=(self.krl[1][0], self.krl[1][1]),
                                  stride=(self.srd[1][0], self.srd[1][1]),
                                  out_padding=(self.opd[1][0], self.opd[1][1]),
                                  padding=(self.pd[1][0], self.pd[1][1])
                                  ),
                ConvUnitTranspose(8, 1, kernel=(self.krl[0][0], self.krl[0][1]),
                                  stride=(self.srd[0][0], self.srd[0][1]),
                                  out_padding=(self.opd[0][0], self.opd[0][1]),
                                  padding=(self.pd[0][0], self.pd[0][1]),
                                  nonlinearity=nn.Tanh())
            )
            self.deconv_fc_logsigma = nn.Sequential(
                LinearUnit(self.dec_init_dim, self.conv_dim * 2),
                LinearUnit(self.conv_dim * 2, self.cd[0] * self.cd[1] * self.cd[2]))
            self.deconv_logsigma = nn.Sequential(
                ConvUnitTranspose(64, 32, kernel=(self.krl[2][0], self.krl[2][1]),
                                  stride=(self.srd[2][0], self.srd[2][1]),
                                  out_padding=(self.opd[2][0], self.opd[2][1]),
                                  padding=(self.pd[2][0], self.pd[2][1])),
                ConvUnitTranspose(32, 8, kernel=(self.krl[1][0], self.krl[1][1]),
                                  stride=(self.srd[1][0], self.srd[1][1]),
                                  out_padding=(self.opd[1][0], self.opd[1][1]),
                                  padding=(self.pd[1][0], self.pd[1][1])),
                ConvUnitTranspose(8, 1, kernel=(self.krl[0][0], self.krl[0][1]),
                                  stride=(self.srd[0][0], self.srd[0][1]),
                                  out_padding=(self.opd[0][0], self.opd[0][1]),
                                  padding=(self.pd[0][0], self.pd[0][1]),
                                  nonlinearity=nn.Tanh())
            )
       
        else:
            raise ValueError('Unknown encoder and decoder: {}'.format(self.enc_dec))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d) or  isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)

    # Time-dependent prior distribution of dynamic latent variables 
    def sample_d_lstmcell(self, batch_size, random_sampling=True):
        d_out = None
        d_means = None
        d_logvars = None
            
        d_t = torch.zeros(batch_size, self.d_dim, device=self.device)
        # Here we assume that p(d_0) = N(0,I), thus d_mean_0 = 0, d_logvar_0 = 0 due to log1 = 0
        d_mean_t = torch.zeros(batch_size, self.d_dim, device=self.device)
        d_logvar_t = torch.zeros(batch_size, self.d_dim, device=self.device)
        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)
            
        for _ in range(self.T):
            '''
            When t = 1:
            # According to Figure 5(a), in the beginning, we use the information hidden in h_0 (no any information) to get d_1
            # Here d_mean_1 and d_logvar_1 are still 0, due to h_0 is 0, thus prior p(d_1|d_0) = N(0,I)
            # So we sample d_1 from N(0, I) based on reparameterization trick
            # Next, we update h_1 by using Eq. (2), h1 = r_p(h_0, d_1), also see Figure 5(a)
            
            When t = 2:
            # Here d_2 ~ p(d_2|h_1), since h1 = r_p(h_0, d_1), d_2 ~ p(d_2|d_1), we still sample d_2 based on reparameterization trick
            # It should be noted that p(d_2|d_1) is not N(0,I), due to d_mean_2 and d_logvar_2 are no longer 0,
            # but parameterized by NNs.
            # Then update h_2 by using Eq. (2), h2 = r_p(h_1, d_2),
            # So we construct the time-dependent prior of latent variable d
            
            When t = 3:
            ...
            '''
            enc_d_t = self.enc_d_prior(h_t)
            d_mean_t = self.d_mean_prior(enc_d_t)
            d_logvar_t = self.d_logvar_prior(enc_d_t)
            d_t = self.reparameterize(d_mean_t, d_logvar_t, random_sampling)
            phi_d_t = self.phi_d_prior(d_t)
            h_t, c_t = self.d_lstm_prior(phi_d_t, (h_t, c_t))
            if d_out is None:
                d_out = d_t.unsqueeze(1)
                d_means = d_mean_t.unsqueeze(1)
                d_logvars = d_logvar_t.unsqueeze(1)
            else:
                d_out = torch.cat((d_out, d_t.unsqueeze(1)), dim=1)
                d_means = torch.cat((d_means, d_mean_t.unsqueeze(1)), dim=1)
                d_logvars = torch.cat((d_logvars, d_logvar_t.unsqueeze(1)), dim=1)
        d_logvars = torch.clamp(d_logvars, min=min_log_sigma)
        return d_means, d_logvars, d_out



    def encoder_x(self, x):
        if self.enc_dec == 'CNN':
            # print(x.shape)
            # breakpoint()
            x = x.view(-1, 1, self.n, self.w)
            x = self.conv(x)
            x = x.view(-1, self.cd[0]*self.cd[1]*self.cd[2])
            x = self.conv_fc(x)
            # print(f'after conv_fc {x.shape}')
            x = x.view(-1, self.T, self.conv_dim)
        else:
            raise ValueError('Unknown encoder and decoder: {}'.format(self.enc_dec))
        return x

    def decoder_mu(self, sdh):
        if self.enc_dec == 'CNN':
            # print(f"cd:{self.cd}")
            x = self.deconv_fc_mu(sdh)
            # print(f"1--x.shape:{x.shape}")
            x = x.view(-1, self.cd[0], self.cd[1], self.cd[2])
            # print(f"2--x.shape:{x.shape}")
            x = self.deconv_mu(x)
            # print(f"3--x.shape:{x.shape}")
            # breakpoint()
            x = x.view(-1, self.T, 1, self.n, self.w)
        else:
            raise ValueError('Unknown encoder and decoder: {}'.format(self.enc_dec))
        return x
    
 
    def decoder_logsigma(self, sdh):
        if self.enc_dec == 'CNN':
            x = self.deconv_fc_logsigma(sdh)
            x = x.view(-1, self.cd[0], self.cd[1], self.cd[2])
            x = self.deconv_logsigma(x)
            # print(f"decoder_logsigma x:{x.shape}")
            x = x.view(-1, self.T, 1, self.n, self.w)
            x = torch.clamp(x, min_log_sigma)
        else:
            raise ValueError('Unknown encoder and decoder: {}'.format(self.enc_dec))
        return x

    # See https://arxiv.org/pdf/1606.05908.pdf, Page 11, Equation (10) for details.
    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5*logvar)
            z = mean + eps*std
            return z
        else:
            return mean
        
    def encode_s(self, x):
        lstm_out, _ = self.s_lstm(x)
        # print(f"encode_s lstm_out:{lstm_out.shape} hidden: {len(_)} {_[0].shape} {_[1].shape}")
        # backward = lstm_out[:, 0, self.hidden_dim:self.hidden_dim*2]
        # frontal = lstm_out[:, 1, 0:self.hidden_dim]
        # lstm_out = torch.cat((frontal, backward),dim=1)
        lstm_out = torch.mean(lstm_out, dim=1)
        # print(f"encode_s cat lstm_out:{lstm_out.shape}")
        mean = self.s_mean(lstm_out)
        logvar = self.s_logvar(lstm_out)
        s = self.reparameterize(mean, logvar, self.training)
        logvar = torch.clamp(logvar, min_log_sigma)
        return mean, logvar, s
    
    
    def encode_d(self, batch_size, x):
        d_out = None
        d_means = None
        d_logvars = None
        h_out = None

        d_t = torch.zeros(batch_size, self.d_dim, device=self.device) 
        d_mean_t = torch.zeros(batch_size, self.d_dim, device=self.device) 
        d_logvar_t = torch.zeros(batch_size, self.d_dim, device=self.device) 
        h_t = torch.zeros(batch_size, self.hidden_dim, device=self.device) 
        c_t = torch.zeros(batch_size, self.hidden_dim, device=self.device)

        for t in range(self.T):
            '''
            Note: the following t in the comments denotes the index of x_t, not the t in the loop
            When t = 1:
            # (1) d_1 ~ p(d_1|d_<1,x_=<1), Eq. (10) (sample d_1 based on reparameterization trick)
            # (2) h_1 = r(h_0, d_1, x_1), Eq. (3)
            
            When t = 2:
            # (1) d_2 ~ p(d_2|h_1,x_2), thus d_2 ~ p(d_2|d_<2,x_=<2) Eq. (10) (sample d_2 based on reparameterization trick)
            # (2) h_2 = r(h_1, d_2, x_2), Eq. (3)
            
            When t = 3:
            ...
            '''
            phi_conv_t = self.phi_conv(x[:,t,:]) 
            # print(f"x[:,t,:].shape:{x[:,t,:].shape}----phi_conv_t.shape:{phi_conv_t.shape}----h_t.shape:{h_t.shape}")
            enc_d_t = self.enc_d(torch.cat([phi_conv_t, h_t], 1)) 
            d_mean_t = self.d_mean(enc_d_t) 
            d_logvar_t = self.d_logvar(enc_d_t) 
            d_t = self.reparameterize(d_mean_t, d_logvar_t, self.training) 
            phi_d_t = self.phi_d(d_t) 
            if d_out is None:
                d_out = d_t.unsqueeze(1) 
                d_means = d_mean_t.unsqueeze(1) 
                d_logvars = d_logvar_t.unsqueeze(1) 
                h_out = h_t.unsqueeze(1) 
            else:
                d_out = torch.cat((d_out, d_t.unsqueeze(1)), dim=1) 
                d_means = torch.cat((d_means, d_mean_t.unsqueeze(1)), dim=1) 
                d_logvars = torch.cat((d_logvars, d_logvar_t.unsqueeze(1)), dim=1) 
                h_out = torch.cat((h_out, h_t.unsqueeze(1)), dim=1)
            # We update h_t here, because we want to get the tupples like (h_0, d_1), (h_1, d_2), i.e., (h_(t-1), d_t)
            h_t, c_t = self.d_rnn(torch.cat([phi_conv_t, phi_d_t], 1), (h_t, c_t))
        d_logvars = torch.clamp(d_logvars, min_log_sigma)
        return d_means, d_logvars, d_out, h_out
 
    def forward(self, x):
        x = x.float()
        d_mean_prior, d_logvar_prior, _ = self.sample_d_lstmcell(x.size(0), random_sampling = self.training)
        # print(f"x.shape:{x.shape}")
        x_hat = self.encoder_x(x)
        # print(f"x_hat.shape:{x_hat.shape}")
        d_mean, d_logvar, d, h = self.encode_d(x.size(0), x_hat) 
        s_mean, s_logvar, s = self.encode_s(x_hat)
        s_expand = s.unsqueeze(1).expand(-1, self.T, self.s_dim)
        # We concat [d_t, h_(t-1), s] and feed it to decoder network
        ds = torch.cat((d, s_expand), dim=2)
        dsh = torch.cat((ds, h), dim=2)
        # print(f"dsh.shape:{dsh.shape} d.shape:{d.shape}  s_expand.shape:{s_expand.shape} h.shape:{h.shape}")
        recon_x_mu = self.decoder_mu(dsh)
        recon_x_logsigma = self.decoder_logsigma(dsh)
        # print(f"recon_x_mu:{recon_x_mu.shape} recon_x_logsigma:{recon_x_logsigma.shape}")
        recon_x_mu = torch.squeeze(recon_x_mu)
        # .permute(0, 1, 3, 2)
        recon_x_logsigma = torch.squeeze(recon_x_logsigma)
        # .permute(0, 1, 3, 2)        
        return s_mean, s_logvar, s, d_mean, d_logvar, d, d_mean_prior, d_logvar_prior, recon_x_mu, recon_x_logsigma
    