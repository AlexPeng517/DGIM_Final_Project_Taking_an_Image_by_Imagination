import torch
from diffusers import ModelMixin, ConfigMixin
from torch import nn 
from torch.nn import functional as F
from diffusers.configuration_utils import register_to_config


class CameraSettingEmbedding(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, embedding_dim=1024, hidden_dim=1024,
                 num_layers=1, activation=None, layer_norm=True,
                 zero_init=True, logize_input=True):
        '''
        Maping the camera setting from EXIF to same dimension as the token
        embedding in CLIP. 
        '''
        super().__init__()

        self.zero_init = zero_init
        self.logize_input = logize_input
        self.activation = activation

        self.embed_focal_length = []
        self.embed_aperture = []
        self.embed_iso_speed = []
        self.embed_exposure_time = []
        for i in range(num_layers):
            if num_layers == 1 and i == 0:
                self.embed_focal_length.append(nn.Linear(1, embedding_dim))
                self.embed_aperture.append(nn.Linear(1, embedding_dim))
                self.embed_iso_speed.append(nn.Linear(1, embedding_dim))
                self.embed_exposure_time.append(nn.Linear(1, embedding_dim))
            elif i == 0:
                self.embed_focal_length.append(nn.Linear(1, hidden_dim))
                self.embed_aperture.append(nn.Linear(1, hidden_dim))
                self.embed_iso_speed.append(nn.Linear(1, hidden_dim))
                self.embed_exposure_time.append(nn.Linear(1, hidden_dim))
            elif i == num_layers - 1:
                self.embed_focal_length.append(nn.Linear(hidden_dim, embedding_dim))
                self.embed_aperture.append(nn.Linear(hidden_dim, embedding_dim))
                self.embed_iso_speed.append(nn.Linear(hidden_dim, embedding_dim))
                self.embed_exposure_time.append(nn.Linear(hidden_dim, embedding_dim))
            else:
                self.embed_focal_length.append(nn.Linear(hidden_dim, hidden_dim))
                self.embed_aperture.append(nn.Linear(hidden_dim, hidden_dim))
                self.embed_iso_speed.append(nn.Linear(hidden_dim, hidden_dim))
                self.embed_exposure_time.append(nn.Linear(hidden_dim, hidden_dim))
            
            if self.zero_init:
                nn.init.zeros_(self.embed_focal_length[-1].weight)
                nn.init.zeros_(self.embed_aperture[-1].weight)
                nn.init.zeros_(self.embed_iso_speed[-1].weight)
                nn.init.zeros_(self.embed_exposure_time[-1].weight)

                nn.init.zeros_(self.embed_focal_length[-1].bias)
                nn.init.zeros_(self.embed_aperture[-1].bias)
                nn.init.zeros_(self.embed_iso_speed[-1].bias)
                nn.init.zeros_(self.embed_exposure_time[-1].bias)

            if layer_norm and i != num_layers - 1:
                self.embed_focal_length.append(nn.LayerNorm(hidden_dim))
                self.embed_aperture.append(nn.LayerNorm(hidden_dim))
                self.embed_iso_speed.append(nn.LayerNorm(hidden_dim))
                self.embed_exposure_time.append(nn.LayerNorm(hidden_dim))
            elif layer_norm and i == num_layers - 1:
                self.embed_focal_length.append(nn.LayerNorm(embedding_dim))
                self.embed_aperture.append(nn.LayerNorm(embedding_dim))
                self.embed_iso_speed.append(nn.LayerNorm(embedding_dim))
                self.embed_exposure_time.append(nn.LayerNorm(embedding_dim))
            
            if i != num_layers - 1 and self.activation is not None:

                if self.activation == 'silu':
                    activation_layer = nn.SiLU()
                elif activation == 'relu':
                    activation_layer = nn.ReLU()
                elif activation == 'gelu':
                    activation_layer = nn.GELU()

                self.embed_focal_length.append(activation_layer)
                self.embed_aperture.append(activation_layer)
                self.embed_iso_speed.append(activation_layer)
                self.embed_exposure_time.append(activation_layer)

        self.embed_focal_length = nn.Sequential(*self.embed_focal_length)
        self.embed_aperture = nn.Sequential(*self.embed_aperture)
        self.embed_iso_speed = nn.Sequential(*self.embed_iso_speed)
        self.embed_exposure_time = nn.Sequential(*self.embed_exposure_time)    
            
    def focal_length_forward(self, x_focal_length):
        if self.logize_input:
            x_focal_length = torch.log(x_focal_length + 1e-6)
        y_focal_length = self.embed_focal_length(x_focal_length).unsqueeze(1)
        return y_focal_length
    
    def aperture_forward(self, x_aperture):
        if self.logize_input:
            x_aperture = torch.log(x_aperture + 1e-6)
        y_aperture = self.embed_aperture(x_aperture).unsqueeze(1)
        return y_aperture
    
    def iso_speed_forward(self, x_iso_speed):
        if self.logize_input:
            x_iso_speed = torch.log(x_iso_speed + 1e-6)
        y_iso_speed = self.embed_iso_speed(x_iso_speed).unsqueeze(1)
        return y_iso_speed
    
    def exposure_time_forward(self, x_exposure_time):
        if self.logize_input:
            x_exposure_time = torch.log(x_exposure_time + 1e-6)
        y_exposure_time = self.embed_exposure_time(x_exposure_time).unsqueeze(1)
        return y_exposure_time

        
    def forward(self, x_focal_length, x_aperture, x_iso_speed, x_exposure_time):

        if self.logize_input:
            x_focal_length = torch.log(x_focal_length + 1e-6)
            x_aperture = torch.log(x_aperture + 1e-6)
            x_iso_speed = torch.log(x_iso_speed + 1e-6)
            x_exposure_time = torch.log(x_exposure_time + 1e-6)

        y_focal_length = self.embed_focal_length(x_focal_length).unsqueeze(1)
        y_aperture = self.embed_aperture(x_aperture).unsqueeze(1)
        y_iso_speed = self.embed_iso_speed(x_iso_speed).unsqueeze(1)
        y_exposure_time = self.embed_exposure_time(x_exposure_time).unsqueeze(1)
        
        y = torch.cat([y_focal_length, y_aperture, y_iso_speed, y_exposure_time], dim=1)
        return y
    
