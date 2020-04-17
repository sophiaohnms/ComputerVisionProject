import torch
from math import sqrt
from itertools import product


class PriorBox:
    def __init__(self, cfg):
        self.image_size = cfg.INPUT.IMAGE_SIZE
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides = prior_config.STRIDES
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        self.clip = prior_config.CLIP

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        
        #*CHANGE THE COMPUTATION OF THE SCALE PARAMETER IN BOTH X AND Y, AND THE W/H 
       # print("PRINTAS: ", enumerate(self.feature_maps))
        
        priors = []
        
        
        for k, f in enumerate(self.feature_maps):
            print("Feature map: ", f )
            #print(" K: ", k )
            #print("Image size: ", [self.image_size[0], self.image_size[1]])
            print("Stride: ", [self.strides[k][0], self.strides[k][1]])
            
            scaleY = self.image_size[1] / self.strides[k][1] #Separate scale factor in axis
            scaleX = self.image_size[0] / self.strides[k][0]
            
            print("Scale: ", [scaleX, scaleY])
            #scale = self.image_size / self.strides[k]
            
            for i in range(int(scaleY)):
                #print("i : ",  i)
                for j in range(int(scaleX)):
                    cx = (j + 0.5) / scaleX
                    cy = (i + 0.5) / scaleY
                    
                    # small sized square box
                    size = self.min_sizes[k]
                    h = size / self.image_size[1]
                    w = size  / self.image_size[0]
               
                    priors.append([cx, cy, w, h])

                    # big sized square box
                    size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                    h = size / self.image_size[1]
                    w = size  / self.image_size[0]
                
                    priors.append([cx, cy, w, h])

                    # change h/w ratio of the small sized box
                    size = self.min_sizes[k]
                    h = size / self.image_size[1]
                    w = size  / self.image_size[0]
              
                    for ratio in self.aspect_ratios[k]:
                        ratio = sqrt(ratio)
                        priors.append([cx, cy, w * ratio, h / ratio])
                        priors.append([cx, cy, w / ratio, h * ratio])
                    
        """
            for i, j in product(range(f[1]), range(f[0])):
               # unit center x,y
                #print("I: ", i)
                #print("J: ", j)
                
                cx = (j + 0.5) / scaleX
                cy = (i + 0.5) / scaleY

                # small sized square box
                size = self.min_sizes[k]
                h = size / self.image_size[1]
                w = size  / self.image_size[0]
               
                priors.append([cx, cy, w, h])

                # big sized square box
                size = sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = size / self.image_size[1]
                w = size  / self.image_size[0]
                
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                h = size / self.image_size[1]
                w = size  / self.image_size[0]
              
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])
        """
                    
                    
        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
