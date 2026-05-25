
import torch.nn as nn
import sys
import torch
import torch.nn.functional as F
import kornia
import torchvision.models as models

# Don't generate pyc codes
sys.dont_write_bytecode = True

def LossFn(delta, labels):

    loss = F.l1_loss(delta, labels, reduction='mean')
    return loss

class ResNetHomography(nn.Module):
    def __init__(self):
        super(ResNetHomography, self).__init__()
        try:
            self.resnet = models.resnet18(weights=None)
        except TypeError:
            self.resnet = models.resnet18(pretrained=False)
            
        self.resnet.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.dropout = nn.Dropout(0.5)
        self.resnet.fc = nn.Linear(512, 8)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.resnet.fc(x)
        return x

class HomographyModel(nn.Module):
    def __init__(self, backbone="ConvNet"):
        super(HomographyModel, self).__init__()
        self.backbone_name = backbone
        if backbone.lower() == "resnet18":
            self.model = ResNetHomography()
        else:
            self.model = UnsuperNet()

    def forward(self, I1, CoordinateBatch, Ca, Cb, Pa):
        if self.backbone_name.lower() == "resnet18":
            H4pt_predict = self.model(I1)
            out = Tensor_DLT(H4pt_predict, Ca)
            batch_size = I1.shape[0]
            Pa = Pa.view(batch_size, 1, 128, 128)
            Pa = Pa.float()
            out = out.float()
            PB_pred = kornia.geometry.transform.warp_perspective(Pa, out, dsize=(128,128),
                                                                mode='bilinear', padding_mode='zeros', 
                                                                align_corners=True).requires_grad_()
            return PB_pred, H4pt_predict
        else:
            return self.model(I1, CoordinateBatch, Ca, Cb, Pa)

    def training_step(self, batch, labels):
        # img_a, patch_a, patch_b, corners, gt = batch
        delta = self.model(batch)
        loss = LossFn(delta, labels)
        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, VI1, VCoordinateBatch, VCa, VCb, VPa, VPb): 
        # img_a, patch_a, patch_b, corners, gt = batch
        PbPredicted, H4ptPrecicted = self.model(VI1, VCoordinateBatch, VCa, VCb, VPa)
        loss = LossFn(PbPredicted, VPb)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([out["val_loss"] for out in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}

class UnsuperNet(nn.Module):
    def __init__(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        
        # Spatial transformer localization-network
        self.conv1 = nn.Sequential(nn.Conv2d(2, 64, kernel_size=3),nn.BatchNorm2d(64),nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3),nn.BatchNorm2d(64),nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3),nn.BatchNorm2d(64),nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3),nn.BatchNorm2d(64),nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3),nn.BatchNorm2d(128),nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3),nn.BatchNorm2d(128),nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3),nn.BatchNorm2d(128),nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3),nn.BatchNorm2d(128),nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.ReLU(True)
        )
        self.fc2 = nn.Linear(1024, 8)
        self.flatten = nn.Flatten()

    def forward(self, Ibatch, CoordBatch, Ca, Cb, Pa):
        
        out = self.conv1(Ibatch)
        out = self.conv2(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.maxpool(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.maxpool(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.dropout(out)
        H4pt_predict = self.fc2(out)
        out = Tensor_DLT(H4pt_predict, Ca)
        batch_size = Ibatch.shape[0]
        Pa = Pa.view(batch_size, 1, 128, 128)
        Pa = Pa.float()
        out = out.float()
        PB_pred = kornia.geometry.transform.warp_perspective(Pa, out, dsize = (128,128),
                                                            mode='bilinear', padding_mode='zeros', 
                                                            align_corners=True).requires_grad_()

        return PB_pred, H4pt_predict

def Tensor_DLT(H4pt, C4pt_A):
    C4pt_B = H4pt + C4pt_A

    device = H4pt.device
    dtype = H4pt.dtype
    
    H_all_list = []
    values = [0, 2, 4, 6]
    for i in range(C4pt_A.shape[0]):
        rows_A = []
        rows_b = []
        for val in values:
            u_i = C4pt_A[i, val]
            v_i = C4pt_A[i, val + 1]
            u_pi = C4pt_B[i, val]
            v_pi = C4pt_B[i, val + 1]

            zero = torch.zeros((), dtype=dtype, device=device)
            one = torch.ones((), dtype=dtype, device=device)
            a = torch.stack([zero, zero, zero, -u_i, -v_i, -one, v_pi * u_i, v_i * v_pi])
            b = torch.stack([u_i, v_i, one, zero, zero, zero, -u_pi * u_i, -u_pi * v_i])
            c_val = torch.stack([-v_pi, u_pi])

            rows_A.append(a)
            rows_A.append(b)
            rows_b.append(c_val[0:1])
            rows_b.append(c_val[1:2])

        Aunderscore = torch.stack(rows_A)
        b_ = torch.stack(rows_b)

        A_inv = torch.pinverse(Aunderscore)
        h = torch.matmul(A_inv, b_)
        constant_term = torch.ones((1, 1), dtype=dtype, device=device)
        Hunderscore = torch.cat((h, constant_term), dim=0)
        H_all_list.append(Hunderscore.view(3, 3))

    H_all = torch.stack(H_all_list)
    return H_all
