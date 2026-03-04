{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-24T10:50:43.017093Z",
     "iopub.status.busy": "2026-02-24T10:50:43.016808Z",
     "iopub.status.idle": "2026-02-24T10:50:43.029026Z",
     "shell.execute_reply": "2026-02-24T10:50:43.028037Z",
     "shell.execute_reply.started": "2026-02-24T10:50:43.017069Z"
    }
   },
   "outputs": [],
   "source": [
    "class FastResidualBlock(nn.Module):\n",
    "    def __init__(self, in_ch, out_ch, norm=True):\n",
    "        super().__init__()\n",
    "\n",
    "        self.conv1 = nn.Conv3d(in_ch, out_ch, (1,3,3), padding=(0,1,1), bias=False)\n",
    "        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)\n",
    "\n",
    "        self.bn1 = nn.BatchNorm3d(out_ch) if norm else nn.Identity()\n",
    "        self.bn2 = nn.BatchNorm3d(out_ch) if norm else nn.Identity()\n",
    "\n",
    "        self.relu = nn.ReLU(inplace=True)\n",
    "\n",
    "        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch!=out_ch else nn.Identity()\n",
    "\n",
    "    def forward(self, x):\n",
    "        identity = self.skip(x)\n",
    "        x = self.relu(self.bn1(self.conv1(x)))\n",
    "        x = self.bn2(self.conv2(x))\n",
    "        out = self.relu(x + identity)\n",
    "        return out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-24T10:50:43.030441Z",
     "iopub.status.busy": "2026-02-24T10:50:43.030154Z",
     "iopub.status.idle": "2026-02-24T10:50:43.044994Z",
     "shell.execute_reply": "2026-02-24T10:50:43.044025Z",
     "shell.execute_reply.started": "2026-02-24T10:50:43.030418Z"
    }
   },
   "outputs": [],
   "source": [
    "class FastDecoderBlock(nn.Module):\n",
    "    def __init__(self, in_ch, skip_ch, out_ch):\n",
    "        super().__init__()\n",
    "\n",
    "        self.up = nn.Sequential(\n",
    "            nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False),\n",
    "            nn.Conv3d(in_ch, out_ch, 1, bias=False)\n",
    "        )\n",
    "\n",
    "        self.conv = FastResidualBlock(out_ch + skip_ch, out_ch, norm=False)\n",
    "\n",
    "    def forward(self, x, skip):\n",
    "        x = self.up(x)\n",
    "        x = torch.cat([x, skip], dim=1)\n",
    "        return self.conv(x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-24T10:50:43.046409Z",
     "iopub.status.busy": "2026-02-24T10:50:43.046132Z",
     "iopub.status.idle": "2026-02-24T10:50:43.062315Z",
     "shell.execute_reply": "2026-02-24T10:50:43.061300Z",
     "shell.execute_reply.started": "2026-02-24T10:50:43.046385Z"
    }
   },
   "outputs": [],
   "source": [
    "class FastBottleneck(nn.Module):\n",
    "    def __init__(self, ch):\n",
    "        super().__init__()\n",
    "        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1, dilation=1)\n",
    "        self.conv2 = nn.Conv3d(ch, ch, 3, padding=2, dilation=2)\n",
    "        self.conv3 = nn.Conv3d(ch, ch, 3, padding=4, dilation=4)\n",
    "        self.fuse  = nn.Conv3d(ch*3, ch, 1)\n",
    "        self.relu = nn.ReLU(inplace=True)\n",
    "\n",
    "    def forward(self, x):\n",
    "        c1 = self.conv1(x)\n",
    "        c2 = self.conv2(x)\n",
    "        c3 = self.conv3(x)\n",
    "        return x + self.fuse(self.relu(torch.cat([c1,c2,c3],1)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-02-24T10:50:43.063750Z",
     "iopub.status.busy": "2026-02-24T10:50:43.063395Z",
     "iopub.status.idle": "2026-02-24T10:50:43.073893Z",
     "shell.execute_reply": "2026-02-24T10:50:43.073002Z",
     "shell.execute_reply.started": "2026-02-24T10:50:43.063714Z"
    }
   },
   "outputs": [],
   "source": [
    "class FastResidualUNet3D(nn.Module):\n",
    "    def __init__(self, in_channels=4, num_classes=4, base_c=32):\n",
    "        super().__init__()\n",
    "        self.out_channels = num_classes \n",
    "\n",
    "        self.enc1 = FastResidualBlock(in_channels, base_c)\n",
    "        self.enc2 = FastResidualBlock(base_c, base_c*2)\n",
    "        self.enc3 = FastResidualBlock(base_c*2, base_c*4)\n",
    "        self.enc4 = FastResidualBlock(base_c*4, base_c*8)\n",
    "\n",
    "        self.pool = nn.MaxPool3d((1,2,2))\n",
    "\n",
    "        self.bottleneck = nn.Sequential(\n",
    "        FastResidualBlock(base_c*8, base_c*12),\n",
    "        FastResidualBlock(base_c*12, base_c*12),\n",
    "        FastBottleneck(base_c*12),\n",
    "        nn.Dropout3d(0.15)\n",
    "        )\n",
    "\n",
    "\n",
    "        self.dec4 = FastDecoderBlock(base_c*12, base_c*8, base_c*6)\n",
    "        self.dec3 = FastDecoderBlock(base_c*6, base_c*4, base_c*4)\n",
    "        self.dec2 = FastDecoderBlock(base_c*4, base_c*2, base_c*2)\n",
    "        self.dec1 = FastDecoderBlock(base_c*2, base_c, base_c)\n",
    "\n",
    "        self.out_conv = nn.Conv3d(base_c, num_classes, 1)\n",
    "\n",
    "    def forward(self, x):\n",
    "        e1 = self.enc1(x)\n",
    "        e2 = self.enc2(self.pool(e1))\n",
    "        e3 = self.enc3(self.pool(e2))\n",
    "        e4 = self.enc4(self.pool(e3))\n",
    "\n",
    "        b = self.bottleneck(self.pool(e4))\n",
    "\n",
    "        d4 = self.dec4(b, e4)\n",
    "        d3 = self.dec3(d4, e3)\n",
    "        d2 = self.dec2(d3, e2)\n",
    "        d1 = self.dec1(d2, e1)\n",
    "\n",
    "        out = self.out_conv(d1)\n",
    "        \n",
    "        return out"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 9221004,
     "sourceId": 14476048,
     "sourceType": "datasetVersion"
    },
    {
     "isSourceIdPinned": false,
     "modelId": 589527,
     "modelInstanceId": 577204,
     "sourceId": 764080,
     "sourceType": "modelInstanceVersion"
    }
   ],
   "dockerImageVersionId": 31236,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python [conda env:practicum]",
   "language": "python",
   "name": "conda-env-practicum-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
