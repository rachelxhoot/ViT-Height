vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

for param in vgg16.parameters():
    param.requires_grad = False

model_vgg16 = nn.Sequential(
          vgg16.features,
          nn.Flatten(),
          nn.Linear(2048, 256),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5),
          nn.Linear(256, 1),
        )
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model_vgg16.parameters(), lr=0.001, weight_decay=1e-6)