model_rs50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# weights=VGG16_Weights.DEFAULT
for param in model_rs50.parameters():
    param.requires_grad = False

model_rs50.fc = nn.Sequential(
          # resnet50.features,
          nn.Flatten(),
          nn.Linear(2048, 256),
          nn.ReLU(inplace=True),
          nn.Dropout(p=0.5),
          nn.Linear(256, 1),
        )

criterion = nn.MSELoss()
optimizer = optim.RMSprop(model_rs50.parameters(), lr=0.001, weight_decay=1e-6)