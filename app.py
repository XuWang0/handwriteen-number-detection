from flask import Flask, render_template, request, jsonify
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import io, base64, numpy as np, os

app = Flask(__name__)
MODEL_PATH = "mnist_cnn.pt"

# -------- CNN 模型定义 --------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.25, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# -------- 加载或训练模型 --------
def get_or_train_model():
    model = CNN()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model

    print("🚀 训练模型中...")
    (x_train, y_train), _ = datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    train_loader = torch.utils.data.DataLoader(
        list(zip(x_train, y_train)), batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(2):
        for data, target in train_loader:
            data = torch.tensor(data).permute(0,3,1,2)
            target = torch.tensor(target, dtype=torch.long)
            optimizer.zero_grad()
            loss = F.nll_loss(model(data), target)
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ 模型训练完成")
    model.eval()
    return model

model = get_or_train_model()

# -------- 图像预处理 --------
def preprocess(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, (0, 1))  # (1,1,28,28)
    return torch.tensor(arr)

# -------- 路由 --------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["image"]
    image_bytes = base64.b64decode(data.split(",")[1])
    x = preprocess(image_bytes)
    with torch.no_grad():
        pred = model(x)
        probs = torch.exp(pred)[0].numpy()
    top3 = probs.argsort()[-3:][::-1]
    result = {
        "top3": [
            {"digit": int(i), "prob": round(float(probs[i]), 4)} for i in top3
        ]
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3456, debug=True)