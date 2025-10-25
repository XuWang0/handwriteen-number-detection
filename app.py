from flask import Flask, render_template, request, jsonify
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import io, base64, numpy as np, os

app = Flask(__name__)
MODEL_PATH = "better_mnist_cnn.pt"
EPOCHS = 3


# ---------------- Model define ----------------
class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*5*5, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- Training / Load model ----------------
def get_or_train_model():
    model = BetterCNN()

    if os.path.exists(MODEL_PATH):
        print("Loading model...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model

    print("No model detected. Training MNIST Dataset...")
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/5 loss={total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Training complete, model saved to {MODEL_PATH}")
    model.eval()
    return model


# ---------------- preprocessing ----------------
def preprocess(img_bytes):
    from PIL import ImageFilter
    image = Image.open(io.BytesIO(img_bytes)).convert("L")

    # color invert, same as MINST dataset
    image = ImageOps.invert(image)

    # 二值化去噪
    image = image.point(lambda x: 0 if x < 30 else 255, 'L')

    # 自动裁剪有效区域
    np_img = np.array(image)
    mask = np_img > 0
    if mask.any():
        ys, xs = np.where(mask)
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        image = image.crop((x1, y1, x2 + 1, y2 + 1))

    # 缩放到 20x20
    image = image.resize((20, 20), Image.LANCZOS)

    # 居中贴到 28x28
    canvas = Image.new("L", (28, 28), 0)
    left = (28 - 20) // 2
    top = (28 - 20) // 2
    canvas.paste(image, (left, top))

    # 模拟 MNIST 的灰度分布：轻微模糊 + 归一化
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(canvas).astype("float32") / 255.0
    arr = (arr - 0.1307) / 0.3081
    arr = np.expand_dims(arr, (0, 1))
    return torch.tensor(arr)


# ---------------- Flask 路由 ----------------
@app.route("/")
def index():
    return render_template("index.html")  # 对应 templates/index.html


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("image", None)
    if not data:
        return jsonify({"error": "No image received"}), 400

    image_bytes = base64.b64decode(data.split(",")[1])
    x = preprocess(image_bytes)

    with torch.no_grad():
        pred = model(x)
        probs = torch.exp(pred)[0].numpy()

    top3 = probs.argsort()[-3:][::-1]
    result = {
        "top3": [{"digit": int(i), "prob": round(float(probs[i]), 4)} for i in top3]
    }
    return jsonify(result)


if __name__ == "__main__":
    model = get_or_train_model()
    app.run(host="0.0.0.0", port=3456, debug=True)